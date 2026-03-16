"""
Prediction generator for raw OCR benchmark v2.

Generates per-sample files expected by evaluation.benchmark_v2.run:
  <output_root>/<sample_id>.raw.md
  <output_root>/<sample_id>.ocr_debug.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from logger import get_logger
from llm_settings import DEFAULT_MARKER_LLM_MODEL
from services.ocr.base import OCRStrategy
from services.ocr.docling import DoclingOCRService
from services.ocr.financial_table_reconstruction import reconstruct_financial_table_markdown
from services.ocr.marker import MarkerOCRService

from .dataset import BenchmarkDatasetV2, IncludeScope
from .telemetry import capture_cuda_peak_memory, reset_cuda_peak_memory

logger = get_logger(__name__)

EngineName = Literal["docling", "hybrid", "marker"]


def _load_hybrid_overrides(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Hybrid overrides must be a JSON object: {path}")
    return data


def _build_ocr_service(
    *,
    engine: EngineName,
    device: str,
    hybrid_threshold: float,
    hybrid_number_threshold: float,
    hybrid_overrides: Dict[str, Any],
    marker_use_llm: bool,
    marker_llm_model: str,
    marker_force_ocr: bool,
    marker_extract_images: bool,
) -> OCRStrategy:
    if engine == "docling":
        return DoclingOCRService(use_hybrid=False, device=device)
    if engine == "hybrid":
        return DoclingOCRService(
            use_hybrid=True,
            device=device,
            hybrid_confidence_threshold=float(hybrid_threshold),
            hybrid_number_confidence_threshold=float(hybrid_number_threshold),
            hybrid_option_overrides=hybrid_overrides,
        )
    if engine == "marker":
        return MarkerOCRService(
            use_llm=bool(marker_use_llm),
            llm_model=str(marker_llm_model),
            force_ocr=bool(marker_force_ocr),
            extract_images=bool(marker_extract_images),
            device=device,
        )
    raise ValueError(f"Unsupported engine: {engine}")


def _extract_single_page_pdf(source_pdf: Path, page_index_1based: int) -> Path:
    if page_index_1based < 1:
        raise ValueError(f"Invalid page index: {page_index_1based}")
    src = fitz.open(source_pdf)
    try:
        page_zero = page_index_1based - 1
        if page_zero >= len(src):
            raise ValueError(
                f"Page index out of range for {source_pdf.name}: {page_index_1based}/{len(src)}"
            )
        dst = fitz.open()
        dst.insert_pdf(src, from_page=page_zero, to_page=page_zero)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        dst.save(tmp_path)
        dst.close()
        return tmp_path
    finally:
        src.close()


def _content_crop_image(page_image: Path, *, threshold: int = 245, margin_px: int = 18) -> Tuple[Path, Dict[str, Any]]:
    with Image.open(page_image) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        gray = ImageOps.grayscale(rgb)
        mask = gray.point(lambda p: 255 if p < threshold else 0)
        bbox = mask.getbbox()
        debug: Dict[str, Any] = {
            "page_image_path": str(page_image),
            "crop_applied": False,
            "original_size": [rgb.width, rgb.height],
        }
        cropped = rgb
        if bbox is not None:
            left, top, right, bottom = bbox
            left = max(0, left - margin_px)
            top = max(0, top - margin_px)
            right = min(rgb.width, right + margin_px)
            bottom = min(rgb.height, bottom + margin_px)
            if (left, top, right, bottom) != (0, 0, rgb.width, rgb.height):
                cropped = rgb.crop((left, top, right, bottom))
                debug["crop_applied"] = True
                debug["crop_box"] = [left, top, right, bottom]
        debug["final_size"] = [cropped.width, cropped.height]
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        cropped.save(tmp_path, "PDF", resolution=200.0)
    return tmp_path, debug


def _extract_markdown_table_stats(markdown: str) -> Dict[str, Any]:
    pipe_rows: List[List[str]] = []
    giant_cells = 0
    max_cell_chars = 0
    max_numeric_tokens = 0
    for raw_line in (markdown or "").splitlines():
        line = raw_line.strip()
        if line.count("|") < 2:
            continue
        parts = [part.strip() for part in line.split("|")]
        if parts and parts[0] == "":
            parts = parts[1:]
        if parts and parts[-1] == "":
            parts = parts[:-1]
        if not parts:
            continue
        if all((set(part.replace(":", "").strip()) <= {"-"} and "-" in part) or part == "" for part in parts):
            continue
        pipe_rows.append(parts)
        for cell in parts:
            cell_chars = len(cell)
            max_cell_chars = max(max_cell_chars, cell_chars)
            numeric_tokens = len(re.findall(r"(?<![A-Za-z])[-(]?\d[\d.,]*\)?", cell))
            max_numeric_tokens = max(max_numeric_tokens, numeric_tokens)
            if cell_chars >= 90 and numeric_tokens >= 4:
                giant_cells += 1
    stats = {
        "pipe_row_count": len(pipe_rows),
        "max_cell_chars": max_cell_chars,
        "max_numeric_tokens_in_cell": max_numeric_tokens,
        "giant_cell_count": giant_cells,
    }
    stats["structure_health_score"] = (
        stats["pipe_row_count"] * 5
        - stats["giant_cell_count"] * 18
        - max(0, stats["max_numeric_tokens_in_cell"] - 4) * 2
        - max(0, stats["max_cell_chars"] - 80) / 10.0
    )
    return stats


def _looks_structurally_collapsed(stats: Dict[str, Any]) -> bool:
    return bool(
        stats.get("pipe_row_count", 0) <= 2
        or stats.get("giant_cell_count", 0) >= 1
        or stats.get("max_numeric_tokens_in_cell", 0) >= 10
    )


def _compare_table_structures(
    baseline_stats: Dict[str, Any],
    candidate_stats: Dict[str, Any],
    *,
    candidate_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    baseline_health = float(baseline_stats.get("structure_health_score", 0.0) or 0.0)
    candidate_health = float(candidate_stats.get("structure_health_score", 0.0) or 0.0)
    baseline_collapse = _looks_structurally_collapsed(baseline_stats)
    candidate_collapse = _looks_structurally_collapsed(candidate_stats)
    candidate_debug = candidate_debug or {}
    reconstructed_rows = int(candidate_debug.get("reconstructed_row_count", 0) or 0)
    numeric_columns = int(candidate_debug.get("numeric_column_count", 0) or 0)
    header_present = bool(candidate_debug.get("header_present"))
    select_candidate = bool(
        candidate_stats.get("pipe_row_count", 0) >= 3
        and reconstructed_rows >= 3
        and numeric_columns >= 2
        and (
            header_present
            or candidate_health > baseline_health + 4.0
            or (baseline_collapse and not candidate_collapse)
        )
        and (
            candidate_health > baseline_health + 2.0
            or (baseline_collapse and not candidate_collapse)
            or (
                candidate_stats.get("giant_cell_count", 0) < baseline_stats.get("giant_cell_count", 0)
                and candidate_stats.get("max_numeric_tokens_in_cell", 0)
                <= baseline_stats.get("max_numeric_tokens_in_cell", 0)
            )
        )
    )
    return {
        "baseline_structure_health_score": baseline_health,
        "candidate_structure_health_score": candidate_health,
        "baseline_collapsed": baseline_collapse,
        "candidate_collapsed": candidate_collapse,
        "health_delta": candidate_health - baseline_health,
        "giant_cell_delta": int(candidate_stats.get("giant_cell_count", 0) or 0)
        - int(baseline_stats.get("giant_cell_count", 0) or 0),
        "pipe_row_delta": int(candidate_stats.get("pipe_row_count", 0) or 0)
        - int(baseline_stats.get("pipe_row_count", 0) or 0),
        "max_numeric_tokens_delta": int(candidate_stats.get("max_numeric_tokens_in_cell", 0) or 0)
        - int(baseline_stats.get("max_numeric_tokens_in_cell", 0) or 0),
        "select_candidate": select_candidate,
    }


def _time_call(fn) -> Tuple[Any, float]:
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000.0


def _get_ocr_debug_payload(ocr_service: OCRStrategy) -> Optional[Dict[str, Any]]:
    getter = getattr(ocr_service, "get_debug_artifacts", None)
    if not callable(getter):
        return None
    payload = getter()
    return payload if isinstance(payload, dict) else None


def _get_reconstruction_payload(ocr_service: OCRStrategy) -> Optional[Dict[str, Any]]:
    getter = getattr(ocr_service, "get_reconstruction_artifacts", None)
    if not callable(getter):
        return None
    payload = getter()
    return payload if isinstance(payload, dict) else None


def _run_ocr_for_sample(
    *,
    engine: EngineName,
    ocr_service: OCRStrategy,
    source_pdf: Optional[Path],
    page_image: Optional[Path],
    page_index: int,
    device: str,
) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    reset_cuda_peak_memory(device)
    total_started = time.perf_counter()
    stage_timings: Dict[str, Any] = {}

    if source_pdf is not None:
        one_page_pdf = _extract_single_page_pdf(source_pdf, page_index)
        try:
            markdown, stage_timings["base_ocr_latency_ms"] = _time_call(
                lambda: ocr_service.process_pdf(str(one_page_pdf))
            )
        finally:
            try:
                os.unlink(one_page_pdf)
            except Exception:
                pass

        debug: Dict[str, Any] = {
            "ocr_input_strategy": "single_page_pdf",
            "base_markdown_stats": _extract_markdown_table_stats(markdown),
        }
        chosen_markdown = markdown
        chosen_stats = debug["base_markdown_stats"]
        chosen_strategy = "single_page_pdf"

        if engine in {"docling", "hybrid"} and page_image is not None:
            base_stats = debug["base_markdown_stats"]
            if _looks_structurally_collapsed(base_stats):
                cropped_pdf = None
                try:
                    cropped_pdf, crop_debug = _content_crop_image(page_image)
                    alt_markdown, stage_timings["layout_retry_ocr_latency_ms"] = _time_call(
                        lambda: ocr_service.process_pdf(str(cropped_pdf))
                    )
                    alt_stats = _extract_markdown_table_stats(alt_markdown)
                    debug["layout_retry"] = {
                        "triggered": True,
                        "base_stats": base_stats,
                        "cropped_stats": alt_stats,
                        "crop_debug": crop_debug,
                        "selected_strategy": "single_page_pdf",
                    }
                    if alt_stats["structure_health_score"] > base_stats["structure_health_score"] + 2.0:
                        chosen_markdown = alt_markdown
                        chosen_stats = alt_stats
                        chosen_strategy = "cropped_page_image_pdf"
                        debug["ocr_input_strategy"] = chosen_strategy
                        debug["layout_retry"]["selected_strategy"] = chosen_strategy
                finally:
                    if cropped_pdf is not None:
                        try:
                            os.unlink(cropped_pdf)
                        except Exception:
                            pass

            if _looks_structurally_collapsed(chosen_stats):
                reconstructed_markdown = ""
                reconstruction_debug: Dict[str, Any] = {}
                try:
                    reconstruction_payload = _get_reconstruction_payload(ocr_service) or {}
                    page_size_raw = reconstruction_payload.get("page_size")
                    page_size = None
                    if isinstance(page_size_raw, list) and len(page_size_raw) == 2:
                        page_size = (int(page_size_raw[0]), int(page_size_raw[1]))
                    tokens = reconstruction_payload.get("word_tokens") or reconstruction_payload.get("textline_tokens")
                    (reconstructed_markdown, reconstruction_debug), stage_timings[
                        "reconstruction_latency_ms"
                    ] = _time_call(
                        lambda: reconstruct_financial_table_markdown(
                            str(page_image) if page_image is not None else None,
                            ocr_tokens=tokens if isinstance(tokens, list) else None,
                            page_size=page_size,
                            table_regions=(
                                reconstruction_payload.get("table_regions")
                                if isinstance(reconstruction_payload.get("table_regions"), list)
                                else None
                            ),
                            ocr_regions=(
                                reconstruction_payload.get("ocr_regions")
                                if isinstance(reconstruction_payload.get("ocr_regions"), list)
                                else None
                            ),
                        )
                    )
                except Exception as exc:
                    reconstruction_debug = {"reconstruction_applied": False, "error": str(exc)}
                reconstructed_stats = _extract_markdown_table_stats(reconstructed_markdown)
                reconstruction_comparison = _compare_table_structures(
                    chosen_stats,
                    reconstructed_stats,
                    candidate_debug=reconstruction_debug,
                )
                debug["deterministic_reconstruction"] = {
                    **reconstruction_debug,
                    "attempted": True,
                    "baseline_stats": chosen_stats,
                    "stats": reconstructed_stats,
                    "comparison": reconstruction_comparison,
                    "selected_strategy": chosen_strategy,
                }
                if reconstructed_markdown and reconstruction_comparison["select_candidate"]:
                    chosen_markdown = reconstructed_markdown
                    chosen_stats = reconstructed_stats
                    chosen_strategy = "deterministic_table_reconstruction"
                    debug["ocr_input_strategy"] = chosen_strategy
                    debug["deterministic_reconstruction"]["selected_strategy"] = chosen_strategy

        telemetry = {
            **capture_cuda_peak_memory(device),
            "total_latency_ms": (time.perf_counter() - total_started) * 1000.0,
            "stage_timings_ms": stage_timings,
            "selected_ocr_strategy": chosen_strategy,
        }
        return chosen_markdown, debug, telemetry

    if page_image is not None:
        if engine == "marker":
            raise ValueError("Marker requires source_pdf_path for per-page predictions")
        if not hasattr(ocr_service, "process_image"):
            raise ValueError(f"Engine does not support image OCR fallback: {engine}")
        img = Image.open(page_image)
        try:
            markdown, stage_timings["base_ocr_latency_ms"] = _time_call(
                lambda: ocr_service.process_image(img)  # type: ignore[misc]
            )
        finally:
            img.close()
        telemetry = {
            **capture_cuda_peak_memory(device),
            "total_latency_ms": (time.perf_counter() - total_started) * 1000.0,
            "stage_timings_ms": stage_timings,
            "selected_ocr_strategy": "page_image",
        }
        return markdown, {
            "ocr_input_strategy": "page_image",
            "base_markdown_stats": _extract_markdown_table_stats(markdown),
        }, telemetry

    raise ValueError("No valid OCR input for sample (need source_pdf_path or page_image_path)")


def generate_predictions(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    engine: EngineName,
    split: Literal["dev", "test", "all"],
    include_scope: IncludeScope = "all",
    skip_existing: bool = True,
    device: str = "cuda",
    hybrid_threshold: float = 0.9,
    hybrid_number_threshold: float = 0.95,
    hybrid_options_json: Optional[str] = None,
    marker_use_llm: bool = False,
    marker_llm_model: str = DEFAULT_MARKER_LLM_MODEL,
    marker_force_ocr: bool = True,
    marker_extract_images: bool = False,
) -> Dict[str, int]:
    ds = BenchmarkDatasetV2(dataset_root, include_scope=include_scope)
    required_splits = ("dev",) if split == "dev" else ("test",) if split == "test" else None
    ds.validate(check_files=False, required_splits=required_splits, require_company_disjoint=True)

    if split == "dev":
        samples = ds.get_split_samples("dev")
    elif split == "test":
        samples = ds.get_split_samples("test")
    else:
        samples = ds.get_split_samples("dev") + ds.get_split_samples("test")

    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    hybrid_overrides = _load_hybrid_overrides(hybrid_options_json)
    ocr_service = _build_ocr_service(
        engine=engine,
        device=device,
        hybrid_threshold=hybrid_threshold,
        hybrid_number_threshold=hybrid_number_threshold,
        hybrid_overrides=hybrid_overrides,
        marker_use_llm=marker_use_llm,
        marker_llm_model=marker_llm_model,
        marker_force_ocr=marker_force_ocr,
        marker_extract_images=marker_extract_images,
    )

    counts = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    errors: List[Dict[str, Any]] = []

    run_config = {
        "engine": engine,
        "split": split,
        "include_scope": include_scope,
        "device": device,
        "skip_existing": bool(skip_existing),
        "hybrid_threshold": float(hybrid_threshold),
        "hybrid_number_threshold": float(hybrid_number_threshold),
        "hybrid_overrides": hybrid_overrides,
        "marker_use_llm": bool(marker_use_llm),
        "marker_llm_model": marker_llm_model,
        "marker_force_ocr": bool(marker_force_ocr),
        "marker_extract_images": bool(marker_extract_images),
    }
    (out_root / "_predict_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for sample in samples:
        counts["total"] += 1
        raw_out = out_root / f"{sample.sample_id}.raw.md"
        ocr_debug_out = out_root / f"{sample.sample_id}.ocr_debug.json"
        if skip_existing and raw_out.exists() and ocr_debug_out.exists():
            counts["skipped"] += 1
            continue
        try:
            source_pdf = sample.resolve_path(ds.dataset_root, sample.source_pdf_path)
            page_image = sample.resolve_path(ds.dataset_root, sample.page_image_path)
            markdown, run_debug, telemetry = _run_ocr_for_sample(
                engine=engine,
                ocr_service=ocr_service,
                source_pdf=source_pdf,
                page_image=page_image,
                page_index=sample.page_index,
                device=device,
            )
            raw_out.write_text(markdown, encoding="utf-8")
            ocr_debug = _get_ocr_debug_payload(ocr_service) or {}
            ocr_debug_payload = {
                "sample_id": sample.sample_id,
                "engine": engine,
                "page_index": sample.page_index,
                **ocr_debug,
                **run_debug,
                "telemetry": telemetry,
            }
            ocr_debug_out.write_text(
                json.dumps(ocr_debug_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            counts["success"] += 1
            logger.info(f"Predicted: {sample.sample_id}")
        except Exception as exc:
            counts["failed"] += 1
            errors.append({"sample_id": sample.sample_id, "error": str(exc)})
            logger.error(f"Failed {sample.sample_id}: {exc}")
        finally:
            try:
                ocr_service.cleanup_after_page()
            except Exception:
                pass

    if errors:
        err_path = out_root / "_prediction_errors.json"
        err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning(f"Saved prediction errors to {err_path}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw OCR benchmark v2 prediction artifacts")
    parser.add_argument("--dataset-root", required=True, type=str, help="Dataset root containing manifest.json")
    parser.add_argument("--output-root", required=True, type=str, help="Directory to write prediction files")
    parser.add_argument("--engine", required=True, type=str, choices=["docling", "hybrid", "marker"])
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument(
        "--include-scope",
        type=str,
        default="all",
        choices=["all", "included", "not_included"],
        help="Filter samples using included_samples.json before split selection",
    )
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing prediction files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--hybrid-threshold", type=float, default=0.9, help="Hybrid confidence threshold")
    parser.add_argument("--hybrid-number-threshold", type=float, default=0.95)
    parser.add_argument("--hybrid-options-json", type=str, default=None)
    parser.add_argument("--marker-use-llm", action="store_true")
    parser.add_argument("--marker-llm-model", type=str, default=DEFAULT_MARKER_LLM_MODEL)
    parser.add_argument("--marker-no-force-ocr", action="store_true")
    parser.add_argument("--marker-extract-images", action="store_true")
    args = parser.parse_args()

    counts = generate_predictions(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        engine=args.engine,  # type: ignore[arg-type]
        split=args.split,  # type: ignore[arg-type]
        include_scope=args.include_scope,  # type: ignore[arg-type]
        skip_existing=not bool(args.no_skip),
        device=args.device,
        hybrid_threshold=float(args.hybrid_threshold),
        hybrid_number_threshold=float(args.hybrid_number_threshold),
        hybrid_options_json=args.hybrid_options_json,
        marker_use_llm=bool(args.marker_use_llm),
        marker_llm_model=str(args.marker_llm_model),
        marker_force_ocr=not bool(args.marker_no_force_ocr),
        marker_extract_images=bool(args.marker_extract_images),
    )
    logger.info(
        "Prediction generation done: "
        f"total={counts['total']} success={counts['success']} failed={counts['failed']} skipped={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
