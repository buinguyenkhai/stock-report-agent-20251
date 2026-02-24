"""
Prediction generator for benchmark v2.

Generates per-sample files expected by evaluation.benchmark_v2.run:
  <output_root>/<sample_id>.raw.md
  <output_root>/<sample_id>.structured.json
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import fitz  # PyMuPDF
from PIL import Image

from logger import get_logger
from services.ocr.base import OCRStrategy
from services.ocr.docling import DoclingOCRService
from services.ocr.marker import MarkerOCRService
from services.pipeline import create_pipeline

from .dataset import BenchmarkDatasetV2

logger = get_logger(__name__)

EngineName = Literal["docling", "hybrid", "marker"]


def _load_hybrid_overrides(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Hybrid overrides must be a JSON object: {p}")
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


def _run_ocr_for_sample(
    *,
    engine: EngineName,
    ocr_service: OCRStrategy,
    source_pdf: Optional[Path],
    page_image: Optional[Path],
    page_index: int,
) -> str:
    if source_pdf is not None:
        one_page_pdf = _extract_single_page_pdf(source_pdf, page_index)
        try:
            return ocr_service.process_pdf(str(one_page_pdf))
        finally:
            try:
                os.unlink(one_page_pdf)
            except Exception:
                pass

    if page_image is not None:
        if engine == "marker":
            raise ValueError("Marker requires source_pdf_path for per-page predictions")
        if not hasattr(ocr_service, "process_image"):
            raise ValueError(f"Engine does not support image OCR fallback: {engine}")
        img = Image.open(page_image)
        try:
            return ocr_service.process_image(img)  # type: ignore[misc]
        finally:
            img.close()

    raise ValueError("No valid OCR input for sample (need source_pdf_path or page_image_path)")


def generate_predictions(
    *,
    dataset_root: str | Path,
    output_root: str | Path,
    engine: EngineName,
    split: Literal["dev", "test", "all"],
    include_structured: bool = True,
    skip_existing: bool = True,
    device: str = "cuda",
    hybrid_threshold: float = 0.9,
    hybrid_number_threshold: float = 0.95,
    hybrid_options_json: Optional[str] = None,
    marker_use_llm: bool = False,
    marker_llm_model: str = "mistralai/mistral-small-3.1-24b-instruct",
    marker_force_ocr: bool = True,
    marker_extract_images: bool = False,
) -> Dict[str, int]:
    ds = BenchmarkDatasetV2(dataset_root)
    ds.validate(check_files=False)

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
    pipeline = (
        create_pipeline(mode="separate", extract_notes=False, extract_metadata=True)
        if include_structured
        else None
    )

    counts = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    errors: List[Dict[str, str]] = []

    run_config = {
        "engine": engine,
        "split": split,
        "device": device,
        "include_structured": bool(include_structured),
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

    for s in samples:
        counts["total"] += 1
        raw_out = out_root / f"{s.sample_id}.raw.md"
        struct_out = out_root / f"{s.sample_id}.structured.json"

        if skip_existing and raw_out.exists() and (not include_structured or struct_out.exists()):
            counts["skipped"] += 1
            continue

        try:
            source_pdf = s.resolve_path(ds.dataset_root, s.source_pdf_path)
            page_image = s.resolve_path(ds.dataset_root, s.page_image_path)

            md = _run_ocr_for_sample(
                engine=engine,
                ocr_service=ocr_service,
                source_pdf=source_pdf,
                page_image=page_image,
                page_index=s.page_index,
            )

            raw_out.write_text(md, encoding="utf-8")

            if include_structured and pipeline is not None:
                parsed = pipeline.process(md)
                parsed_dict = pipeline.to_dict(parsed)
                struct_out.write_text(
                    json.dumps(parsed_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            counts["success"] += 1
            logger.info(f"Predicted: {s.sample_id}")
        except Exception as e:
            counts["failed"] += 1
            errors.append({"sample_id": s.sample_id, "error": str(e)})
            logger.error(f"Failed {s.sample_id}: {e}")

    if errors:
        err_path = out_root / "_prediction_errors.json"
        err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.warning(f"Saved prediction errors to {err_path}")

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate benchmark v2 prediction artifacts")
    parser.add_argument("--dataset-root", required=True, type=str, help="Dataset root containing manifest.json")
    parser.add_argument("--output-root", required=True, type=str, help="Directory to write prediction files")
    parser.add_argument(
        "--engine",
        required=True,
        type=str,
        choices=["docling", "hybrid", "marker"],
        help="OCR engine",
    )
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--raw-only", action="store_true", help="Generate only raw markdown predictions")
    parser.add_argument("--no-skip", action="store_true", help="Do not skip existing prediction files")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--hybrid-threshold", type=float, default=0.9, help="Hybrid confidence threshold")
    parser.add_argument(
        "--hybrid-number-threshold",
        type=float,
        default=0.95,
        help="Hybrid numeric confidence threshold",
    )
    parser.add_argument(
        "--hybrid-options-json",
        type=str,
        default=None,
        help="Path to JSON object of extra HybridOcrOptions overrides",
    )
    parser.add_argument("--marker-use-llm", action="store_true", help="Enable Marker LLM post-processing")
    parser.add_argument(
        "--marker-llm-model",
        type=str,
        default="mistralai/mistral-small-3.1-24b-instruct",
        help="Marker LLM model name",
    )
    parser.add_argument("--marker-no-force-ocr", action="store_true", help="Disable Marker force_ocr")
    parser.add_argument("--marker-extract-images", action="store_true", help="Enable Marker image extraction")
    args = parser.parse_args()

    counts = generate_predictions(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        engine=args.engine,  # type: ignore[arg-type]
        split=args.split,  # type: ignore[arg-type]
        include_structured=not bool(args.raw_only),
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
