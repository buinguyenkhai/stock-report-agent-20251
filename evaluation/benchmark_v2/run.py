"""
CLI runner for benchmark v2.

This runner evaluates predictions against a manifest-defined dataset.
Prediction file convention by default:
  <predictions_root>/<sample_id>.raw.md
  <predictions_root>/<sample_id>.structured.json

Scoring policy:
- raw metrics: per-page (sample-level), table-only
- structured metrics: report-level (assembled from all pages sharing report_id)
"""

from __future__ import annotations

import argparse
import json
import random
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from logger import get_logger

from .dataset import BenchmarkDatasetV2, IncludeScope, TableSample
from .metrics_raw import RawScope, calculate_raw_metrics
from .metrics_structured import calculate_structured_metrics
from .report_assembler import assemble_report_structured_from_pages

logger = get_logger(__name__)

SplitChoice = Literal["dev", "test", "all"]
STATEMENTS = ("balance_sheet", "income_statement", "cash_flow")


@dataclass
class SampleEvalResult:
    sample_id: str
    split: str
    company: str
    page_index: int
    raw_available: bool
    structured_available: bool
    raw_metrics: Optional[Dict[str, Any]]
    structured_metrics: Optional[Dict[str, Any]]
    errors: List[str]


@dataclass
class ReportStructuredEvalResult:
    report_id: str
    split: str
    company: str
    page_count: int
    sample_ids: List[str]
    structured_available: bool
    structured_metrics: Optional[Dict[str, Any]]
    gt_merge_conflict_count: int
    pred_merge_conflict_count: int
    errors: List[str]
    gt_unit_audit: Dict[str, Any] = field(default_factory=dict)


def _read_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _mean(values: Iterable[float]) -> float:
    seq = list(values)
    return float(sum(seq) / len(seq)) if seq else 0.0


def _bootstrap_ci(
    values: List[float],
    *,
    iterations: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if len(values) == 1:
        v = float(values[0])
        return {"mean": v, "ci_low": v, "ci_high": v}

    rng = random.Random(seed)
    means: List[float] = []
    n = len(values)
    for _ in range(max(1, int(iterations))):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()

    lower_idx = int((alpha / 2) * (len(means) - 1))
    upper_idx = int((1 - alpha / 2) * (len(means) - 1))
    return {
        "mean": float(sum(values) / len(values)),
        "ci_low": float(means[lower_idx]),
        "ci_high": float(means[upper_idx]),
    }


def _normalize_text(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _normalize_text_ascii(s: str) -> str:
    text = unicodedata.normalize("NFKD", s or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("đ", "d").replace("Đ", "D")
    return _normalize_text(text)


def _coerce_numeric(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return None


def _detect_unit_scale_from_markdown(markdown: str) -> Tuple[str | None, float]:
    norm = _normalize_text_ascii(markdown)
    if not norm:
        return None, 1.0

    patterns = [
        (("ty vnd", "ty dong", "ty vietnam dong", "ty d"), 1_000_000_000.0),
        (("trieu vnd", "trieu dong", "trieu vietnam dong", "trieu d"), 1_000_000.0),
        (("nghin vnd", "nghin dong", "nghin vietnam dong", "nghin d"), 1_000.0),
        (("ngan vnd", "ngan dong", "ngan vietnam dong", "ngan d"), 1_000.0),
        (("vnd", "dong", "viet nam dong", "vietnam dong"), 1.0),
    ]
    for aliases, scale in patterns:
        for alias in aliases:
            if alias in norm:
                return alias, scale
    return None, 1.0


def _build_gt_unit_audit(
    dataset_root: Path,
    samples: List[TableSample],
) -> Dict[str, Any]:
    page_detections: List[Dict[str, Any]] = []
    seen_scales: Dict[float, int] = {}
    seen_labels: Dict[str, int] = {}
    normalized_count = 0

    for sample in samples:
        gt_md_path = dataset_root / sample.gt_markdown_path
        meta_path = dataset_root / "gt_csv" / sample.sample_id / "meta.json"
        try:
            gt_md = _read_text(gt_md_path)
        except Exception:
            continue
        label, scale = _detect_unit_scale_from_markdown(gt_md)
        normalized_to_vnd = False
        if meta_path.exists():
            try:
                meta = _read_json(meta_path)
                normalized_to_vnd = str(meta.get("value_unit_normalized_to") or "").strip().upper() == "VND"
            except Exception:
                normalized_to_vnd = False
        if normalized_to_vnd:
            normalized_count += 1
            scale = 1.0
        page_detections.append(
            {
                "sample_id": sample.sample_id,
                "page_index": sample.page_index,
                "label": label,
                "multiplier": scale,
                "normalized_to_vnd": normalized_to_vnd,
            }
        )
        seen_scales[scale] = seen_scales.get(scale, 0) + 1
        if label:
            seen_labels[label] = seen_labels.get(label, 0) + 1

    multiplier = 1.0
    label = None
    if seen_scales:
        multiplier = max(seen_scales.items(), key=lambda item: (item[1], item[0]))[0]
    if seen_labels:
        label = max(seen_labels.items(), key=lambda item: (item[1], item[0]))[0]

    return {
        "detected_multiplier": float(multiplier),
        "detected_label": label,
        "normalized_gt_to_vnd": normalized_count > 0,
        "mixed_page_units": len(seen_scales) > 1,
        "page_detections": sorted(page_detections, key=lambda row: (row["page_index"], row["sample_id"])),
    }


def _row_key(statement: str, item: Dict[str, Any], *, fallback: str) -> str:
    code = str(item.get("item_code") or "").strip()
    if code:
        return f"{statement}|code:{_normalize_text(code)}"
    name = str(item.get("item_name") or "").strip()
    if name:
        return f"{statement}|name:{_normalize_text(name)}"
    return f"{statement}|fallback:{fallback}"


def _normalized_item(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "item_code": item.get("item_code"),
        "item_name": item.get("item_name"),
        "value": item.get("value"),
        "notes_ref": item.get("notes_ref"),
        "original_name": item.get("original_name"),
    }


def _assemble_report_structured(
    pages: List[Tuple[TableSample, Dict[str, Any]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    merged: Dict[str, Any] = {
        "balance_sheet": {"items": []},
        "income_statement": {"items": []},
        "cash_flow": {"items": []},
    }
    key_index: Dict[str, Dict[str, int]] = {st: {} for st in STATEMENTS}
    conflicts: List[Dict[str, Any]] = []

    for sample, obj in sorted(pages, key=lambda x: (x[0].page_index, x[0].sample_id)):
        for statement in STATEMENTS:
            node = obj.get(statement, {})
            items = node.get("items", []) if isinstance(node, dict) else []
            if not isinstance(items, list):
                continue
            for i, raw_item in enumerate(items):
                if not isinstance(raw_item, dict):
                    continue
                item = _normalized_item(raw_item)
                key = _row_key(
                    statement,
                    item,
                    fallback=f"{sample.sample_id}:{i}",
                )
                existing_idx = key_index[statement].get(key)
                if existing_idx is None:
                    key_index[statement][key] = len(merged[statement]["items"])
                    merged[statement]["items"].append(item)
                    continue

                existing = merged[statement]["items"][existing_idx]
                old_num = _coerce_numeric(existing.get("value"))
                new_num = _coerce_numeric(item.get("value"))
                if old_num != new_num:
                    conflicts.append(
                        {
                            "sample_id": sample.sample_id,
                            "statement": statement,
                            "row_key": key,
                            "kept_value": existing.get("value"),
                            "dropped_value": item.get("value"),
                        }
                    )
                # Fill missing metadata fields from later pages without replacing value.
                for field in ("notes_ref", "original_name", "item_name", "item_code"):
                    if (existing.get(field) is None or str(existing.get(field)).strip() == "") and (
                        item.get(field) is not None and str(item.get(field)).strip() != ""
                    ):
                        existing[field] = item.get(field)

    return merged, conflicts


def _group_samples_by_report(samples: List[TableSample]) -> Dict[str, List[TableSample]]:
    out: Dict[str, List[TableSample]] = {}
    for s in samples:
        out.setdefault(s.report_id, []).append(s)
    return out


def _aggregate_results(
    sample_results: List[SampleEvalResult],
    report_results: List[ReportStructuredEvalResult],
    bootstrap_iters: int,
    seed: int,
) -> Dict[str, Any]:
    raw_fields = [
        "table_only_cer",
        "table_only_wer",
        "table_cell_f1",
        "number_f1",
    ]
    struct_fields = [
        "schema_valid",
        "row_precision",
        "row_recall",
        "row_f1",
        "value_exact_accuracy",
        "value_tolerant_accuracy",
    ]

    agg: Dict[str, Any] = {
        "counts": {
            "samples_total": len(sample_results),
            "samples_raw_scored": sum(1 for r in sample_results if r.raw_metrics is not None),
            "reports_total": len(report_results),
            "reports_structured_scored": sum(
                1 for r in report_results if r.structured_metrics is not None
            ),
            # Backward-compatible alias (structured scoring is now report-level only).
            "samples_structured_scored": sum(
                1 for r in report_results if r.structured_metrics is not None
            ),
        },
        "raw": {},
        "structured": {},
        "per_company": {},
    }

    for field in raw_fields:
        vals = [float(r.raw_metrics[field]) for r in sample_results if r.raw_metrics is not None]
        agg["raw"][field] = _bootstrap_ci(vals, iterations=bootstrap_iters, seed=seed)

    for field in struct_fields:
        vals = [
            float(r.structured_metrics[field])
            for r in report_results
            if r.structured_metrics is not None
        ]
        agg["structured"][field] = _bootstrap_ci(vals, iterations=bootstrap_iters, seed=seed)

    companies = sorted({r.company for r in sample_results} | {r.company for r in report_results})
    for c in companies:
        subset_samples = [r for r in sample_results if r.company == c]
        subset_reports = [r for r in report_results if r.company == c]
        raw_num_f1 = [
            float(r.raw_metrics["number_f1"])
            for r in subset_samples
            if r.raw_metrics is not None and "number_f1" in r.raw_metrics
        ]
        struct_row_f1 = [
            float(r.structured_metrics["row_f1"])
            for r in subset_reports
            if r.structured_metrics is not None and "row_f1" in r.structured_metrics
        ]
        agg["per_company"][c] = {
            "samples": len(subset_samples),
            "reports": len(subset_reports),
            "raw_number_f1_mean": _mean(raw_num_f1),
            "structured_row_f1_mean": _mean(struct_row_f1),
        }
    return agg


def _collect_split_samples(ds: BenchmarkDatasetV2, split: SplitChoice) -> List[TableSample]:
    if split == "dev":
        return ds.get_split_samples("dev")
    if split == "test":
        return ds.get_split_samples("test")
    return ds.get_split_samples("dev") + ds.get_split_samples("test")


def run_benchmark(
    *,
    dataset_root: str | Path,
    predictions_root: str | Path,
    split: SplitChoice = "test",
    include_scope: IncludeScope = "all",
    engine_name: str = "unknown",
    raw_suffix: str = ".raw.md",
    structured_suffix: str = ".structured.json",
    strict_missing: bool = False,
    bootstrap_iters: int = 1000,
    seed: int = 42,
    raw_scope: RawScope = "table_only",
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root, include_scope=include_scope)
    required_splits = ("dev",) if split == "dev" else ("test",) if split == "test" else None
    ds.validate(
        check_files=False,
        required_splits=required_splits,
        require_company_disjoint=True,
    )
    dataset_stats = ds.get_stats()

    pred_root = Path(predictions_root)
    split_samples = _collect_split_samples(ds, split)
    if not split_samples:
        raise ValueError(f"No samples found for split={split}")

    sample_results: List[SampleEvalResult] = []
    for s in split_samples:
        errors: List[str] = []
        gt_md_path = ds.dataset_root / s.gt_markdown_path
        pred_md_path = pred_root / f"{s.sample_id}{raw_suffix}"

        raw_metrics = None

        if pred_md_path.exists():
            try:
                gt_md = _read_text(gt_md_path)
                pred_md = _read_text(pred_md_path)
                raw_metrics = calculate_raw_metrics(pred_md, gt_md, scope=raw_scope).to_dict()
            except Exception as e:
                errors.append(f"raw_metrics_error: {e}")
        else:
            msg = f"missing_raw_prediction: {pred_md_path}"
            if strict_missing:
                raise FileNotFoundError(msg)
            errors.append(msg)

        sample_results.append(
            SampleEvalResult(
                sample_id=s.sample_id,
                split=s.split,
                company=s.company,
                page_index=s.page_index,
                raw_available=pred_md_path.exists(),
                structured_available=False,
                raw_metrics=raw_metrics,
                structured_metrics=None,
                errors=errors,
            )
        )

    report_results: List[ReportStructuredEvalResult] = []
    by_report = _group_samples_by_report(split_samples)
    for report_id, report_samples in sorted(by_report.items()):
        report_errors: List[str] = []
        gt_pages: List[Tuple[TableSample, Dict[str, Any]]] = []
        pred_pages: List[Tuple[TableSample, Dict[str, Any]]] = []

        sorted_samples = sorted(report_samples, key=lambda x: (x.page_index, x.sample_id))
        company = sorted_samples[0].company
        split_name = sorted_samples[0].split

        for s in sorted_samples:
            gt_struct_path = ds.dataset_root / s.gt_structured_path
            pred_struct_path = pred_root / f"{s.sample_id}{structured_suffix}"

            try:
                gt_obj = _read_json(gt_struct_path)
                gt_pages.append((s, gt_obj))
            except Exception as e:
                report_errors.append(f"gt_structured_error {s.sample_id}: {e}")

            if pred_struct_path.exists():
                try:
                    pred_obj = _read_json(pred_struct_path)
                    pred_pages.append((s, pred_obj))
                except Exception as e:
                    report_errors.append(f"pred_structured_error {s.sample_id}: {e}")
            else:
                msg = f"missing_structured_prediction: {pred_struct_path}"
                if strict_missing:
                    raise FileNotFoundError(msg)
                report_errors.append(msg)

        struct_metrics = None
        gt_conflicts: List[Dict[str, Any]] = []
        pred_conflicts: List[Dict[str, Any]] = []
        gt_unit_audit = _build_gt_unit_audit(ds.dataset_root, sorted_samples)
        structured_available = len(pred_pages) == len(sorted_samples) and len(pred_pages) > 0
        if not report_errors and structured_available:
            gt_assembled, gt_meta = assemble_report_structured_from_pages(gt_pages)
            pred_assembled, pred_meta = assemble_report_structured_from_pages(pred_pages)
            gt_conflicts = list(gt_meta.get("conflicts", [])) if isinstance(gt_meta, dict) else []
            pred_conflicts = list(pred_meta.get("conflicts", [])) if isinstance(pred_meta, dict) else []
            struct_metrics = calculate_structured_metrics(
                pred_assembled,
                gt_assembled,
            ).to_dict()

        report_results.append(
            ReportStructuredEvalResult(
                report_id=report_id,
                split=split_name,
                company=company,
                page_count=len(sorted_samples),
                sample_ids=[s.sample_id for s in sorted_samples],
                structured_available=structured_available,
                structured_metrics=struct_metrics,
                gt_merge_conflict_count=len(gt_conflicts),
                pred_merge_conflict_count=len(pred_conflicts),
                gt_unit_audit=gt_unit_audit,
                errors=report_errors,
            )
        )

    summary = _aggregate_results(
        sample_results,
        report_results,
        bootstrap_iters=bootstrap_iters,
        seed=seed,
    )
    payload = {
        "benchmark_version": "v2",
        "engine_name": engine_name,
        "split": split,
        "include_scope": include_scope,
        "raw_scope": raw_scope,
        "structured_scope": "report_only",
        "dataset_root": str(Path(dataset_root).resolve()),
        "predictions_root": str(pred_root.resolve()),
        "dataset_stats": dataset_stats,
        "summary": summary,
        "sample_results": [asdict(r) for r in sample_results],
        "report_structured_results": [asdict(r) for r in report_results],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark v2 (dev/test) from manifest and predictions")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root containing manifest.json")
    parser.add_argument("--predictions-root", type=str, required=True, help="Path to model prediction files")
    parser.add_argument("--engine-name", type=str, default="unknown", help="Name of evaluated OCR/Extraction system")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"], help="Split to evaluate")
    parser.add_argument(
        "--include-scope",
        type=str,
        default="all",
        choices=["all", "included", "not_included"],
        help="Filter samples using included_samples.json before split selection",
    )
    parser.add_argument("--raw-suffix", type=str, default=".raw.md", help="Per-sample raw markdown prediction suffix")
    parser.add_argument("--structured-suffix", type=str, default=".structured.json", help="Per-sample structured prediction suffix")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if any prediction file is missing")
    parser.add_argument("--bootstrap-iters", type=int, default=1000, help="Bootstrap iterations for confidence intervals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--raw-scope",
        type=str,
        default="table_only",
        choices=["table_only"],
        help="Raw OCR scoring scope",
    )
    parser.add_argument("--output", type=str, default="results/benchmark_v2_results.json", help="Output JSON file path")
    args = parser.parse_args()

    result = run_benchmark(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        split=args.split,
        include_scope=args.include_scope,  # type: ignore[arg-type]
        engine_name=args.engine_name,
        raw_suffix=args.raw_suffix,
        structured_suffix=args.structured_suffix,
        strict_missing=bool(args.strict_missing),
        bootstrap_iters=int(args.bootstrap_iters),
        seed=int(args.seed),
        raw_scope=args.raw_scope,  # type: ignore[arg-type]
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    counts = result["summary"]["counts"]
    logger.info("Benchmark v2 complete")
    logger.info(
        f"Samples: total={counts['samples_total']} raw_scored={counts['samples_raw_scored']} "
        f"structured_scored={counts['samples_structured_scored']}"
    )
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
