"""
CLI runner for raw OCR benchmark v2.

Prediction file convention by default:
  <predictions_root>/<sample_id>.raw.md
  <predictions_root>/<sample_id>.ocr_debug.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from logger import get_logger

from .dataset import BenchmarkDatasetV2, IncludeScope, TableSample
from .metrics_raw import RawScope, calculate_raw_metrics
from .telemetry import collect_numeric, summarize_numeric

logger = get_logger(__name__)

SplitChoice = Literal["dev", "test", "all"]


@dataclass
class SampleEvalResult:
    sample_id: str
    split: str
    company: str
    report_id: str
    page_index: int
    raw_available: bool
    raw_metrics: Optional[Dict[str, Any]]
    ocr_debug: Optional[Dict[str, Any]] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)
    ocr_debug_available: bool = False
    errors: List[str] = field(default_factory=list)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _read_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
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


def _collect_split_samples(ds: BenchmarkDatasetV2, split: SplitChoice) -> List[TableSample]:
    if split == "dev":
        return ds.get_split_samples("dev")
    if split == "test":
        return ds.get_split_samples("test")
    return ds.get_split_samples("dev") + ds.get_split_samples("test")


def _hardest_pages(sample_results: List[SampleEvalResult], field: str, *, limit: int = 10) -> List[Dict[str, Any]]:
    rows = [
        {
            "sample_id": result.sample_id,
            "company": result.company,
            "report_id": result.report_id,
            "page_index": result.page_index,
            "value": float(result.raw_metrics[field]),
        }
        for result in sample_results
        if result.raw_metrics is not None and field in result.raw_metrics
    ]
    return sorted(rows, key=lambda row: row["value"])[:limit]


def _aggregate_results(
    sample_results: List[SampleEvalResult],
    *,
    bootstrap_iters: int,
    seed: int,
) -> Dict[str, Any]:
    raw_fields = ["table_only_cer", "table_only_wer", "table_cell_f1", "number_f1"]
    agg: Dict[str, Any] = {
        "counts": {
            "samples_total": len(sample_results),
            "samples_raw_scored": sum(1 for r in sample_results if r.raw_metrics is not None),
            "samples_with_ocr_debug": sum(1 for r in sample_results if r.ocr_debug_available),
            "samples_failed": sum(1 for r in sample_results if r.errors),
            "completion_rate": (
                sum(1 for r in sample_results if r.raw_metrics is not None) / len(sample_results)
                if sample_results
                else 0.0
            ),
        },
        "raw": {},
        "telemetry": {},
        "per_company": {},
        "hardest_pages": {
            "table_cell_f1": _hardest_pages(sample_results, "table_cell_f1"),
            "number_f1": _hardest_pages(sample_results, "number_f1"),
        },
    }
    for field in raw_fields:
        vals = [float(r.raw_metrics[field]) for r in sample_results if r.raw_metrics is not None]
        agg["raw"][field] = _bootstrap_ci(vals, iterations=bootstrap_iters, seed=seed)

    latency_values = collect_numeric((r.telemetry for r in sample_results), "total_latency_ms")
    reserved_values = collect_numeric((r.telemetry for r in sample_results), "peak_vram_reserved_mb")
    allocated_values = collect_numeric((r.telemetry for r in sample_results), "peak_vram_allocated_mb")
    agg["telemetry"] = {
        "latency_ms": summarize_numeric(latency_values),
        "peak_vram_reserved_mb": summarize_numeric(reserved_values),
        "peak_vram_allocated_mb": summarize_numeric(allocated_values),
        "cuda_sample_count": sum(1 for r in sample_results if r.telemetry.get("cuda_enabled") is True),
    }

    companies = sorted({r.company for r in sample_results})
    for company in companies:
        subset = [r for r in sample_results if r.company == company]
        raw_num_f1 = [
            float(r.raw_metrics["number_f1"])
            for r in subset
            if r.raw_metrics is not None and "number_f1" in r.raw_metrics
        ]
        cell_f1 = [
            float(r.raw_metrics["table_cell_f1"])
            for r in subset
            if r.raw_metrics is not None and "table_cell_f1" in r.raw_metrics
        ]
        latency = collect_numeric((r.telemetry for r in subset), "total_latency_ms")
        agg["per_company"][company] = {
            "samples": len(subset),
            "raw_number_f1_mean": _mean(raw_num_f1),
            "table_cell_f1_mean": _mean(cell_f1),
            "latency_ms_mean": _mean(latency),
        }
    return agg


def run_benchmark(
    *,
    dataset_root: str | Path,
    predictions_root: str | Path,
    split: SplitChoice = "test",
    include_scope: IncludeScope = "all",
    engine_name: str = "unknown",
    raw_suffix: str = ".raw.md",
    strict_missing: bool = False,
    bootstrap_iters: int = 1000,
    seed: int = 42,
    raw_scope: RawScope = "table_only",
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root, include_scope=include_scope)
    required_splits = ("dev",) if split == "dev" else ("test",) if split == "test" else None
    ds.validate(check_files=False, required_splits=required_splits, require_company_disjoint=True)
    dataset_stats = ds.get_stats()

    pred_root = Path(predictions_root)
    split_samples = _collect_split_samples(ds, split)
    if not split_samples:
        raise ValueError(f"No samples found for split={split}")

    sample_results: List[SampleEvalResult] = []
    for sample in split_samples:
        errors: List[str] = []
        gt_md_path = ds.dataset_root / sample.gt_markdown_path
        pred_md_path = pred_root / f"{sample.sample_id}{raw_suffix}"
        debug_path = pred_root / f"{sample.sample_id}.ocr_debug.json"

        raw_metrics = None
        telemetry: Dict[str, Any] = {}
        debug_payload: Optional[Dict[str, Any]] = None
        if pred_md_path.exists():
            try:
                gt_md = _read_text(gt_md_path)
                pred_md = _read_text(pred_md_path)
                raw_metrics = calculate_raw_metrics(pred_md, gt_md, scope=raw_scope).to_dict()
            except Exception as exc:
                errors.append(f"raw_metrics_error: {exc}")
        else:
            msg = f"missing_raw_prediction: {pred_md_path}"
            if strict_missing:
                raise FileNotFoundError(msg)
            errors.append(msg)

        if debug_path.exists():
            try:
                debug_payload = _read_json(debug_path)
                telemetry = dict(debug_payload.get("telemetry") or {})
            except Exception as exc:
                errors.append(f"ocr_debug_error: {exc}")

        sample_results.append(
            SampleEvalResult(
                sample_id=sample.sample_id,
                split=sample.split,
                company=sample.company,
                report_id=sample.report_id,
                page_index=sample.page_index,
                raw_available=pred_md_path.exists(),
                raw_metrics=raw_metrics,
                ocr_debug=debug_payload,
                telemetry=telemetry,
                ocr_debug_available=debug_payload is not None,
                errors=errors,
            )
        )

    summary = _aggregate_results(sample_results, bootstrap_iters=bootstrap_iters, seed=seed)
    return {
        "benchmark_version": "v2_raw_only",
        "engine_name": engine_name,
        "split": split,
        "include_scope": include_scope,
        "raw_scope": raw_scope,
        "dataset_root": str(Path(dataset_root).resolve()),
        "predictions_root": str(pred_root.resolve()),
        "dataset_stats": dataset_stats,
        "summary": summary,
        "sample_results": [asdict(result) for result in sample_results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run raw OCR benchmark v2 from manifest and predictions")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--predictions-root", type=str, required=True)
    parser.add_argument("--engine-name", type=str, default="unknown")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument(
        "--include-scope",
        type=str,
        default="all",
        choices=["all", "included", "not_included"],
    )
    parser.add_argument("--raw-suffix", type=str, default=".raw.md")
    parser.add_argument("--strict-missing", action="store_true")
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-scope", type=str, default="table_only", choices=["table_only"])
    parser.add_argument("--output", type=str, default="results/benchmark_v2_results.json")
    args = parser.parse_args()

    result = run_benchmark(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        split=args.split,
        include_scope=args.include_scope,  # type: ignore[arg-type]
        engine_name=args.engine_name,
        raw_suffix=args.raw_suffix,
        strict_missing=bool(args.strict_missing),
        bootstrap_iters=int(args.bootstrap_iters),
        seed=int(args.seed),
        raw_scope=args.raw_scope,  # type: ignore[arg-type]
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = result["summary"]["counts"]
    logger.info("Benchmark v2 raw-only complete")
    logger.info(
        f"Samples: total={counts['samples_total']} raw_scored={counts['samples_raw_scored']} failed={counts['samples_failed']}"
    )
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
