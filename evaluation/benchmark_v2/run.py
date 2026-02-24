"""
CLI runner for benchmark v2.

This runner evaluates predictions against a manifest-defined dataset.
Prediction file convention by default:
  <predictions_root>/<sample_id>.raw.md
  <predictions_root>/<sample_id>.structured.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from logger import get_logger

from .dataset import BenchmarkDatasetV2, TableSample
from .metrics_raw import calculate_raw_metrics
from .metrics_structured import calculate_structured_metrics

logger = get_logger(__name__)

SplitChoice = Literal["dev", "test", "all"]


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


def _aggregate_results(results: List[SampleEvalResult], bootstrap_iters: int, seed: int) -> Dict[str, Any]:
    raw_fields = [
        "format_agnostic_cer",
        "format_agnostic_wer",
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
            "samples_total": len(results),
            "samples_raw_scored": sum(1 for r in results if r.raw_metrics is not None),
            "samples_structured_scored": sum(1 for r in results if r.structured_metrics is not None),
        },
        "raw": {},
        "structured": {},
        "per_company": {},
    }

    for field in raw_fields:
        vals = [float(r.raw_metrics[field]) for r in results if r.raw_metrics is not None]
        agg["raw"][field] = _bootstrap_ci(vals, iterations=bootstrap_iters, seed=seed)

    for field in struct_fields:
        vals = [float(r.structured_metrics[field]) for r in results if r.structured_metrics is not None]
        agg["structured"][field] = _bootstrap_ci(vals, iterations=bootstrap_iters, seed=seed)

    companies = sorted({r.company for r in results})
    for c in companies:
        subset = [r for r in results if r.company == c]
        raw_num_f1 = [
            float(r.raw_metrics["number_f1"])
            for r in subset
            if r.raw_metrics is not None and "number_f1" in r.raw_metrics
        ]
        struct_row_f1 = [
            float(r.structured_metrics["row_f1"])
            for r in subset
            if r.structured_metrics is not None and "row_f1" in r.structured_metrics
        ]
        agg["per_company"][c] = {
            "samples": len(subset),
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
    engine_name: str = "unknown",
    raw_suffix: str = ".raw.md",
    structured_suffix: str = ".structured.json",
    strict_missing: bool = False,
    bootstrap_iters: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    ds = BenchmarkDatasetV2(dataset_root)
    ds.validate(check_files=False)
    dataset_stats = ds.get_stats()

    pred_root = Path(predictions_root)
    split_samples = _collect_split_samples(ds, split)
    if not split_samples:
        raise ValueError(f"No samples found for split={split}")

    sample_results: List[SampleEvalResult] = []
    for s in split_samples:
        errors: List[str] = []
        gt_md_path = ds.dataset_root / s.gt_markdown_path
        gt_struct_path = ds.dataset_root / s.gt_structured_path
        pred_md_path = pred_root / f"{s.sample_id}{raw_suffix}"
        pred_struct_path = pred_root / f"{s.sample_id}{structured_suffix}"

        raw_metrics = None
        struct_metrics = None

        if pred_md_path.exists():
            try:
                gt_md = _read_text(gt_md_path)
                pred_md = _read_text(pred_md_path)
                raw_metrics = calculate_raw_metrics(pred_md, gt_md).to_dict()
            except Exception as e:
                errors.append(f"raw_metrics_error: {e}")
        else:
            msg = f"missing_raw_prediction: {pred_md_path}"
            if strict_missing:
                raise FileNotFoundError(msg)
            errors.append(msg)

        if pred_struct_path.exists():
            try:
                gt_struct = _read_json(gt_struct_path)
                pred_struct = _read_json(pred_struct_path)
                struct_metrics = calculate_structured_metrics(pred_struct, gt_struct).to_dict()
            except Exception as e:
                errors.append(f"structured_metrics_error: {e}")
        else:
            msg = f"missing_structured_prediction: {pred_struct_path}"
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
                structured_available=pred_struct_path.exists(),
                raw_metrics=raw_metrics,
                structured_metrics=struct_metrics,
                errors=errors,
            )
        )

    summary = _aggregate_results(sample_results, bootstrap_iters=bootstrap_iters, seed=seed)
    payload = {
        "benchmark_version": "v2",
        "engine_name": engine_name,
        "split": split,
        "dataset_root": str(Path(dataset_root).resolve()),
        "predictions_root": str(pred_root.resolve()),
        "dataset_stats": dataset_stats,
        "summary": summary,
        "sample_results": [asdict(r) for r in sample_results],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark v2 (dev/test) from manifest and predictions")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to dataset root containing manifest.json")
    parser.add_argument("--predictions-root", type=str, required=True, help="Path to model prediction files")
    parser.add_argument("--engine-name", type=str, default="unknown", help="Name of evaluated OCR/Extraction system")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"], help="Split to evaluate")
    parser.add_argument("--raw-suffix", type=str, default=".raw.md", help="Per-sample raw markdown prediction suffix")
    parser.add_argument("--structured-suffix", type=str, default=".structured.json", help="Per-sample structured prediction suffix")
    parser.add_argument("--strict-missing", action="store_true", help="Fail if any prediction file is missing")
    parser.add_argument("--bootstrap-iters", type=int, default=1000, help="Bootstrap iterations for confidence intervals")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/benchmark_v2_results.json", help="Output JSON file path")
    args = parser.parse_args()

    result = run_benchmark(
        dataset_root=args.dataset_root,
        predictions_root=args.predictions_root,
        split=args.split,
        engine_name=args.engine_name,
        raw_suffix=args.raw_suffix,
        structured_suffix=args.structured_suffix,
        strict_missing=bool(args.strict_missing),
        bootstrap_iters=int(args.bootstrap_iters),
        seed=int(args.seed),
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

