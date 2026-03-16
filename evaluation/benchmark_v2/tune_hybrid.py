"""
Hybrid hyperparameter tuning on the raw OCR benchmark v2 dev split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from logger import get_logger

from .predict import generate_predictions
from .run import run_benchmark

logger = get_logger(__name__)


def _parse_float_list(raw: str) -> List[float]:
    vals = []
    for part in (raw or "").split(","):
        p = part.strip()
        if p:
            vals.append(float(p))
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def _objective_score(summary: Dict[str, object], objective: str) -> float:
    raw = summary.get("raw", {})
    if not isinstance(raw, dict):
        return 0.0

    def _mean(d: Dict[str, object], key: str) -> float:
        val = d.get(key, {})
        if isinstance(val, dict):
            try:
                return float(val.get("mean", 0.0))
            except Exception:
                return 0.0
        return 0.0

    if objective == "raw_number_f1":
        return _mean(raw, "number_f1")
    if objective == "table_cell_f1":
        return _mean(raw, "table_cell_f1")
    return 0.7 * _mean(raw, "number_f1") + 0.3 * _mean(raw, "table_cell_f1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune hybrid OCR thresholds on raw benchmark v2 dev split")
    parser.add_argument("--dataset-root", required=True, type=str)
    parser.add_argument("--work-root", required=True, type=str)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--hybrid-thresholds", type=str, default="0.7,0.8,0.9")
    parser.add_argument("--hybrid-number-thresholds", type=str, default="0.85,0.9,0.95")
    parser.add_argument(
        "--objective",
        type=str,
        default="blended_raw",
        choices=["blended_raw", "raw_number_f1", "table_cell_f1"],
    )
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    thresholds = _parse_float_list(args.hybrid_thresholds)
    num_thresholds = _parse_float_list(args.hybrid_number_thresholds)

    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    trials: List[Dict[str, object]] = []
    best_trial: Dict[str, object] | None = None

    for thr in thresholds:
        for nthr in num_thresholds:
            trial_name = f"hybrid_t{thr:.3f}_n{nthr:.3f}".replace(".", "p")
            pred_root = work_root / trial_name / "predictions"
            out_json = work_root / trial_name / "dev_results.json"
            pred_root.mkdir(parents=True, exist_ok=True)

            logger.info(f"Trial {trial_name}: generating predictions...")
            pred_counts = generate_predictions(
                dataset_root=args.dataset_root,
                output_root=pred_root,
                engine="hybrid",
                split="dev",
                skip_existing=False,
                device=args.device,
                hybrid_threshold=float(thr),
                hybrid_number_threshold=float(nthr),
            )

            logger.info(f"Trial {trial_name}: scoring dev split...")
            result = run_benchmark(
                dataset_root=args.dataset_root,
                predictions_root=pred_root,
                split="dev",
                engine_name=f"hybrid_t{thr:.3f}_n{nthr:.3f}",
                bootstrap_iters=int(args.bootstrap_iters),
                seed=int(args.seed),
            )
            out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            score = _objective_score(result.get("summary", {}), args.objective)
            trial = {
                "trial_name": trial_name,
                "hybrid_threshold": float(thr),
                "hybrid_number_threshold": float(nthr),
                "objective": args.objective,
                "score": float(score),
                "prediction_counts": pred_counts,
                "result_path": str(out_json),
            }
            trials.append(trial)
            if best_trial is None or float(trial["score"]) > float(best_trial["score"]):
                best_trial = trial
            logger.info(f"Trial {trial_name}: score={score:.6f}")

    summary = {
        "objective": args.objective,
        "device": args.device,
        "threshold_grid": {
            "hybrid_thresholds": thresholds,
            "hybrid_number_thresholds": num_thresholds,
        },
        "best_trial": best_trial,
        "trials": sorted(trials, key=lambda x: float(x["score"]), reverse=True),
    }
    summary_path = work_root / "tuning_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Tuning complete. Best: {best_trial}")
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
