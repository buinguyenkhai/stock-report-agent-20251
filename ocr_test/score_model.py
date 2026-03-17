from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.benchmark_v2.metrics_raw import calculate_raw_metrics


SAMPLES = [
    "MWG_Q2_2023_p008",
    "TCB_Q3_2024_p006",
    "VCB_Q1_2022_p004",
    "VIC_Q4_2021_p009",
]


def _load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sample_record(dataset_root: Path, predictions_root: Path, sample_id: str) -> dict:
    gt_path = dataset_root / "gt_markdown" / f"{sample_id}.md"
    pred_path = predictions_root / f"{sample_id}.md"
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction markdown: {pred_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing GT markdown: {gt_path}")

    metrics = calculate_raw_metrics(
        hypothesis_markdown=_load_text(pred_path),
        reference_markdown=_load_text(gt_path),
    ).to_dict()

    return {
        "sample_id": sample_id,
        "prediction_path": str(pred_path),
        "gt_markdown_path": str(gt_path),
        "raw_metrics": metrics,
    }


def _summary(sample_results: list[dict]) -> dict:
    metric_keys = [
        "table_only_cer",
        "table_only_wer",
        "table_cell_f1",
        "number_f1",
        "number_precision",
        "number_recall",
    ]
    return {
        "samples_total": len(sample_results),
        "raw": {
            key: mean(float(row["raw_metrics"][key]) for row in sample_results)
            for key in metric_keys
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score ocr_test/<model_name> markdowns with benchmark v2 raw metrics."
    )
    parser.add_argument("model_name", help="Directory name under ocr_test/")
    parser.add_argument(
        "--dataset-root",
        default=str(REPO_ROOT / "data" / "benchmark_v2"),
        help="Path to benchmark_v2 dataset root",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON output path. Defaults to ocr_test/<model_name>/metrics.json",
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    predictions_root = (REPO_ROOT / "ocr_test" / args.model_name).resolve()
    output_path = (
        Path(args.output).resolve()
        if args.output
        else predictions_root / "metrics.json"
    )

    sample_results = [
        _sample_record(dataset_root=dataset_root, predictions_root=predictions_root, sample_id=sample_id)
        for sample_id in SAMPLES
    ]
    result = {
        "benchmark_version": "benchmark_v2_smoke",
        "model_name": args.model_name,
        "dataset_root": str(dataset_root),
        "predictions_root": str(predictions_root),
        "sample_results": sample_results,
        "summary": _summary(sample_results),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {output_path}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("\nPer-page:")
    for row in sample_results:
        m = row["raw_metrics"]
        print(
            f"- {row['sample_id']}: "
            f"CER={m['table_only_cer']:.4f} "
            f"WER={m['table_only_wer']:.4f} "
            f"CellF1={m['table_cell_f1']:.4f} "
            f"NumF1={m['number_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
