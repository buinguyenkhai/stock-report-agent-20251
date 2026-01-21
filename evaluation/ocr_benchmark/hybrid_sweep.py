"""Hybrid Docling hyperparameter sweep runner.

Example:
  python -m evaluation.ocr_benchmark.hybrid_sweep \
    --companies AAA ACB FPT \
    --max-pages 3 \
    --confidence-thresholds 0.6 0.7 0.8 \
    --number-thresholds 0.8 0.85 0.9 \
    --outdir results/sweeps/s1
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any, Iterable

from evaluation.ocr_benchmark.page_level_benchmark import PageLevelBenchmark


def _iter_floats(xs: Iterable[str]) -> list[float]:
    out: list[float] = []
    for x in xs:
        out.append(float(x))
    return out


def run_sweep(
    *,
    companies: list[str] | None,
    max_pages: int | None,
    dpi: int,
    confidence_thresholds: list[float],
    number_thresholds: list[float],
    outdir: Path,
    minimal_json: bool,
    financial_only: bool,
    table_only: bool,
    page_offsets: dict[str, int] | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for conf_t, num_t in itertools.product(confidence_thresholds, number_thresholds):
        bench = PageLevelBenchmark(
            ocr_engine="hybrid_docling",
            dpi=dpi,
            table_only=table_only,
            financial_only=financial_only,
            hybrid_confidence_threshold=float(conf_t),
            hybrid_number_confidence_threshold=float(num_t),
            minimal_json=minimal_json,
            page_offsets=page_offsets,
        )

        result = bench.run(companies=companies, max_pages_per_company=max_pages)

        config_id = f"ct{conf_t:.3f}_nt{num_t:.3f}".replace(".", "p")
        json_path = outdir / f"{config_id}.json"
        # Use benchmark.save_results so --minimal-json prunes large payload fields.
        bench.save_results(result, json_path)

        row: dict[str, Any] = {
            "config_id": config_id,
            "dpi": dpi,
            "companies": " ".join(companies or []),
            "max_pages": max_pages if max_pages is not None else "all",
            "table_only": bool(table_only),
            "financial_only": bool(financial_only),
            "confidence_threshold": float(conf_t),
            "number_confidence_threshold": float(num_t),
            "overall_avg_cer": float(result.overall_avg_format_agnostic_cer),
            "overall_avg_word_recall": float(result.overall_avg_content_word_recall),
            "overall_avg_num_f1": float(result.overall_avg_number_f1),
            "overall_agg_word_recall": float(result.overall_aggregated_word_recall),
            "overall_agg_num_f1": float(result.overall_aggregated_number_f1),
            "overall_agg_num_precision": float(result.overall_aggregated_number_precision),
            "overall_agg_num_recall": float(result.overall_aggregated_number_recall),
            "quality_score": float(result.quality_score),
            "total_time_seconds": float(result.total_time_seconds),
        }

        rows.append(row)

    csv_path = outdir / "summary.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Hybrid Docling sweep runner")
    p.add_argument("--companies", nargs="*", default=None)
    p.add_argument("--max-pages", type=int, default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--confidence-thresholds", nargs="+", required=True)
    p.add_argument("--number-thresholds", nargs="+", required=True)
    p.add_argument("--outdir", type=str, default="results/sweeps/hybrid")
    p.add_argument("--minimal-json", action="store_true", help="Use compact JSON per run")
    p.add_argument("--financial-only", action="store_true")
    p.add_argument("--table-only", action="store_true")
    p.add_argument(
        "--page-offsets",
        nargs="*",
        type=str,
        default=None,
        help="Per-company PDF page offset(s) as CODE:INT (pdf_page = dataset_page + INT). Example: --page-offsets TCB:1",
    )

    args = p.parse_args()

    page_offsets: dict[str, int] = {}
    if args.page_offsets:
        for item in args.page_offsets:
            if not isinstance(item, str) or ":" not in item:
                raise SystemExit(f"Invalid --page-offsets entry: {item!r} (expected CODE:INT)")
            code, raw = item.split(":", 1)
            code = code.strip().upper()
            try:
                offset = int(raw.strip())
            except ValueError as e:
                raise SystemExit(f"Invalid offset for {code}: {raw!r} (expected int)") from e
            page_offsets[code] = offset

    run_sweep(
        companies=list(args.companies) if args.companies else None,
        max_pages=args.max_pages,
        dpi=int(args.dpi),
        confidence_thresholds=_iter_floats(args.confidence_thresholds),
        number_thresholds=_iter_floats(args.number_thresholds),
        outdir=Path(args.outdir),
        minimal_json=bool(args.minimal_json),
        financial_only=bool(args.financial_only),
        table_only=bool(args.table_only),
        page_offsets=page_offsets or None,
    )


if __name__ == "__main__":
    main()
