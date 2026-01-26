"""Analyze OCR benchmark result JSONs.

Prints:
  - overall deltas vs docling
  - best/worst pages for hybrid vs docling
  - routing stats totals (hybrid)
  - update diffs reason breakdown (hybrid)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class PageKey:
    company: str
    page_number: int


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON at {path}")
    return obj


def _iter_pages(result: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for comp in result.get("company_results", []) or []:
        if not isinstance(comp, dict):
            continue
        for pr in comp.get("page_results", []) or []:
            if isinstance(pr, dict):
                yield pr


def _pages_by_key(result: Dict[str, Any]) -> Dict[PageKey, Dict[str, Any]]:
    out: Dict[PageKey, Dict[str, Any]] = {}
    for pr in _iter_pages(result):
        company = str(pr.get("company") or "")
        page = int(pr.get("page_number") or 0)
        if not company or page <= 0:
            continue
        out[PageKey(company=company, page_number=page)] = pr
    return out


def _fmt_float(x: Optional[float], *, digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.{digits}f}"


def _print_overall_delta(*, base: Dict[str, Any], other: Dict[str, Any], label: str) -> None:
    keys = [
        "overall_avg_number_f1",
        "overall_avg_content_word_recall",
        "overall_avg_format_agnostic_cer",
        "overall_aggregated_number_f1",
        "overall_aggregated_word_recall",
        "successful_pages",
        "total_time_seconds",
    ]
    print(f"\n== {label} vs docling ==")
    for k in keys:
        b = base.get(k)
        o = other.get(k)
        if isinstance(b, (int, float)) and isinstance(o, (int, float)):
            d = float(o) - float(b)
            print(f"{k}: {float(o):.6f} (delta {d:+.6f})")
        else:
            print(f"{k}: {o!r} (docling {b!r})")


def _sum_hybrid_stats(result: Dict[str, Any]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    for pr in _iter_pages(result):
        st = pr.get("hybrid_ocr_stats")
        if not isinstance(st, dict):
            continue
        for k, v in st.items():
            if k == "surya_percentage":
                continue
            if isinstance(v, (int, float)):
                tot[k] = tot.get(k, 0.0) + float(v)
    return tot


def _print_hybrid_stats(result: Dict[str, Any]) -> None:
    tot = _sum_hybrid_stats(result)
    if not tot:
        print("\n(no hybrid stats found)")
        return
    keys = [
        "total_cells",
        "surya_cells",
        "surya_cells_updated",
        "surya_update_skipped_sanity",
        "surya_update_skipped_non_numeric",
        "surya_update_skipped_count_mismatch",
        "routed_low_conf",
        "routed_low_num_conf",
        "inferred_table_boxes",
        "surya_failures",
    ]
    print("\n== Hybrid Routing Totals ==")
    for k in keys:
        if k in tot:
            print(f"{k}: {int(tot[k])}")


def _top_changes(
    *,
    base_pages: Dict[PageKey, Dict[str, Any]],
    other_pages: Dict[PageKey, Dict[str, Any]],
    metric: str,
    top_k: int,
) -> Tuple[List[Tuple[float, PageKey]], List[Tuple[float, PageKey]]]:
    deltas: List[Tuple[float, PageKey]] = []
    for key, b in base_pages.items():
        o = other_pages.get(key)
        if not o:
            continue
        if not bool(b.get("success", True)) or not bool(o.get("success", True)):
            continue
        bv = b.get(metric)
        ov = o.get(metric)
        if not isinstance(bv, (int, float)) or not isinstance(ov, (int, float)):
            continue
        deltas.append((float(ov) - float(bv), key))
    deltas.sort(key=lambda x: x[0])
    worst = deltas[:top_k]
    best = deltas[-top_k:][::-1]
    return best, worst


def _print_top_changes(*, base: Dict[str, Any], other: Dict[str, Any], label: str) -> None:
    base_pages = _pages_by_key(base)
    other_pages = _pages_by_key(other)
    print(f"\n== Best/Worst Pages ({label} - docling) ==")
    for metric in ("number_f1", "content_word_recall", "format_agnostic_cer"):
        best, worst = _top_changes(base_pages=base_pages, other_pages=other_pages, metric=metric, top_k=8)
        print(f"\n-- {metric} --")
        print("best:")
        for d, k in best:
            print(f"  {k.company} page {k.page_number:03d}: {d:+.4f}")
        print("worst:")
        for d, k in worst:
            print(f"  {k.company} page {k.page_number:03d}: {d:+.4f}")


def _iter_diff_files(diffs_root: Path) -> Iterable[Path]:
    for p in diffs_root.glob("*/*.json"):
        if p.name.startswith("page_") and p.suffix == ".json":
            yield p


def _diff_reason_breakdown(diffs_root: Path) -> Counter[str]:
    reasons: Counter[str] = Counter()
    for path in _iter_diff_files(diffs_root):
        try:
            obj = _load_json(path)
        except Exception:
            continue
        diffs = obj.get("update_diffs")
        if not isinstance(diffs, list):
            continue
        for d in diffs:
            if not isinstance(d, dict):
                continue
            r = str(d.get("reason") or "")
            if not r:
                r = "(missing)"
            reasons[r] += 1
    return reasons


def _print_diff_reason_breakdown(diffs_root: Optional[Path]) -> None:
    if diffs_root is None:
        return
    if not diffs_root.exists():
        print(f"\n(diffs root not found: {diffs_root})")
        return
    reasons = _diff_reason_breakdown(diffs_root)
    if not reasons:
        print(f"\n(no diffs found in {diffs_root})")
        return
    print("\n== Hybrid Update Diffs: Reason Breakdown ==")
    for reason, n in reasons.most_common(30):
        print(f"{reason}: {n}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docling", required=True, type=Path)
    ap.add_argument("--hybrid", required=False, type=Path)
    ap.add_argument("--marker", required=False, type=Path)
    ap.add_argument("--diffs-root", required=False, type=Path)
    args = ap.parse_args()

    docling = _load_json(args.docling)

    if args.hybrid:
        hybrid = _load_json(args.hybrid)
        _print_overall_delta(base=docling, other=hybrid, label="hybrid")
        _print_hybrid_stats(hybrid)
        _print_top_changes(base=docling, other=hybrid, label="hybrid")

    if args.marker:
        marker = _load_json(args.marker)
        _print_overall_delta(base=docling, other=marker, label="marker")

    _print_diff_reason_breakdown(args.diffs_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
