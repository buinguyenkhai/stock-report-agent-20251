"""Compare two page-level benchmark result JSONs.

Verify that hybrid_docling does not regress docling_pdf on pages
where Surya is inactive, and quickly find which Surya-updated pages regress.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageKey:
    company: str
    page_number: int


def _load(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise SystemExit(f"Invalid JSON: expected object at top-level: {path}")
    return obj


def _iter_pages(payload: dict[str, Any]) -> dict[PageKey, dict[str, Any]]:
    out: dict[PageKey, dict[str, Any]] = {}
    for cr in payload.get("company_results", []) or []:
        if not isinstance(cr, dict):
            continue
        company = str(cr.get("company") or "")
        for pr in cr.get("page_results", []) or []:
            if not isinstance(pr, dict):
                continue
            key = PageKey(company=company, page_number=int(pr.get("page_number") or 0))
            out[key] = pr
    return out


def _metric(pr: dict[str, Any], name: str) -> float:
    try:
        return float(pr.get(name) or 0.0)
    except Exception:
        return 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Compare two benchmark JSON runs")
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--candidate", type=str, required=True)
    p.add_argument(
        "--only-surya-updated",
        action="store_true",
        help="Only consider pages where candidate has surya_cells_updated > 0",
    )
    p.add_argument(
        "--show",
        type=int,
        default=25,
        help="Number of rows to print (default: 25)",
    )

    args = p.parse_args()

    base = _load(Path(args.baseline))
    cand = _load(Path(args.candidate))

    base_pages = _iter_pages(base)
    cand_pages = _iter_pages(cand)

    keys = sorted(set(base_pages.keys()) & set(cand_pages.keys()), key=lambda k: (k.company, k.page_number))
    if not keys:
        raise SystemExit("No overlapping pages found between runs")

    rows: list[dict[str, Any]] = []
    for k in keys:
        b = base_pages[k]
        c = cand_pages[k]

        c_stats = c.get("hybrid_ocr_stats") if isinstance(c.get("hybrid_ocr_stats"), dict) else {}
        surya_updated = int(c_stats.get("surya_cells_updated", 0) or 0)
        if bool(args.only_surya_updated) and surya_updated <= 0:
            continue

        row = {
            "company": k.company,
            "page": k.page_number,
            "surya_updated": surya_updated,
            "d_cer": _metric(c, "format_agnostic_cer") - _metric(b, "format_agnostic_cer"),
            "d_word_recall": _metric(c, "content_word_recall") - _metric(b, "content_word_recall"),
            "d_num_f1": _metric(c, "number_f1") - _metric(b, "number_f1"),
            "base_cer": _metric(b, "format_agnostic_cer"),
            "cand_cer": _metric(c, "format_agnostic_cer"),
            "base_word_recall": _metric(b, "content_word_recall"),
            "cand_word_recall": _metric(c, "content_word_recall"),
            "base_num_f1": _metric(b, "number_f1"),
            "cand_num_f1": _metric(c, "number_f1"),
        }
        rows.append(row)

    if not rows:
        print("No matching pages after filtering")
        return

    # Sort by worst CER regression then worst recall regression.
    rows.sort(key=lambda r: (r["d_cer"], -r["d_word_recall"]), reverse=True)

    show = max(1, int(args.show))
    print(f"Compared {len(keys)} overlapping pages")
    if args.only_surya_updated:
        print(f"Showing {min(show, len(rows))}/{len(rows)} Surya-updated pages")
    else:
        print(f"Showing {min(show, len(rows))}/{len(rows)} pages")

    header = (
        "company page surya_upd | dCER dRecall dNumF1 | baseCER candCER | baseR candR | baseN candN"
    )
    print(header)
    for r in rows[:show]:
        print(
            f"{r['company']:>4} {int(r['page']):>4} {int(r['surya_updated']):>8} | "
            f"{r['d_cer']:+.4f} {r['d_word_recall']:+.3f} {r['d_num_f1']:+.3f} | "
            f"{r['base_cer']:.4f} {r['cand_cer']:.4f} | "
            f"{r['base_word_recall']:.3f} {r['cand_word_recall']:.3f} | "
            f"{r['base_num_f1']:.3f} {r['cand_num_f1']:.3f}"
        )


if __name__ == "__main__":
    main()
