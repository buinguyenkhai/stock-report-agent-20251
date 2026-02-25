"""
Render per-sample page images from source PDFs for benchmark v2 datasets.

For each manifest sample, this writes:
  <dataset_root>/<page_image_path>
from:
  <dataset_root>/<source_pdf_path> + page_index
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Literal

import fitz  # PyMuPDF

from logger import get_logger

from .dataset import BenchmarkDatasetV2

logger = get_logger(__name__)

SplitChoice = Literal["dev", "test", "all"]


def _render_page_to_image(
    *,
    doc: fitz.Document,
    source_pdf: Path,
    page_index_1based: int,
    out_path: Path,
    dpi: int,
) -> None:
    if page_index_1based < 1:
        raise ValueError(f"Invalid page index: {page_index_1based}")

    page_zero = page_index_1based - 1
    if page_zero >= len(doc):
        raise ValueError(
            f"Page index out of range for {source_pdf.name}: {page_index_1based}/{len(doc)}"
        )

    pix = doc[page_zero].get_pixmap(dpi=int(dpi), alpha=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(out_path)


def render_page_images(
    *,
    dataset_root: str | Path,
    split: SplitChoice = "all",
    dpi: int = 200,
    skip_existing: bool = True,
) -> Dict[str, int]:
    ds = BenchmarkDatasetV2(dataset_root)
    ds.validate(check_files=False)

    if split == "dev":
        samples = ds.get_split_samples("dev")
    elif split == "test":
        samples = ds.get_split_samples("test")
    else:
        samples = ds.get_split_samples("dev") + ds.get_split_samples("test")

    counts = {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    open_docs: dict[Path, fitz.Document] = {}

    try:
        for s in samples:
            counts["total"] += 1
            out_path = s.resolve_path(ds.dataset_root, s.page_image_path)
            if out_path is None:
                counts["failed"] += 1
                logger.error(f"Missing page_image_path for sample: {s.sample_id}")
                continue

            if skip_existing and out_path.exists():
                counts["skipped"] += 1
                continue

            src_path = s.resolve_path(ds.dataset_root, s.source_pdf_path)
            if src_path is None:
                counts["failed"] += 1
                logger.error(f"Missing source_pdf_path for sample: {s.sample_id}")
                continue
            if not src_path.exists():
                counts["failed"] += 1
                logger.error(f"Source PDF not found for {s.sample_id}: {src_path}")
                continue

            try:
                doc = open_docs.get(src_path)
                if doc is None:
                    doc = fitz.open(src_path)
                    open_docs[src_path] = doc

                _render_page_to_image(
                    doc=doc,
                    source_pdf=src_path,
                    page_index_1based=int(s.page_index),
                    out_path=out_path,
                    dpi=int(dpi),
                )
                counts["success"] += 1
                logger.info(f"Rendered: {s.sample_id} -> {out_path}")
            except Exception as e:
                counts["failed"] += 1
                logger.error(f"Failed {s.sample_id}: {e}")
    finally:
        for doc in open_docs.values():
            try:
                doc.close()
            except Exception:
                pass

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render benchmark v2 page images from source PDFs"
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=str,
        help="Dataset root containing manifest.json",
    )
    parser.add_argument("--split", type=str, default="all", choices=["dev", "test", "all"])
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI for output images",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-render images even if output files already exist",
    )
    args = parser.parse_args()

    counts = render_page_images(
        dataset_root=args.dataset_root,
        split=args.split,  # type: ignore[arg-type]
        dpi=int(args.dpi),
        skip_existing=not bool(args.no_skip),
    )

    logger.info(
        "Page image rendering done: "
        f"total={counts['total']} success={counts['success']} "
        f"failed={counts['failed']} skipped={counts['skipped']}"
    )


if __name__ == "__main__":
    main()
