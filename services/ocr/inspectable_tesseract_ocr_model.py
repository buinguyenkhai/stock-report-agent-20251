from __future__ import annotations

import copy
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, List, Optional, Type

import pandas as pd
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import OcrOptions, TesseractCliOcrOptions
from docling.utils.ocr_utils import tesseract_box_to_bounding_rectangle
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc.base import BoundingBox, CoordOrigin
from docling_core.types.doc.page import TextCell

from .hybrid_ocr_model import TesseractOcrCliModel, _parse_orientation_compat


def _bbox_obj(bbox: Optional[BoundingBox]) -> Optional[dict[str, float]]:
    if bbox is None:
        return None
    return {
        "left": float(bbox.l),
        "top": float(bbox.t),
        "right": float(bbox.r),
        "bottom": float(bbox.b),
    }


class InspectableTesseractOcrCliModel(TesseractOcrCliModel):  # type: ignore[misc]
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: TesseractCliOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self._last_snapshot: Optional[dict[str, Any]] = None

    def _cell_to_obj(self, cell: TextCell) -> dict[str, Any]:
        bbox = None
        try:
            bbox = _bbox_obj(cell.rect.to_bounding_box())
        except Exception:
            bbox = None
        region_bbox = _bbox_obj(getattr(cell, "_region_bbox", None))
        line_key = getattr(cell, "_tsv_line_key", None)
        return {
            "index": int(getattr(cell, "index", 0) or 0),
            "text": str(getattr(cell, "text", "") or ""),
            "orig": str(getattr(cell, "orig", "") or ""),
            "confidence": float(getattr(cell, "confidence", 0.0) or 0.0),
            "from_ocr": bool(getattr(cell, "from_ocr", False)),
            "bbox": bbox,
            "region_bbox": region_bbox,
            "line_key": list(line_key) if isinstance(line_key, tuple) else None,
            "ocr_region_index": int(getattr(cell, "_ocr_region_index", -1) or -1),
            "source_tag": str(getattr(cell, "_source_tag", "baseline") or "baseline"),
            "in_table_region": bool(getattr(cell, "_in_table_region", False)),
        }

    def _make_debug_snapshot(
        self,
        page: Page,
        all_ocr_cells: List[TextCell],
        ocr_rects: List[BoundingBox],
    ) -> dict[str, Any]:
        parsed = getattr(page, "parsed_page", None)
        return {
            "page_number": int(getattr(page, "page_num", 0) or 0),
            "all_ocr_cells": [self._cell_to_obj(cell) for cell in (all_ocr_cells or [])],
            "ocr_regions": [_bbox_obj(rect) for rect in (ocr_rects or []) if _bbox_obj(rect) is not None],
            "counts": {
                "all_ocr_cells": int(len(all_ocr_cells or [])),
                "ocr_regions": int(len(ocr_rects or [])),
                "parsed_textline_cells": int(len(getattr(parsed, "textline_cells", None) or []))
                if parsed is not None
                else 0,
                "parsed_word_cells": int(len(getattr(parsed, "word_cells", None) or []))
                if parsed is not None
                else 0,
            },
        }

    def get_debug_snapshot_full(self) -> Optional[dict[str, Any]]:
        return copy.deepcopy(self._last_snapshot) if isinstance(self._last_snapshot, dict) else None

    def get_debug_snapshot(self) -> Optional[dict[str, Any]]:
        snap = self.get_debug_snapshot_full()
        if not isinstance(snap, dict):
            return None
        return {
            "page_number": snap.get("page_number"),
            "counts": snap.get("counts", {}),
        }

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        if not self.enabled:
            yield from page_batch
            return

        for page_i, page in enumerate(page_batch):
            self._last_snapshot = None
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: List[TextCell] = []
                for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                    if ocr_rect.area() == 0:
                        continue

                    high_res_image = page._backend.get_page_image(scale=self.scale, cropbox=ocr_rect)
                    fname = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".png", mode="w+b", delete=False) as image_file:
                            fname = image_file.name
                            high_res_image.save(image_file)

                        doc_orientation = 0
                        df_osd: Optional[pd.DataFrame] = None
                        try:
                            df_osd = self._perform_osd(fname)
                            doc_orientation = _parse_orientation_compat(df_osd)
                        except subprocess.CalledProcessError:
                            if self._is_auto:
                                continue

                        if doc_orientation != 0:
                            high_res_image = high_res_image.rotate(-doc_orientation, expand=True)
                            high_res_image.save(fname)

                        try:
                            df_result = self._run_tesseract(fname, df_osd)
                        except subprocess.CalledProcessError:
                            continue
                    finally:
                        if fname and os.path.exists(fname):
                            os.remove(fname)

                    for ix, row in df_result.iterrows():
                        text = row["text"]
                        conf = row["conf"]
                        try:
                            block_num = int(float(row.get("block_num") or 0))
                            par_num = int(float(row.get("par_num") or 0))
                            line_num = int(float(row.get("line_num") or 0))
                        except Exception:
                            block_num, par_num, line_num = 0, 0, 0

                        left, top = float(row["left"]), float(row["top"])
                        right = left + float(row["width"])
                        bottom = top + row["height"]
                        region_bbox = BoundingBox(
                            l=left,
                            t=top,
                            r=right,
                            b=bottom,
                            coord_origin=CoordOrigin.TOPLEFT,
                        )
                        rect = tesseract_box_to_bounding_rectangle(
                            region_bbox,
                            original_offset=ocr_rect,
                            scale=self.scale,
                            orientation=doc_orientation,
                            im_size=high_res_image.size,
                        )
                        cell = TextCell(
                            index=int(ix) if isinstance(ix, (int, float, str)) else 0,
                            text=str(text),
                            orig=str(text),
                            from_ocr=True,
                            confidence=float(conf) / 100.0,
                            rect=rect,
                        )
                        setattr(cell, "_region_bbox", region_bbox)
                        setattr(cell, "_ocr_region_index", int(ocr_rect_i))
                        setattr(cell, "_source_tag", "baseline")
                        if block_num > 0 and par_num > 0 and line_num > 0:
                            setattr(cell, "_tsv_line_key", (block_num, par_num, line_num))
                        all_ocr_cells.append(cell)

                self._last_snapshot = self._make_debug_snapshot(page, all_ocr_cells, list(ocr_rects or []))
                self.post_process_cells(all_ocr_cells, page)
                yield page

    @classmethod
    def get_options_type(cls) -> Type[OcrOptions]:
        return TesseractCliOcrOptions
