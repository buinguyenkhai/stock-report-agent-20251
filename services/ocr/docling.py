from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractCliOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, ImageFormatOption
from docling.datamodel.accelerator_options import AcceleratorDevice
from .base import OCRStrategy
from PIL import Image
import tempfile
import os
from typing import Any, Dict, Optional
from .inspectable_pdf_pipeline import InspectablePdfPipeline

try:
    from .hybrid_ocr_model import HybridOcrOptions
    from .hybrid_pdf_pipeline import HybridPdfPipeline
    HAS_HYBRID = True
except ImportError:
    HAS_HYBRID = False

class DoclingOCRService(OCRStrategy):
    def __init__(
        self,
        use_hybrid: bool = False,
        device: str = "cuda",
        lang: Optional[list[str]] = None,
        hybrid_confidence_threshold: float = 0.9,
        hybrid_number_confidence_threshold: float = 0.95,
        hybrid_option_overrides: Optional[Dict[str, Any]] = None,
    ):
        lang = lang or ["vie"]
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options.device = (
            AcceleratorDevice.CUDA if str(device).lower() == "cuda" else AcceleratorDevice.CPU
        )
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        # Docling table post-processing: match detected text boxes into table cells.
        # On some Docling versions this field may not exist, so guard it.
        tso = getattr(self.pipeline_options, "table_structure_options", None)
        if tso is not None and hasattr(tso, "do_cell_matching"):
            tso.do_cell_matching = True

        if use_hybrid and HAS_HYBRID:
            from .hybrid_ocr_model import HybridOcrOptions
            from .hybrid_pdf_pipeline import HybridPdfPipeline
            self.ocr_options = HybridOcrOptions(
                lang=lang,
                force_full_page_ocr=True,
                confidence_threshold=float(hybrid_confidence_threshold),
                number_confidence_threshold=float(hybrid_number_confidence_threshold),
            )
            if isinstance(hybrid_option_overrides, dict):
                for key, value in hybrid_option_overrides.items():
                    if hasattr(self.ocr_options, key):
                        setattr(self.ocr_options, key, value)
            pipeline_cls = HybridPdfPipeline
        else:
            self.ocr_options = TesseractCliOcrOptions(force_full_page_ocr=True, lang=lang)
            pipeline_cls = InspectablePdfPipeline if HAS_HYBRID else None

        self.pipeline_options.ocr_options = self.ocr_options
        
        # PDF converter
        pdf_format_option = PdfFormatOption(
            pipeline_options=self.pipeline_options,
        )
        if pipeline_cls:
            pdf_format_option.pipeline_cls = pipeline_cls

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: pdf_format_option
            }
        )
        
        # Image converter (lazy-loaded)
        self._image_converter = None
        self._last_debug_artifacts: Optional[Dict[str, Any]] = None
        self._last_reconstruction_artifacts: Optional[Dict[str, Any]] = None


    @property
    def image_converter(self):
        """Lazy-load image converter with same pipeline options."""
        if self._image_converter is None:
            self._image_converter = DocumentConverter(
                allowed_formats=[InputFormat.IMAGE],
                format_options={
                    InputFormat.IMAGE: ImageFormatOption(
                        pipeline_options=self.pipeline_options,
                    )
                }
            )
        return self._image_converter

    def _bbox_to_dict(self, bbox: Any) -> Optional[Dict[str, float]]:
        if bbox is None:
            return None
        try:
            return {
                "left": float(getattr(bbox, "l")),
                "top": float(getattr(bbox, "t")),
                "right": float(getattr(bbox, "r")),
                "bottom": float(getattr(bbox, "b")),
            }
        except Exception:
            return None

    def _extract_page_size(self, page: Any) -> Optional[list[float]]:
        size = getattr(page, "size", None)
        if size is None:
            return None
        width = getattr(size, "width", None)
        height = getattr(size, "height", None)
        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            return [float(width), float(height)]
        as_tuple = getattr(size, "as_tuple", None)
        if callable(as_tuple):
            try:
                value = as_tuple()
            except Exception:
                value = None
            if isinstance(value, tuple) and len(value) == 2:
                return [float(value[0]), float(value[1])]
        return None

    def _bbox_signature(self, bbox_obj: Dict[str, float] | None) -> tuple[int, int, int, int] | None:
        if bbox_obj is None:
            return None
        try:
            return (
                int(round(float(bbox_obj["left"]))),
                int(round(float(bbox_obj["top"]))),
                int(round(float(bbox_obj["right"]))),
                int(round(float(bbox_obj["bottom"]))),
            )
        except Exception:
            return None

    def _extract_updated_bbox_signatures(self, payload: Dict[str, Any]) -> set[tuple[int, int, int, int]]:
        accepted: set[tuple[int, int, int, int]] = set()
        diffs = payload.get("update_diffs")
        if not isinstance(diffs, list):
            return accepted
        for diff in diffs:
            if not isinstance(diff, dict) or not bool(diff.get("accepted")):
                continue
            bbox = diff.get("bbox")
            if not isinstance(bbox, dict):
                continue
            sig = self._bbox_signature(
                {
                    "left": float(bbox.get("l", 0.0) or 0.0),
                    "top": float(bbox.get("t", 0.0) or 0.0),
                    "right": float(bbox.get("r", 0.0) or 0.0),
                    "bottom": float(bbox.get("b", 0.0) or 0.0),
                }
            )
            if sig is not None:
                accepted.add(sig)
        return accepted

    def _extract_cell_tokens(self, cells: Any, *, updated_bboxes: set[tuple[int, int, int, int]] | None = None) -> list[Dict[str, Any]]:
        tokens: list[Dict[str, Any]] = []
        updated = updated_bboxes or set()
        for idx, cell in enumerate(cells or []):
            text = str(getattr(cell, "text", "") or "").strip()
            if not text:
                continue
            rect = getattr(cell, "rect", None)
            if rect is None:
                continue
            try:
                bbox = rect.to_bounding_box()
            except Exception:
                bbox = None
            bbox_obj = self._bbox_to_dict(bbox)
            if bbox_obj is None:
                continue
            line_key = getattr(cell, "_tsv_line_key", None)
            bbox_sig = self._bbox_signature(bbox_obj)
            source_tag = "surya_updated" if bbox_sig is not None and bbox_sig in updated else "baseline"
            tokens.append(
                {
                    "index": int(getattr(cell, "index", idx) or idx),
                    "text": text,
                    "left": bbox_obj["left"],
                    "top": bbox_obj["top"],
                    "right": bbox_obj["right"],
                    "bottom": bbox_obj["bottom"],
                    "confidence": float(getattr(cell, "confidence", 0.0) or 0.0),
                    "line_key": list(line_key) if isinstance(line_key, tuple) else None,
                    "from_ocr": bool(getattr(cell, "from_ocr", False)),
                    "source_tag": source_tag,
                    "ocr_region_index": int(getattr(cell, "_ocr_region_index", -1) or -1),
                }
            )
        return tokens

    def _extract_snapshot_tokens(self, cells: Any) -> list[Dict[str, Any]]:
        tokens: list[Dict[str, Any]] = []
        for idx, cell in enumerate(cells or []):
            if not isinstance(cell, dict):
                continue
            text = str(cell.get("text") or "").strip()
            bbox = cell.get("bbox")
            if not text or not isinstance(bbox, dict):
                continue
            try:
                token = {
                    "index": int(cell.get("index", idx) or idx),
                    "text": text,
                    "left": float(bbox.get("l") if "l" in bbox else bbox.get("left")),
                    "top": float(bbox.get("t") if "t" in bbox else bbox.get("top")),
                    "right": float(bbox.get("r") if "r" in bbox else bbox.get("right")),
                    "bottom": float(bbox.get("b") if "b" in bbox else bbox.get("bottom")),
                    "confidence": float(cell.get("confidence", 0.0) or 0.0),
                    "line_key": cell.get("line_key"),
                    "from_ocr": bool(cell.get("from_ocr", False)),
                    "source_tag": str(cell.get("source_tag") or "baseline"),
                    "ocr_region_index": int(cell.get("ocr_region_index", -1) or -1),
                }
            except Exception:
                continue
            tokens.append(token)
        return tokens

    def _extract_snapshot_regions(self, regions: Any) -> list[Dict[str, float]]:
        out: list[Dict[str, float]] = []
        for region in regions or []:
            if not isinstance(region, dict):
                continue
            try:
                out.append(
                    {
                        "left": float(region.get("left") if "left" in region else region.get("l")),
                        "top": float(region.get("top") if "top" in region else region.get("t")),
                        "right": float(region.get("right") if "right" in region else region.get("r")),
                        "bottom": float(region.get("bottom") if "bottom" in region else region.get("b")),
                    }
                )
            except Exception:
                continue
        return out

    def _extract_table_regions(self, page: Any) -> list[Dict[str, float]]:
        regions: list[Dict[str, float]] = []
        predictions = getattr(page, "predictions", None)
        tablestructure = getattr(predictions, "tablestructure", None)
        table_map = getattr(tablestructure, "table_map", {}) if tablestructure is not None else {}
        if isinstance(table_map, dict):
            for table in table_map.values():
                cluster = getattr(table, "cluster", None)
                bbox_obj = self._bbox_to_dict(getattr(cluster, "bbox", None))
                if bbox_obj is not None:
                    regions.append(bbox_obj)
        if regions:
            return regions

        layout = getattr(predictions, "layout", None)
        clusters = getattr(layout, "clusters", []) if layout is not None else []
        for cluster in clusters or []:
            label = str(getattr(cluster, "label", "") or "").lower()
            if "table" not in label:
                continue
            bbox_obj = self._bbox_to_dict(getattr(cluster, "bbox", None))
            if bbox_obj is not None:
                regions.append(bbox_obj)
        return regions

    def _capture_conversion_debug_artifacts(self, result: Any, *, input_format: InputFormat) -> None:
        self._last_debug_artifacts = None
        self._last_reconstruction_artifacts = None
        try:
            pipeline_get = getattr(self.converter, "_get_pipeline", None)
            if not callable(pipeline_get):
                pipeline_get = getattr(self.image_converter, "_get_pipeline", None)

            payload: Dict[str, Any] = {}
            if callable(pipeline_get):
                pipeline = pipeline_get(input_format)
                model = getattr(pipeline, "_hybrid_ocr_model", None) if pipeline is not None else None
            else:
                model = None

            if model is not None:
                stats_get = getattr(model, "get_stats", None)
                if callable(stats_get):
                    stats = stats_get()
                    if isinstance(stats, dict):
                        payload["hybrid_ocr_stats"] = stats

                small_snap_get = getattr(model, "get_debug_snapshot", None)
            if callable(small_snap_get):
                snap = small_snap_get()
                if isinstance(snap, dict):
                    payload["ocr_cells_debug"] = snap

            full_snap_get = getattr(model, "get_debug_snapshot_full", None)
            full_snapshot = full_snap_get() if callable(full_snap_get) else None

            diffs_get = getattr(model, "get_update_diffs", None)
            if callable(diffs_get):
                diffs = diffs_get()
                if isinstance(diffs, list):
                    payload["update_diffs"] = diffs

            updated_bboxes = self._extract_updated_bbox_signatures(payload)

            pages = getattr(result, "pages", None)
            page = pages[0] if isinstance(pages, list) and pages else None
            if page is not None:
                parsed_page = getattr(page, "parsed_page", None)
                if isinstance(full_snapshot, dict) and isinstance(full_snapshot.get("all_ocr_cells"), list):
                    word_tokens = self._extract_snapshot_tokens(full_snapshot.get("all_ocr_cells"))
                    ocr_regions = self._extract_snapshot_regions(full_snapshot.get("ocr_regions"))
                    textline_tokens = self._extract_cell_tokens(
                        getattr(parsed_page, "textline_cells", None),
                        updated_bboxes=updated_bboxes,
                    )
                    token_source = "ocr_region_cells"
                else:
                    textline_tokens = self._extract_cell_tokens(
                        getattr(parsed_page, "textline_cells", None),
                        updated_bboxes=updated_bboxes,
                    )
                    word_tokens = self._extract_cell_tokens(
                        getattr(parsed_page, "word_cells", None),
                        updated_bboxes=updated_bboxes,
                    )
                    ocr_regions = []
                    token_source = "parsed_page_cells"
                table_regions = self._extract_table_regions(page)
                reconstruction_artifacts = {
                    "page_size": self._extract_page_size(page),
                    "textline_tokens": textline_tokens,
                    "word_tokens": word_tokens,
                    "table_regions": table_regions,
                    "ocr_regions": ocr_regions,
                    "token_source": token_source,
                }
                self._last_reconstruction_artifacts = reconstruction_artifacts
                payload["reconstruction_debug"] = {
                    "page_size": reconstruction_artifacts["page_size"],
                    "textline_token_count": len(textline_tokens),
                    "word_token_count": len(word_tokens),
                    "table_region_count": len(table_regions),
                    "ocr_region_count": len(ocr_regions),
                    "token_source": token_source,
                }

            if payload:
                self._last_debug_artifacts = payload
        except Exception:
            self._last_debug_artifacts = None
            self._last_reconstruction_artifacts = None

    def get_debug_artifacts(self) -> Optional[Dict[str, Any]]:
        payload = self._last_debug_artifacts
        return dict(payload) if isinstance(payload, dict) else None

    def get_reconstruction_artifacts(self) -> Optional[Dict[str, Any]]:
        payload = self._last_reconstruction_artifacts
        return dict(payload) if isinstance(payload, dict) else None

    def process_pdf(self, pdf_url: str) -> str:
        """
        Process PDF and return Markdown content.
        """
        def _is_http_url(s: str) -> bool:
            return s.startswith("http://") or s.startswith("https://")

        tmp_path = None
        try:
            input_path = pdf_url
            # Docling's converter works best with a local file path.
            # If we receive a URL (e.g. Vietstock), download it first.
            if _is_http_url(pdf_url):
                import requests

                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(pdf_url, stream=True, headers=headers, timeout=60)
                resp.raise_for_status()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name
                try:
                    resp.close()
                except Exception:
                    pass
                input_path = tmp_path

            result = self.converter.convert(input_path)
            self._capture_conversion_debug_artifacts(result, input_format=InputFormat.PDF)
            return result.document.export_to_markdown()
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
    
    def process_image(self, image: Image.Image) -> str:
        """
        Process a single image and return markdown text.
        """
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        
        try:
            result = self.image_converter.convert(tmp_path)
            self._capture_conversion_debug_artifacts(result, input_format=InputFormat.IMAGE)
            return result.document.export_to_markdown()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
