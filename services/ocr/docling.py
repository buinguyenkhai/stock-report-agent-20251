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
            pipeline_cls = None

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

    def _capture_hybrid_debug_artifacts(self, *, input_format: InputFormat) -> None:
        self._last_debug_artifacts = None
        try:
            pipeline_get = getattr(self.converter, "_get_pipeline", None)
            if not callable(pipeline_get):
                pipeline_get = getattr(self.image_converter, "_get_pipeline", None)
            if not callable(pipeline_get):
                return

            pipeline = pipeline_get(input_format)
            model = getattr(pipeline, "_hybrid_ocr_model", None) if pipeline is not None else None
            if model is None:
                return

            payload: Dict[str, Any] = {}

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

            diffs_get = getattr(model, "get_update_diffs", None)
            if callable(diffs_get):
                diffs = diffs_get()
                if isinstance(diffs, list):
                    payload["update_diffs"] = diffs

            if payload:
                self._last_debug_artifacts = payload
        except Exception:
            self._last_debug_artifacts = None

    def get_debug_artifacts(self) -> Optional[Dict[str, Any]]:
        payload = self._last_debug_artifacts
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

            doc = self.converter.convert(input_path).document
            self._capture_hybrid_debug_artifacts(input_format=InputFormat.PDF)
            return doc.export_to_markdown()
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
            self._capture_hybrid_debug_artifacts(input_format=InputFormat.IMAGE)
            return result.document.export_to_markdown()
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
