"""
HybridPdfPipeline - Custom Docling Pipeline with Confidence-Gated OCR

This module provides a custom PDF pipeline that extends Docling's StandardPdfPipeline
with our HybridOcrModel for confidence-gated engine routing.

Key Innovation:
- Overrides _make_ocr_model() to inject HybridOcrModel directly
- Maintains full Docling pipeline (layout, table structure, markdown)
"""

import logging
from pathlib import Path
from typing import Any, Optional

from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions

from services.ocr.hybrid_ocr_model import HybridOcrModel, HybridOcrOptions

_log = logging.getLogger(__name__)


class HybridPdfPipeline(StandardPdfPipeline):
    last_ocr_stats: dict[str, Any] | None = None

    """
    PDF pipeline with confidence-gated hybrid OCR.
    
    Extends Docling's StandardPdfPipeline to use HybridOcrModel which:
    - Runs Tesseract first (fast, gets confidence scores)
    - Routes low-confidence cells (especially numbers) to Surya
    - Preserves bounding boxes for table structure detection
    """
    
    def __init__(self, pipeline_options: ThreadedPdfPipelineOptions) -> None:
        """Initialize with standard pipeline options."""
        super().__init__(pipeline_options)
    
    def _make_ocr_model(self, art_path: Optional[Path]) -> Any:
        """
        Override to use our confidence-gated hybrid OCR model.
        """
        _log.info("Initializing HybridOcrModel for confidence-gated OCR")

        if isinstance(self.pipeline_options.ocr_options, HybridOcrOptions):
            hybrid_options = self.pipeline_options.ocr_options
        else:
            # Configure hybrid OCR options from the pipeline's ocr_options if available.
            lang = ["vie"]
            if hasattr(self.pipeline_options.ocr_options, "lang"):
                lang = self.pipeline_options.ocr_options.lang

            hybrid_options = HybridOcrOptions(
                lang=lang,
                force_full_page_ocr=getattr(
                    self.pipeline_options.ocr_options,
                    "force_full_page_ocr",
                    True,
                ),
                confidence_threshold=0.9,
                number_confidence_threshold=0.95,
                log_routing_stats=True,
            )
        
        model = HybridOcrModel(
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=art_path,
            options=hybrid_options,
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        # Expose stats to the caller via the pipeline instance.
        self._hybrid_ocr_model = model
        return model
