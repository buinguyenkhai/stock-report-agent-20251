from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline

from .inspectable_tesseract_ocr_model import InspectableTesseractOcrCliModel


class InspectablePdfPipeline(StandardPdfPipeline):
    def _make_ocr_model(self, art_path: Optional[Path]) -> Any:
        model = InspectableTesseractOcrCliModel(
            enabled=self.pipeline_options.do_ocr,
            artifacts_path=art_path,
            options=self.pipeline_options.ocr_options,
            accelerator_options=self.pipeline_options.accelerator_options,
        )
        self._hybrid_ocr_model = model
        return model
