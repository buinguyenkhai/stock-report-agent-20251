"""
HybridOcrModel - Confidence-Gated Dual-Engine OCR for Docling

This module provides a drop-in replacement for Docling's TesseractOcrCliModel
that intelligently routes low-confidence OCR cells to Surya for re-OCR.

Key Innovation:
- Uses Tesseract's per-word confidence scores to identify unreliable OCR
- Routes low-confidence cells (especially numbers) to Surya for higher accuracy
- Preserves bounding boxes for correct table structure detection
"""

import logging
import re
import gc
from typing import ClassVar, List, Literal, Optional, Iterable, Type
from collections.abc import Iterable as IterableABC
from pathlib import Path

from docling_core.types.doc.page import TextCell
from PIL import Image
from pydantic import ConfigDict

from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import Page
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import TesseractCliOcrOptions
from docling.models.tesseract_ocr_cli_model import TesseractOcrCliModel
from docling.utils.profiling import TimeRecorder

_log = logging.getLogger(__name__)


# Regex pattern to detect numbers in text
NUMBER_PATTERN = re.compile(r'[\d.,]+')


def contains_numbers(text: str) -> bool:
    """Check if text contains numerical content."""
    if not text:
        return False
    return bool(NUMBER_PATTERN.search(text))


class HybridOcrOptions(TesseractCliOcrOptions):
    """
    Extended options for Hybrid OCR with confidence-gated routing.
    
    Inherits all TesseractCliOcrOptions and adds routing thresholds.
    """
    kind: ClassVar[Literal["hybrid"]] = "hybrid"
    
    # Confidence thresholds (0.0 - 1.0)
    confidence_threshold: float = 0.7
    number_confidence_threshold: float = 0.85
    
    # Force Surya for all number-containing cells regardless of confidence
    force_surya_for_numbers: bool = False
    
    # Surya batch size for re-OCR
    surya_batch_size: int = 32
    
    # Logging
    log_routing_stats: bool = True
    
    model_config = ConfigDict(
        extra="forbid",
    )


class HybridOcrModel(TesseractOcrCliModel):
    """
    Confidence-Gated Hybrid OCR Model for Docling.
    
    Extends TesseractOcrCliModel with intelligent routing:
    - Runs Tesseract first (fast, gets bounding boxes + confidence)
    - Filters low-confidence cells
    - Re-OCRs those cells with Surya (more accurate)
    - Returns enhanced cells with original bboxes preserved
    """
    
    def __init__(
        self,
        enabled: bool,
        artifacts_path: Optional[Path],
        options: HybridOcrOptions,
        accelerator_options: AcceleratorOptions,
    ):
        # Initialize parent class (TesseractOcrCliModel)
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        
        # Store hybrid-specific options
        self.hybrid_options = options
        
        # Lazy-loaded Surya model
        self._surya_model = None
        self._surya_foundation = None
        
        # Routing statistics
        self._stats = {
            'total_cells': 0,
            'surya_cells': 0,
            'tesseract_cells': 0,
        }
    
    @property
    def surya_model(self):
        """Lazy load Surya recognition model."""
        if self._surya_model is None:
            _log.info("Loading Surya recognition model for hybrid OCR...")
            try:
                import torch
                from surya.foundation import FoundationPredictor
                from surya.recognition import RecognitionPredictor
                from surya.settings import settings as surya_settings
                
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self._surya_foundation = FoundationPredictor(
                    checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                    device=device,
                )
                self._surya_model = RecognitionPredictor(self._surya_foundation)
                _log.info(f"Surya model loaded on {device}")
            except Exception as e:
                _log.error(f"Failed to load Surya model: {e}")
                raise
        return self._surya_model
    
    def _should_route_to_surya(self, cell: TextCell) -> bool:
        """
        Determine if a cell should be re-OCR'd by Surya.
        
        Returns True if:
        - Cell contains numbers AND confidence < number_threshold
        - OR force_surya_for_numbers is True AND cell has numbers
        - OR confidence < general_threshold
        """
        confidence = cell.confidence or 0.0
        has_numbers = contains_numbers(cell.text)
        
        # Force Surya for all numbers if enabled
        if has_numbers and self.hybrid_options.force_surya_for_numbers:
            return True
        
        # Use stricter threshold for numbers
        threshold = (
            self.hybrid_options.number_confidence_threshold 
            if has_numbers 
            else self.hybrid_options.confidence_threshold
        )
        
        return confidence < threshold
    
    def _surya_reocr_cells(
        self, 
        page_image: Image.Image, 
        cells: List[TextCell],
        scale: float,
    ) -> None:
        """
        Re-OCR cells using Surya and update their text in-place.
        
        Args:
            page_image: High-resolution page image
            cells: List of TextCell objects to re-OCR
            scale: Scale factor applied to the image
        """
        if not cells:
            return
        
        try:
            import torch
            
            # Prepare polygons for Surya (scaled coordinates)
            polygons = []
            for cell in cells:
                bbox = cell.rect.to_bounding_box()
                # Scale coordinates to match high-res image
                l = int(bbox.l * scale)
                t = int(bbox.t * scale)
                r = int(bbox.r * scale)
                b = int(bbox.b * scale)
                polygon = [[l, t], [r, t], [r, b], [l, b]]
                polygons.append(polygon)
            
            # Batch OCR with Surya
            results = self.surya_model(
                images=[page_image],
                polygons=[polygons],
                recognition_batch_size=self.hybrid_options.surya_batch_size,
            )
            
            # Update cell text with Surya results
            if results and results[0].text_lines:
                for idx, text_line in enumerate(results[0].text_lines):
                    if idx < len(cells):
                        original_text = cells[idx].text
                        new_text = text_line.text
                        cells[idx].text = new_text
                        cells[idx].orig = new_text  # Also update original
                        
                        if self.hybrid_options.log_routing_stats:
                            _log.debug(
                                f"Surya re-OCR: '{original_text}' -> '{new_text}'"
                            )
            
        except Exception as e:
            _log.warning(f"Surya re-OCR failed: {e}")
        finally:
            # Clean up GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            except ImportError:
                pass
    
    def __call__(
        self, conv_res: ConversionResult, page_batch: Iterable[Page]
    ) -> Iterable[Page]:
        """
        Process pages with confidence-gated hybrid OCR.
        
        This overrides TesseractOcrCliModel.__call__ to add Surya routing
        after Tesseract OCR but before post-processing.
        """
        if not self.enabled:
            yield from page_batch
            return
        
        for page_i, page in enumerate(page_batch):
            assert page._backend is not None
            if not page._backend.is_valid():
                yield page
                continue
            
            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                
                all_ocr_cells = []
                
                for ocr_rect_i, ocr_rect in enumerate(ocr_rects):
                    # Skip zero area boxes
                    if ocr_rect.area() == 0:
                        continue
                    
                    # Get high-resolution image for this OCR region
                    high_res_image = page._backend.get_page_image(
                        scale=self.scale, cropbox=ocr_rect
                    )
                    
                    # Run Tesseract on this region (use parent's method)
                    try:
                        import tempfile
                        import os
                        import pandas as pd
                        
                        with tempfile.NamedTemporaryFile(
                            suffix=".png", mode="w+b", delete=False
                        ) as image_file:
                            fname = image_file.name
                            high_res_image.save(image_file)
                        
                        try:
                            # OSD for orientation detection
                            df_osd = None
                            doc_orientation = 0
                            try:
                                df_osd = self._perform_osd(fname)
                                from docling.models.tesseract_ocr_cli_model import _parse_orientation
                                doc_orientation = _parse_orientation(df_osd)
                            except Exception:
                                pass
                            
                            # Rotate if needed
                            if doc_orientation != 0:
                                high_res_image = high_res_image.rotate(
                                    -doc_orientation, expand=True
                                )
                                high_res_image.save(fname)
                            
                            # Run Tesseract
                            df_result = self._run_tesseract(fname, df_osd)
                            
                        finally:
                            if os.path.exists(fname):
                                os.remove(fname)
                        
                        # Convert Tesseract results to TextCell objects
                        from docling_core.types.doc import BoundingBox, CoordOrigin
                        from docling.utils.ocr_utils import tesseract_box_to_bounding_rectangle
                        
                        region_cells = []
                        for ix, row in df_result.iterrows():
                            text = row["text"]
                            conf = row["conf"]
                            
                            left, top = float(row["left"]), float(row["top"])
                            right = left + float(row["width"])
                            bottom = top + row["height"]
                            
                            bbox = BoundingBox(
                                l=left, t=top, r=right, b=bottom,
                                coord_origin=CoordOrigin.TOPLEFT,
                            )
                            rect = tesseract_box_to_bounding_rectangle(
                                bbox,
                                original_offset=ocr_rect,
                                scale=self.scale,
                                orientation=doc_orientation,
                                im_size=high_res_image.size,
                            )
                            
                            cell = TextCell(
                                index=ix,
                                text=str(text),
                                orig=str(text),
                                from_ocr=True,
                                confidence=conf / 100.0,
                                rect=rect,
                            )
                            region_cells.append(cell)
                        
                        # === HYBRID ROUTING: Filter and re-OCR ===
                        if region_cells:
                            # Filter cells for Surya re-OCR
                            cells_to_reocr = [
                                c for c in region_cells 
                                if self._should_route_to_surya(c)
                            ]
                            
                            # Update statistics
                            self._stats['total_cells'] += len(region_cells)
                            self._stats['surya_cells'] += len(cells_to_reocr)
                            self._stats['tesseract_cells'] += (
                                len(region_cells) - len(cells_to_reocr)
                            )
                            
                            # Re-OCR with Surya if we have cells to process
                            if cells_to_reocr:
                                if self.hybrid_options.log_routing_stats:
                                    pct = len(cells_to_reocr) / len(region_cells) * 100
                                    _log.info(
                                        f"Routing {len(cells_to_reocr)}/{len(region_cells)} "
                                        f"cells ({pct:.1f}%) to Surya"
                                    )
                                
                                # Re-OCR (modifies cells in-place)
                                self._surya_reocr_cells(
                                    high_res_image, 
                                    cells_to_reocr,
                                    scale=self.scale,
                                )
                        
                        all_ocr_cells.extend(region_cells)
                        
                    except Exception as e:
                        _log.error(f"OCR failed for region {ocr_rect_i}: {e}")
                        continue
                
                # Post-process the cells (parent class method)
                self.post_process_cells(all_ocr_cells, page)
            
            yield page
    
    def get_stats(self) -> dict:
        """Get routing statistics."""
        stats = dict(self._stats)
        total = stats['total_cells']
        if total > 0:
            stats['surya_percentage'] = stats['surya_cells'] / total * 100
        else:
            stats['surya_percentage'] = 0.0
        return stats
    
    def reset_stats(self) -> None:
        """Reset routing statistics."""
        self._stats = {
            'total_cells': 0,
            'surya_cells': 0,
            'tesseract_cells': 0,
        }
    
    @classmethod
    def get_options_type(cls) -> Type[TesseractCliOcrOptions]:
        return HybridOcrOptions
