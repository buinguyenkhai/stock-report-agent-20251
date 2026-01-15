"""
Confidence-Gated OCR Service

A hybrid OCR approach that uses Tesseract for high-confidence regions
and routes low-confidence regions to Surya for re-OCR.
"""

import re
import gc
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from PIL import Image

import torch

from .base import OCRStrategy

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceGatedOptions:
    """Configuration for confidence-gated OCR routing."""
    
    # Thresholds
    confidence_threshold: float = 0.7  # Route cells below this to Surya
    number_confidence_threshold: float = 0.85  # Higher threshold for number cells
    
    # Number handling
    force_surya_for_numbers: bool = False  # If True, always use Surya for number cells
    
    # Surya settings
    surya_batch_size: int = 32  # Reduced from 48 for stability
    
    # Vietnamese OCR
    tesseract_lang: List[str] = field(default_factory=lambda: ['vie'])
    
    # Debug
    log_routing_stats: bool = True


@dataclass  
class OCRCell:
    """Represents an OCR result cell with confidence."""
    text: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # (left, top, right, bottom)
    from_surya: bool = False
    has_numbers: bool = False
    
    @property
    def needs_surya_reocr(self) -> bool:
        """Check if this cell should be re-OCR'd by Surya."""
        return not self.from_surya  # Only reconsider Tesseract cells


def contains_numbers(text: str) -> bool:
    """Check if text contains digit sequences."""
    return bool(re.search(r'\d', text))


def is_financial_number(text: str) -> bool:
    """Check if text looks like a financial number (with commas, decimals)."""
    # Matches: 1,234,567 or 1.234.567 or 1,234.56 etc.
    return bool(re.search(r'\d{1,3}([,\.]\d{3})+([,\.]\d+)?', text))


class ConfidenceGatedOCRService(OCRStrategy):
    """
    Hybrid OCR service that routes between Tesseract and Surya based on confidence.
    
    Flow:
    1. Run Tesseract OCR first (fast)
    2. Identify low-confidence cells
    3. Re-OCR low-confidence cells with Surya (accurate)
    4. Merge results
    """
    
    def __init__(self, options: Optional[ConfidenceGatedOptions] = None):
        self.options = options or ConfidenceGatedOptions()
        
        # Lazy-loaded models
        self._tesseract_initialized = False
        self._surya_model = None
        
        # Statistics
        self._stats = {
            'total_cells': 0,
            'surya_cells': 0,
            'tesseract_cells': 0,
        }
    
    def _init_tesseract(self):
        """Initialize Docling's Tesseract service."""
        if self._tesseract_initialized:
            return
        
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TesseractCliOcrOptions,
        )
        from docling.datamodel.accelerator_options import AcceleratorDevice
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        
        self.pipeline_options = PdfPipelineOptions()
        self.pipeline_options.accelerator_options.device = AcceleratorDevice.CUDA
        self.pipeline_options.do_ocr = True
        self.pipeline_options.do_table_structure = True
        self.pipeline_options.table_structure_options.do_cell_matching = True
        
        self.ocr_options = TesseractCliOcrOptions(
            force_full_page_ocr=True, 
            lang=self.options.tesseract_lang
        )
        self.pipeline_options.ocr_options = self.ocr_options
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=self.pipeline_options,
                )
            }
        )
        
        self._tesseract_initialized = True
        logger.info("Tesseract OCR initialized with lang=%s", self.options.tesseract_lang)
    
    @property
    def surya_model(self):
        """Lazy load Surya recognition model using proper initialization."""
        if self._surya_model is None:
            logger.info("Loading Surya recognition model...")
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.settings import settings as surya_settings
            
            # Create FoundationPredictor with RECOGNITION checkpoint
            foundation = FoundationPredictor(
                checkpoint=surya_settings.RECOGNITION_MODEL_CHECKPOINT,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            self._surya_model = RecognitionPredictor(foundation)
            logger.info("Surya model loaded successfully")
        return self._surya_model
    
    def _should_route_to_surya(self, cell: OCRCell) -> bool:
        """Determine if a cell should be re-OCR'd by Surya."""
        if cell.from_surya:
            return False  # Already from Surya
        
        # Check if cell contains numbers
        has_numbers = contains_numbers(cell.text)
        cell.has_numbers = has_numbers
        
        # Force Surya for all numbers if configured
        if self.options.force_surya_for_numbers and has_numbers:
            return True
        
        # Use appropriate threshold
        threshold = (
            self.options.number_confidence_threshold if has_numbers
            else self.options.confidence_threshold
        )
        
        return cell.confidence < threshold
    
    def _run_tesseract_on_image(self, image: Image.Image) -> List[OCRCell]:
        """
        Run Tesseract directly on image and extract cells with confidence.
        
        Uses Tesseract's TSV output which includes word-level bboxes and confidence.
        """
        import subprocess
        import tempfile
        import csv
        import io
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
        
        try:
            # Run tesseract with TSV output
            lang = '+'.join(self.options.tesseract_lang)
            cmd = ['tesseract', '-l', lang, tmp_path, 'stdout', 'tsv']
            result = subprocess.run(cmd, capture_output=True, check=True)
            tsv_output = result.stdout.decode('utf-8')
            
            # Parse TSV output
            cells = []
            reader = csv.DictReader(io.StringIO(tsv_output), delimiter='\t')
            
            for row in reader:
                try:
                    text = row.get('text', '').strip()
                    if not text:
                        continue
                    
                    conf = float(row.get('conf', 0))
                    left = float(row.get('left', 0))
                    top = float(row.get('top', 0))
                    width = float(row.get('width', 0))
                    height = float(row.get('height', 0))
                    
                    cells.append(OCRCell(
                        text=text,
                        confidence=conf / 100.0,  # Normalize to 0-1
                        bbox=(left, top, left + width, top + height),
                        from_surya=False,
                        has_numbers=contains_numbers(text),
                    ))
                except (ValueError, KeyError):
                    continue
            
            return cells
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Tesseract failed: {e.stderr.decode() if e.stderr else str(e)}")
            return []
        finally:
            import os
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def process_image(self, image: Image.Image) -> str:
        """
        Process image with confidence-gated OCR routing.
        
        1. Run Tesseract to get cells with confidence
        2. Filter low-confidence cells
        3. Re-OCR with Surya
        4. Merge and return text
        """
        # Step 1: Get Tesseract cells
        tesseract_cells = self._run_tesseract_on_image(image)
        
        if not tesseract_cells:
            logger.warning("Tesseract returned no cells")
            return ""
        
        total_cells = len(tesseract_cells)
        
        # Step 2: Filter for Surya
        cells_to_reocr = self._filter_cells_for_surya(tesseract_cells)
        cells_routed = len(cells_to_reocr)
        
        # Step 3: Re-OCR with Surya
        if cells_to_reocr:
            logger.info(f"Routing {cells_routed}/{total_cells} cells ({100*cells_routed/total_cells:.1f}%) to Surya")
            surya_results = self._surya_ocr_cells(image, cells_to_reocr)
            
            # Merge: replace routed cells with Surya results
            reocr_set = {id(c) for c in cells_to_reocr}
            merged_cells = []
            surya_idx = 0
            
            for cell in tesseract_cells:
                if id(cell) in reocr_set and surya_idx < len(surya_results):
                    merged_cells.append(surya_results[surya_idx])
                    surya_idx += 1
                else:
                    merged_cells.append(cell)
        else:
            merged_cells = tesseract_cells
        
        # Update stats
        self._stats['total_cells'] += total_cells
        self._stats['surya_cells'] += cells_routed
        self._stats['tesseract_cells'] += total_cells - cells_routed
        
        # Return combined text
        return '\n'.join(c.text for c in merged_cells if c.text.strip())
    
    def _filter_cells_for_surya(self, cells: List[OCRCell]) -> List[OCRCell]:
        """Filter cells that need Surya re-OCR."""
        return [c for c in cells if self._should_route_to_surya(c)]
    
    def _surya_ocr_cells(
        self, 
        page_image: Image.Image, 
        cells: List[OCRCell]
    ) -> List[OCRCell]:
        """Re-OCR specific cells using Surya."""
        if not cells:
            return []
        
        # Prepare polygons for Surya (must be integers)
        polygons = []
        for cell in cells:
            l, t, r, b = [int(x) for x in cell.bbox]
            polygon = [[l, t], [r, t], [r, b], [l, b]]
            polygons.append(polygon)
        
        # Batch OCR with Surya
        try:
            results = self.surya_model(
                images=[page_image],
                polygons=[polygons],
                recognition_batch_size=self.options.surya_batch_size,
                task_names=['ocr_with_boxes'],
            )
            
            # Parse results
            surya_cells = []
            if results and results[0].text_lines:
                for idx, text_line in enumerate(results[0].text_lines):
                    if idx < len(cells):
                        original_cell = cells[idx]
                        surya_cells.append(OCRCell(
                            text=text_line.text,
                            confidence=text_line.confidence if hasattr(text_line, 'confidence') else 1.0,
                            bbox=original_cell.bbox,
                            from_surya=True,
                            has_numbers=contains_numbers(text_line.text),
                        ))
            
            return surya_cells
            
        except Exception as e:
            logger.error(f"Surya OCR failed: {e}")
            return []
        finally:
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _merge_cells(
        self, 
        tesseract_cells: List[OCRCell], 
        surya_cells: List[OCRCell],
        cells_to_reocr: List[OCRCell]
    ) -> List[OCRCell]:
        """Merge Tesseract and Surya results."""
        # Create mapping of cells that were re-OCR'd
        reocr_indices = {id(c): i for i, c in enumerate(cells_to_reocr)}
        
        merged = []
        surya_idx = 0
        
        for cell in tesseract_cells:
            if id(cell) in reocr_indices and surya_idx < len(surya_cells):
                # Use Surya result
                merged.append(surya_cells[surya_idx])
                surya_idx += 1
            else:
                # Keep Tesseract result
                merged.append(cell)
        
        return merged
    
    def process_pdf(self, pdf_url: str) -> str:
        """
        Process PDF with confidence-gated OCR routing.
        
        Full implementation that:
        1. Uses Docling to get initial OCR with confidence scores
        2. Identifies low-confidence cells
        3. Re-OCRs those cells with Surya
        4. Merges and returns markdown
        """
        self._init_tesseract()
        
        # Step 1: Run Docling's full pipeline to get cells with confidence
        conv_result = self.converter.convert(pdf_url)
        
        # Step 2: Extract cells per page with confidence scores
        total_cells = 0
        cells_routed = 0
        
        for page in conv_result.pages:
            if not hasattr(page, 'cells') or page.cells is None:
                continue
            
            # Convert Docling cells to our OCRCell format
            tesseract_cells = []
            for cell in page.cells:
                bbox = cell.rect.to_bounding_box()
                tesseract_cells.append(OCRCell(
                    text=cell.text or "",
                    confidence=cell.confidence or 0.0,
                    bbox=(bbox.l, bbox.t, bbox.r, bbox.b),
                    from_surya=False,
                    has_numbers=contains_numbers(cell.text or ""),
                ))
            
            if not tesseract_cells:
                continue
            
            total_cells += len(tesseract_cells)
            
            # Step 3: Filter cells for Surya re-OCR
            cells_to_reocr = self._filter_cells_for_surya(tesseract_cells)
            cells_routed += len(cells_to_reocr)
            
            # Step 4: Re-OCR with Surya if needed
            if cells_to_reocr and page._backend is not None:
                try:
                    # Get page image for Surya
                    page_image = page._backend.get_page_image(scale=3)  # High res
                    if page_image:
                        surya_results = self._surya_ocr_cells(page_image, cells_to_reocr)
                        
                        # Update cells in place
                        reocr_map = {id(c): i for i, c in enumerate(cells_to_reocr)}
                        for idx, cell in enumerate(tesseract_cells):
                            if id(cell) in reocr_map:
                                surya_idx = reocr_map[id(cell)]
                                if surya_idx < len(surya_results):
                                    # Replace with Surya result
                                    tesseract_cells[idx] = surya_results[surya_idx]
                except Exception as e:
                    logger.warning(f"Surya re-OCR failed for page: {e}")
        
        # Update statistics
        self._stats['total_cells'] += total_cells
        self._stats['surya_cells'] += cells_routed
        self._stats['tesseract_cells'] += total_cells - cells_routed
        
        if self.options.log_routing_stats and total_cells > 0:
            pct = cells_routed / total_cells * 100
            logger.info(f"Routed {cells_routed}/{total_cells} cells ({pct:.1f}%) to Surya")
        
        # Return markdown from Docling (it has the document structure)
        return conv_result.document.export_to_markdown()
    
    def process_page_image(
        self, 
        page_image: Image.Image,
        tesseract_cells: Optional[List[OCRCell]] = None
    ) -> Tuple[str, dict]:
        """
        Process a single page image with confidence-gated routing.
        
        Args:
            page_image: PIL Image of the page
            tesseract_cells: Optional pre-computed Tesseract cells with confidence
            
        Returns:
            Tuple of (extracted_text, routing_stats)
        """
        # If no cells provided, we need to run Tesseract first
        # This would require integration with Docling's cell-level output
        
        if tesseract_cells is None:
            # Fallback: use standard Docling pipeline
            self._init_tesseract()
            # Would need to extract cells from Docling result
            raise NotImplementedError(
                "Cell extraction from Docling not yet implemented. "
                "Use evaluate_cells() for testing with pre-extracted cells."
            )
        
        # Filter cells for Surya
        cells_to_reocr = self._filter_cells_for_surya(tesseract_cells)
        
        # Track statistics
        self._stats['total_cells'] += len(tesseract_cells)
        self._stats['surya_cells'] += len(cells_to_reocr)
        self._stats['tesseract_cells'] += len(tesseract_cells) - len(cells_to_reocr)
        
        # Re-OCR with Surya
        surya_results = []
        if cells_to_reocr:
            logger.info(f"Routing {len(cells_to_reocr)}/{len(tesseract_cells)} cells to Surya")
            surya_results = self._surya_ocr_cells(page_image, cells_to_reocr)
        
        # Merge results
        merged_cells = self._merge_cells(tesseract_cells, surya_results, cells_to_reocr)
        
        # Combine text
        extracted_text = '\n'.join(c.text for c in merged_cells if c.text.strip())
        
        routing_stats = {
            'total_cells': len(tesseract_cells),
            'surya_cells': len(cells_to_reocr),
            'tesseract_cells': len(tesseract_cells) - len(cells_to_reocr),
            'surya_percentage': len(cells_to_reocr) / max(len(tesseract_cells), 1) * 100,
        }
        
        return extracted_text, routing_stats
    
    def get_stats(self) -> dict:
        """Get cumulative routing statistics."""
        total = max(self._stats['total_cells'], 1)
        return {
            **self._stats,
            'surya_percentage': self._stats['surya_cells'] / total * 100,
        }
    
    def reset_stats(self):
        """Reset routing statistics."""
        self._stats = {
            'total_cells': 0,
            'surya_cells': 0,
            'tesseract_cells': 0,
        }
