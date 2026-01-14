"""
Extracts specific pages from PDFs and benchmarks OCR page-by-page
against the HuggingFace ground truth.
"""

import json
import time
import math
import re
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from PIL import Image
import io
import tempfile
import os

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logger import get_logger
from .dataset_loader import VnPdfDataset, VnPdfSample
from .metrics import calculate_all_metrics, calculate_content_word_recall, calculate_number_precision_recall_f1

logger = get_logger(__name__)

# PDF samples directory
PDF_SAMPLES_DIR = Path("data/pdf_samples")

# Company code mapping
COMPANY_CODES = ["AAA", "ACB", "FPT", "MBB", "MWG", "SHB", "TCB", "VIB", "VPB"]


def compute_std(values: List[float]) -> float:
    """Compute standard deviation of values."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)  # Sample std
    return math.sqrt(variance)


def extract_table_content(text: str) -> str:
    """
    Extract only table content from text.
    
    Ground truth only contains table rows (lines starting with '|').
    This ensures fair comparison by extracting only table content from OCR output.
    
    Args:
        text: Raw OCR output text
        
    Returns:
        Filtered text containing only table rows
    """
    lines = text.split('\n')
    table_lines = []
    for line in lines:
        stripped = line.strip()
        # Table rows start with '|' or contain table separators
        if stripped.startswith('|') or '|---|' in stripped:
            table_lines.append(line)
    return '\n'.join(table_lines)


def count_numbers_in_text(text: str) -> int:
    """
    Count numerical values in text.
    
    Used to determine if a page contains numerical data for NumF1 calculation.
    """
    # Match numbers with optional commas/dots (e.g., 1,234.56 or 1.234,56)
    numbers = re.findall(r'\d[\d,.]*', text)
    # Filter to only count numbers with at least 1 digit
    return len([n for n in numbers if any(c.isdigit() for c in n)])


@dataclass
class PageResult:
    """Result for a single page."""
    company: str
    page_number: int
    # Primary metrics
    format_agnostic_cer: float
    content_word_recall: float
    number_f1: float
    # Number F1 details
    number_precision: float = 0.0
    number_recall: float = 0.0
    # Meta
    ocr_text_length: int = 0
    gt_text_length: int = 0
    processing_time_ms: float = 0.0
    success: bool = True
    error: Optional[str] = None
    # OCR output text (optional, for debugging)
    ocr_text: Optional[str] = None
    # Ground truth text (for aggregation - always stored)
    gt_text: Optional[str] = None
    # Flag to indicate if GT has numbers (for conditional NumF1 averaging)
    gt_has_numbers: bool = True


@dataclass
class CompanyResult:
    """Results for a single company with mean ± std."""
    company: str
    pdf_path: str
    total_pages: int
    successful_pages: int
    
    # Mean metrics
    avg_format_agnostic_cer: float = 0.0
    avg_content_word_recall: float = 0.0
    avg_number_f1: float = 0.0
    
    # Std metrics
    std_format_agnostic_cer: float = 0.0
    std_content_word_recall: float = 0.0
    std_number_f1: float = 0.0
    
    # Aggregated metrics (calculated over all text/numbers in company, not per-page average)
    aggregated_word_recall: float = 0.0  # Total matched words / Total GT words
    aggregated_number_f1: float = 0.0  # F1 over all numbers in company
    aggregated_number_precision: float = 0.0
    aggregated_number_recall: float = 0.0
    pages_with_numbers: int = 0  # Count of pages that have numbers in GT
    
    total_time_seconds: float = 0.0
    
    page_results: List[PageResult] = field(default_factory=list)


@dataclass
class PageLevelBenchmarkResult:
    """Full benchmark results with mean ± std."""
    timestamp: str
    ocr_engine: str
    dpi: int
    total_companies: int
    total_pages: int
    successful_pages: int
    
    # Mean metrics
    overall_avg_format_agnostic_cer: float = 0.0
    overall_avg_content_word_recall: float = 0.0
    overall_avg_number_f1: float = 0.0
    
    # Std metrics
    overall_std_format_agnostic_cer: float = 0.0
    overall_std_content_word_recall: float = 0.0
    overall_std_number_f1: float = 0.0
    
    # Aggregated metrics (calculated over all text/numbers, not per-page average)
    overall_aggregated_word_recall: float = 0.0
    overall_aggregated_number_f1: float = 0.0
    overall_aggregated_number_precision: float = 0.0
    overall_aggregated_number_recall: float = 0.0
    total_pages_with_numbers: int = 0
    
    total_time_seconds: float = 0.0
    
    company_results: List[CompanyResult] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PageLevelBenchmark:
    """
    Benchmark OCR page-by-page by extracting specific pages from PDFs.
    """
    
    def __init__(
        self, 
        pdf_dir: str = None, 
        ocr_engine: str = "docling",  # "docling" or "marker"
        dpi: int = 300,
        marker_use_llm: bool = False,  # Use LLM for Marker post-processing (requires OpenRouter API key)
        table_only: bool = False,  # Only benchmark pages with financial tables
        save_ocr_outputs: bool = False,  # Save OCR text outputs for debugging
        ocr_outputs_dir: str = None,  # Directory to save OCR outputs
    ):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_SAMPLES_DIR
        self.ocr_engine = ocr_engine
        self.dpi = dpi
        self.marker_use_llm = marker_use_llm
        self.table_only = table_only
        self.save_ocr_outputs = save_ocr_outputs
        self.ocr_outputs_dir = Path(ocr_outputs_dir) if ocr_outputs_dir else Path("results/ocr_outputs")
        self._dataset = None
        self._gt_by_company = None
        self._marker_service = None
        self._docling_service = None
    
    @property
    def dataset(self) -> VnPdfDataset:
        if self._dataset is None:
            self._dataset = VnPdfDataset()
        return self._dataset
    
    @property
    def gt_by_company(self) -> Dict[str, Dict[int, VnPdfSample]]:
        """Get ground truth samples organized by company and page number."""
        if self._gt_by_company is None:
            self._gt_by_company = {}
            for sample in self.dataset.get_samples():
                # Filter to table pages only if requested
                if self.table_only and not sample.is_table_page:
                    continue
                company = sample.custom_id.split('/')[2]
                if company not in self._gt_by_company:
                    self._gt_by_company[company] = {}
                self._gt_by_company[company][sample.page_number] = sample
        return self._gt_by_company
    
    def get_pdf_path(self, company: str) -> Optional[Path]:
        """Get PDF path for a company."""
        pattern = f"{company}*.pdf"
        matches = list(self.pdf_dir.glob(pattern))
        return matches[0] if matches else None
    
    def extract_page_image(self, pdf_path: Path, page_num: int) -> Optional[Image.Image]:
        """
        Extract a specific page from PDF as high-resolution image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 1-indexed page number
            
        Returns:
            PIL Image of the page, or None if failed
        """
        try:
            doc = fitz.open(pdf_path)
            
            if page_num < 1 or page_num > len(doc):
                logger.warning(f"Page {page_num} out of range for {pdf_path.name} ({len(doc)} pages)")
                return None
            
            page = doc[page_num - 1]  # fitz uses 0-indexed
            
            # Render at high DPI for better OCR
            zoom = self.dpi / 72  # 72 is default PDF DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            doc.close()
            return img
            
        except Exception as e:
            logger.error(f"Failed to extract page {page_num} from {pdf_path}: {e}")
            return None
    
    def ocr_image(self, img: Image.Image) -> str:
        """Run OCR on an image using Docling."""
        # Lazy-load Docling service
        if self._docling_service is None:
            from services.ocr.docling import DoclingOCRService
            self._docling_service = DoclingOCRService()
        
        return self._docling_service.process_image(img)
    
    def ocr_pdf_page_with_marker(self, pdf_path: Path, page_num: int) -> str:
        """
        Run OCR on a specific PDF page using Marker.
        
        Marker processes the entire PDF but we extract just the page we need.
        For efficiency, we create a single-page PDF extract.
        """
        # Create a temporary single-page PDF for Marker
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Extract single page to temp PDF
            doc = fitz.open(pdf_path)
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_num-1, to_page=page_num-1)
            single_page_doc.save(tmp_path)
            single_page_doc.close()
            doc.close()
            
            # Lazy-load Marker service
            if self._marker_service is None:
                from services.ocr.marker import MarkerOCRService
                self._marker_service = MarkerOCRService(use_llm=self.marker_use_llm)
            
            # Run Marker OCR
            ocr_text = self._marker_service.process_pdf(tmp_path)
            return ocr_text
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _save_ocr_output(self, company: str, page_num: int, ocr_text: str, gt_text: str) -> None:
        """
        Save OCR output and ground truth text for debugging/analysis.
        
        Creates files:
        - {ocr_outputs_dir}/{engine}/{company}/page_{page_num}_ocr.txt
        - {ocr_outputs_dir}/{engine}/{company}/page_{page_num}_gt.txt
        """
        output_dir = self.ocr_outputs_dir / self.ocr_engine / company
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save OCR output
        ocr_path = output_dir / f"page_{page_num:03d}_ocr.txt"
        with open(ocr_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        
        # Save ground truth
        gt_path = output_dir / f"page_{page_num:03d}_gt.txt"
        with open(gt_path, 'w', encoding='utf-8') as f:
            f.write(gt_text)
    
    def benchmark_page(self, company: str, page_num: int, pdf_path: Path, gt_sample: VnPdfSample) -> PageResult:
        """Benchmark a single page."""
        start_time = time.time()
        
        try:
            # Run OCR based on engine
            if self.ocr_engine == "marker":
                logger.info(f"  Processing page {page_num} with Marker...")
                ocr_text = self.ocr_pdf_page_with_marker(pdf_path, page_num)
            else:
                # Docling: Extract image and OCR
                img = self.extract_page_image(pdf_path, page_num)
                if img is None:
                    return PageResult(
                        company=company,
                        page_number=page_num,
                        format_agnostic_cer=1.0,
                        content_word_recall=0.0,
                        number_f1=0.0,
                        success=False,
                        error=f"Failed to extract page {page_num}"
                    )
                
                logger.info(f"  Extracted page {page_num}: {img.size[0]}x{img.size[1]} @ {self.dpi}dpi")
                ocr_text = self.ocr_image(img)
            
            # Extract only table content from OCR output (ground truth only has table rows)
            ocr_table_text = extract_table_content(ocr_text)
            
            # Check if ground truth has numbers (for conditional NumF1 averaging)
            gt_number_count = count_numbers_in_text(gt_sample.text)
            has_numbers = gt_number_count > 0
            
            # Calculate metrics using table-only OCR text
            metrics = calculate_all_metrics(ocr_table_text, gt_sample.text)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            # Extract number F1 details
            num_details = metrics["number_f1"].details or {}
            
            result = PageResult(
                company=company,
                page_number=page_num,
                # Primary metrics
                format_agnostic_cer=metrics["format_agnostic_cer"].value,
                content_word_recall=metrics["content_word_recall"].value,
                number_f1=metrics["number_f1"].value,
                # Number F1 details
                number_precision=num_details.get("precision", 0.0),
                number_recall=num_details.get("recall", 0.0),
                # Meta
                ocr_text_length=len(ocr_table_text),  # Length of table-only text
                gt_text_length=len(gt_sample.text),
                processing_time_ms=elapsed_ms,
                success=True,
                # Always store OCR and GT text for aggregation
                ocr_text=ocr_table_text,
                gt_text=gt_sample.text,
                # Flag for conditional NumF1 averaging
                gt_has_numbers=has_numbers,
            )
            
            # Save OCR output to file if enabled
            if self.save_ocr_outputs:
                self._save_ocr_output(company, page_num, ocr_text, gt_sample.text)
            
            logger.info(f"    FA-CER: {result.format_agnostic_cer:.4f}, WordRecall: {result.content_word_recall:.2%}, NumF1: {result.number_f1:.2%}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing page {page_num}: {e}")
            return PageResult(
                company=company,
                page_number=page_num,
                format_agnostic_cer=1.0,
                content_word_recall=0.0,
                number_f1=0.0,
                success=False,
                error=str(e)
            )
    
    def benchmark_company(self, company: str, max_pages: int = None) -> Optional[CompanyResult]:
        """Benchmark pages for a company.
        
        Args:
            company: Company code
            max_pages: Maximum pages to process (None = all pages)
        """
        pdf_path = self.get_pdf_path(company)
        if not pdf_path:
            logger.warning(f"No PDF found for {company}")
            return None
        
        gt_pages = self.gt_by_company.get(company, {})
        if not gt_pages:
            logger.warning(f"No ground truth for {company}")
            return None
        
        # Apply page limit BEFORE processing
        sorted_pages = sorted(gt_pages.items())
        if max_pages is not None:
            sorted_pages = sorted_pages[:max_pages]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking {company}: {len(sorted_pages)} pages" + (f" (limited from {len(gt_pages)})" if max_pages else ""))
        logger.info(f"PDF: {pdf_path.name}")
        logger.info(f"{'='*60}")
        
        result = CompanyResult(
            company=company,
            pdf_path=str(pdf_path),
            total_pages=len(sorted_pages),
            successful_pages=0,
        )
        
        start_time = time.time()
        
        for page_num, gt_sample in sorted_pages:
            logger.info(f"Processing page {page_num}...")
            page_result = self.benchmark_page(company, page_num, pdf_path, gt_sample)
            result.page_results.append(page_result)
            
            if page_result.success:
                result.successful_pages += 1
            
            # Clear CUDA cache after each page to prevent OOM from memory fragmentation
            if self.ocr_engine == "marker":
                try:
                    import torch
                    import gc
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        gc.collect()
                except ImportError:
                    pass
        
        result.total_time_seconds = time.time() - start_time
        
        # Calculate mean ± std for all metrics
        successful = [p for p in result.page_results if p.success]
        # For NumF1, only include pages where ground truth has numbers
        pages_with_numbers = [p for p in successful if p.gt_has_numbers]
        
        if successful:
            # Mean (FA-CER and WordRecall use all successful pages)
            result.avg_format_agnostic_cer = sum(p.format_agnostic_cer for p in successful) / len(successful)
            result.avg_content_word_recall = sum(p.content_word_recall for p in successful) / len(successful)
            
            # Std
            result.std_format_agnostic_cer = compute_std([p.format_agnostic_cer for p in successful])
            result.std_content_word_recall = compute_std([p.content_word_recall for p in successful])
        
        # NumF1: Only average over pages with numbers in ground truth
        if pages_with_numbers:
            result.avg_number_f1 = sum(p.number_f1 for p in pages_with_numbers) / len(pages_with_numbers)
            result.std_number_f1 = compute_std([p.number_f1 for p in pages_with_numbers])
        
        result.pages_with_numbers = len(pages_with_numbers)
        
        # Calculate AGGREGATED metrics (over all text/numbers, not per-page averages)
        # This is more robust for pages with few numbers
        if successful:
            # Concatenate all OCR and GT text
            all_ocr_text = '\n'.join(p.ocr_text or '' for p in successful)
            all_gt_text = '\n'.join(p.gt_text or '' for p in successful)
            
            # Aggregated Word Recall
            agg_word_recall = calculate_content_word_recall(all_ocr_text, all_gt_text)
            result.aggregated_word_recall = agg_word_recall.value
            
            # Aggregated Number F1
            agg_num_f1 = calculate_number_precision_recall_f1(all_ocr_text, all_gt_text)
            result.aggregated_number_f1 = agg_num_f1.value
            result.aggregated_number_precision = agg_num_f1.details.get("precision", 0.0) if agg_num_f1.details else 0.0
            result.aggregated_number_recall = agg_num_f1.details.get("recall", 0.0) if agg_num_f1.details else 0.0
        
        logger.info(f"\n{company} Summary:")
        logger.info(f"  Pages: {result.successful_pages}/{result.total_pages}")
        logger.info(f"  Per-Page Avg (mean ± std):")
        logger.info(f"    FA-CER: {result.avg_format_agnostic_cer:.2%} ± {result.std_format_agnostic_cer:.2%}")
        logger.info(f"    Word Recall: {result.avg_content_word_recall:.2%} ± {result.std_content_word_recall:.2%}")
        logger.info(f"    Number F1: {result.avg_number_f1:.2%} ± {result.std_number_f1:.2%} (n={result.pages_with_numbers} pages)")
        logger.info(f"  Aggregated:")
        logger.info(f"    Word Recall: {result.aggregated_word_recall:.2%}")
        logger.info(f"    Number F1: {result.aggregated_number_f1:.2%} (P={result.aggregated_number_precision:.2%}, R={result.aggregated_number_recall:.2%})")
        logger.info(f"  Time: {result.total_time_seconds:.1f}s")
        
        return result
    
    def run(self, companies: List[str] = None, max_pages_per_company: int = None) -> PageLevelBenchmarkResult:
        """
        Run benchmark on specified companies.
        
        Args:
            companies: List of company codes, or None for all
            max_pages_per_company: Limit pages per company for quick testing
        """
        logger.info("Starting Page-Level OCR Benchmark")
        logger.info(f"DPI: {self.dpi}, OCR Engine: {self.ocr_engine}")
        
        if companies is None:
            companies = COMPANY_CODES
        
        result = PageLevelBenchmarkResult(
            timestamp=datetime.now().isoformat(),
            ocr_engine=self.ocr_engine,
            dpi=self.dpi,
            total_companies=len(companies),
            total_pages=0,
            successful_pages=0,
        )
        
        start_time = time.time()
        
        for company in companies:
            # Pass max_pages to benchmark_company to limit BEFORE processing
            company_result = self.benchmark_company(company, max_pages=max_pages_per_company)
            if company_result:
                result.company_results.append(company_result)
                result.total_pages += company_result.total_pages
                result.successful_pages += company_result.successful_pages
        
        result.total_time_seconds = time.time() - start_time
        
        # Calculate overall mean ± std
        all_successful = []
        for cr in result.company_results:
            all_successful.extend([p for p in cr.page_results if p.success])
        
        # For NumF1, only include pages where ground truth has numbers
        all_with_numbers = [p for p in all_successful if p.gt_has_numbers]
        
        if all_successful:
            # Mean (FA-CER and WordRecall use all successful pages)
            result.overall_avg_format_agnostic_cer = sum(p.format_agnostic_cer for p in all_successful) / len(all_successful)
            result.overall_avg_content_word_recall = sum(p.content_word_recall for p in all_successful) / len(all_successful)
            
            # Std
            result.overall_std_format_agnostic_cer = compute_std([p.format_agnostic_cer for p in all_successful])
            result.overall_std_content_word_recall = compute_std([p.content_word_recall for p in all_successful])
        
        # NumF1: Only average over pages with numbers in ground truth
        if all_with_numbers:
            result.overall_avg_number_f1 = sum(p.number_f1 for p in all_with_numbers) / len(all_with_numbers)
            result.overall_std_number_f1 = compute_std([p.number_f1 for p in all_with_numbers])
        
        result.total_pages_with_numbers = len(all_with_numbers)
        
        # Calculate OVERALL AGGREGATED metrics
        if all_successful:
            # Concatenate all OCR and GT text
            all_ocr_text = '\n'.join(p.ocr_text or '' for p in all_successful)
            all_gt_text = '\n'.join(p.gt_text or '' for p in all_successful)
            
            # Aggregated Word Recall
            agg_word_recall = calculate_content_word_recall(all_ocr_text, all_gt_text)
            result.overall_aggregated_word_recall = agg_word_recall.value
            
            # Aggregated Number F1
            agg_num_f1 = calculate_number_precision_recall_f1(all_ocr_text, all_gt_text)
            result.overall_aggregated_number_f1 = agg_num_f1.value
            result.overall_aggregated_number_precision = agg_num_f1.details.get("precision", 0.0) if agg_num_f1.details else 0.0
            result.overall_aggregated_number_recall = agg_num_f1.details.get("recall", 0.0) if agg_num_f1.details else 0.0
        
        logger.info(f"\n{'='*60}")
        logger.info("PAGE-LEVEL BENCHMARK COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Companies: {result.total_companies}")
        logger.info(f"Pages: {result.successful_pages}/{result.total_pages}")
        logger.info(f"\nPer-Page Avg (mean ± std):")
        logger.info(f"  FA-CER: {result.overall_avg_format_agnostic_cer:.4f} ± {result.overall_std_format_agnostic_cer:.4f}")
        logger.info(f"  Word Recall: {result.overall_avg_content_word_recall:.2%} ± {result.overall_std_content_word_recall:.2%}")
        logger.info(f"  Number F1: {result.overall_avg_number_f1:.2%} ± {result.overall_std_number_f1:.2%} (n={result.total_pages_with_numbers} pages)")
        logger.info(f"\nAggregated:")
        logger.info(f"  Word Recall: {result.overall_aggregated_word_recall:.2%}")
        logger.info(f"  Number F1: {result.overall_aggregated_number_f1:.2%} (P={result.overall_aggregated_number_precision:.2%}, R={result.overall_aggregated_number_recall:.2%})")
        logger.info(f"\nTotal Time: {result.total_time_seconds:.1f}s")
        
        return result
    
    def save_results(self, result: PageLevelBenchmarkResult, output_path: str) -> None:
        """Save results to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_path}")


def main():
    """Run benchmark from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run page-level OCR benchmark")
    parser.add_argument("--companies", nargs="*", help="Company codes to benchmark")
    parser.add_argument("--max-pages", type=int, default=None, help="Max pages per company")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page extraction (default: 300, try 400-600 for scanned PDFs)")
    parser.add_argument("--engine", type=str, default="docling", choices=["docling", "marker"], help="OCR engine")
    parser.add_argument("--marker-llm", action="store_true", help="Use LLM with Marker (requires OPENROUTER_API_KEY)")
    parser.add_argument("--table-only", action="store_true", help="Only benchmark pages with financial tables")
    parser.add_argument("--output", type=str, default="results/page_level_benchmark.json")
    parser.add_argument("--save-outputs", action="store_true", help="Save OCR outputs for debugging/analysis")
    parser.add_argument("--outputs-dir", type=str, default="results/ocr_outputs", help="Directory to save OCR outputs")
    
    args = parser.parse_args()
    
    benchmark = PageLevelBenchmark(
        ocr_engine=args.engine,
        dpi=args.dpi,
        marker_use_llm=args.marker_llm,
        table_only=args.table_only,
        save_ocr_outputs=args.save_outputs,
        ocr_outputs_dir=args.outputs_dir,
    )
    result = benchmark.run(companies=args.companies, max_pages_per_company=args.max_pages)
    benchmark.save_results(result, args.output)


if __name__ == "__main__":
    main()
