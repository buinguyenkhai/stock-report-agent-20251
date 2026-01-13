"""
Extracts specific pages from PDFs and benchmarks OCR page-by-page
against the HuggingFace ground truth.
"""

import json
import time
import math
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
from .metrics import calculate_all_metrics

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
    ):
        self.pdf_dir = Path(pdf_dir) if pdf_dir else PDF_SAMPLES_DIR
        self.ocr_engine = ocr_engine
        self.dpi = dpi
        self.marker_use_llm = marker_use_llm
        self.table_only = table_only
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
            
            # Calculate metrics
            metrics = calculate_all_metrics(ocr_text, gt_sample.text)
            
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
                ocr_text_length=len(ocr_text),
                gt_text_length=len(gt_sample.text),
                processing_time_ms=elapsed_ms,
                success=True,
            )
            
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
        
        result.total_time_seconds = time.time() - start_time
        
        # Calculate mean ± std for all metrics
        successful = [p for p in result.page_results if p.success]
        if successful:
            # Mean
            result.avg_format_agnostic_cer = sum(p.format_agnostic_cer for p in successful) / len(successful)
            result.avg_content_word_recall = sum(p.content_word_recall for p in successful) / len(successful)
            result.avg_number_f1 = sum(p.number_f1 for p in successful) / len(successful)
            
            # Std
            result.std_format_agnostic_cer = compute_std([p.format_agnostic_cer for p in successful])
            result.std_content_word_recall = compute_std([p.content_word_recall for p in successful])
            result.std_number_f1 = compute_std([p.number_f1 for p in successful])
        
        logger.info(f"\n{company} Summary (mean ± std):")
        logger.info(f"  Pages: {result.successful_pages}/{result.total_pages}")
        logger.info(f"  FA-CER: {result.avg_format_agnostic_cer:.2%} ± {result.std_format_agnostic_cer:.2%}")
        logger.info(f"  Word Recall: {result.avg_content_word_recall:.2%} ± {result.std_content_word_recall:.2%}")
        logger.info(f"  Number F1: {result.avg_number_f1:.2%} ± {result.std_number_f1:.2%}")
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
        
        if all_successful:
            # Mean
            result.overall_avg_format_agnostic_cer = sum(p.format_agnostic_cer for p in all_successful) / len(all_successful)
            result.overall_avg_content_word_recall = sum(p.content_word_recall for p in all_successful) / len(all_successful)
            result.overall_avg_number_f1 = sum(p.number_f1 for p in all_successful) / len(all_successful)
            
            # Std
            result.overall_std_format_agnostic_cer = compute_std([p.format_agnostic_cer for p in all_successful])
            result.overall_std_content_word_recall = compute_std([p.content_word_recall for p in all_successful])
            result.overall_std_number_f1 = compute_std([p.number_f1 for p in all_successful])
        
        logger.info(f"\n{'='*60}")
        logger.info("PAGE-LEVEL BENCHMARK COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Companies: {result.total_companies}")
        logger.info(f"Pages: {result.successful_pages}/{result.total_pages}")
        logger.info(f"FA-CER: {result.overall_avg_format_agnostic_cer:.4f} ± {result.overall_std_format_agnostic_cer:.4f}")
        logger.info(f"Word Recall: {result.overall_avg_content_word_recall:.2%} ± {result.overall_std_content_word_recall:.2%}")
        logger.info(f"Number F1: {result.overall_avg_number_f1:.2%} ± {result.overall_std_number_f1:.2%}")
        logger.info(f"Total Time: {result.total_time_seconds:.1f}s")
        
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
    parser.add_argument("--dpi", type=int, default=300, help="DPI for page extraction")
    parser.add_argument("--engine", type=str, default="docling", choices=["docling", "marker"], help="OCR engine")
    parser.add_argument("--marker-llm", action="store_true", help="Use LLM with Marker (requires OPENROUTER_API_KEY)")
    parser.add_argument("--table-only", action="store_true", help="Only benchmark pages with financial tables")
    parser.add_argument("--output", type=str, default="results/page_level_benchmark.json")
    
    args = parser.parse_args()
    
    benchmark = PageLevelBenchmark(
        ocr_engine=args.engine,
        dpi=args.dpi,
        marker_use_llm=args.marker_llm,
        table_only=args.table_only,
    )
    result = benchmark.run(companies=args.companies, max_pages_per_company=args.max_pages)
    benchmark.save_results(result, args.output)


if __name__ == "__main__":
    main()
