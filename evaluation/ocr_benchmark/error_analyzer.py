"""
Analyzes OCR errors from page-level benchmark results to identify patterns
and areas for improvement.
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from logger import get_logger
from .page_level_benchmark import PageLevelBenchmarkResult, CompanyResult, PageResult

logger = get_logger(__name__)


@dataclass
class ErrorAnalysisResult:
    """Complete error analysis results."""
    total_pages: int
    successful_pages: int
    failed_pages: int
    
    # Metric distribution
    high_cer_pages: int = 0  # FA-CER > 0.3
    low_number_f1_pages: int = 0  # Number F1 < 0.7
    
    # Per-company breakdown
    company_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Worst performing pages
    worst_pages: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_pages": self.total_pages,
            "successful_pages": self.successful_pages,
            "failed_pages": self.failed_pages,
            "high_cer_pages": self.high_cer_pages,
            "low_number_f1_pages": self.low_number_f1_pages,
            "company_stats": self.company_stats,
            "worst_pages": self.worst_pages,
        }


class ErrorAnalyzer:
    """
    Analyzes OCR errors from page-level benchmark results.
    """
    
    def analyze(self, result: PageLevelBenchmarkResult) -> ErrorAnalysisResult:
        """
        Analyze errors from page-level benchmark results.
        """
        logger.info(f"Analyzing errors from {result.total_pages} pages...")
        
        analysis = ErrorAnalysisResult(
            total_pages=result.total_pages,
            successful_pages=result.successful_pages,
            failed_pages=result.total_pages - result.successful_pages,
        )
        
        all_pages = []
        
        for company_result in result.company_results:
            company = company_result.company
            
            # Track company-level stats
            analysis.company_stats[company] = {
                "avg_format_agnostic_cer": company_result.avg_format_agnostic_cer,
                "avg_number_f1": company_result.avg_number_f1,
                "avg_word_recall": company_result.avg_content_word_recall,
                "pages": company_result.total_pages,
                "successful": company_result.successful_pages,
            }
            
            for page in company_result.page_results:
                if not page.success:
                    continue
                
                # Track high error pages
                if page.format_agnostic_cer > 0.3:
                    analysis.high_cer_pages += 1
                
                if page.number_f1 < 0.7:
                    analysis.low_number_f1_pages += 1
                
                all_pages.append({
                    "company": company,
                    "page": page.page_number,
                    "format_agnostic_cer": page.format_agnostic_cer,
                    "number_f1": page.number_f1,
                    "word_recall": page.content_word_recall,
                    "processing_time_ms": page.processing_time_ms,
                })
        
        # Find worst pages (by Format-Agnostic CER)
        all_pages.sort(key=lambda x: x["format_agnostic_cer"], reverse=True)
        analysis.worst_pages = all_pages[:10]
        
        return analysis
    
    def analyze_from_file(self, json_path: str) -> ErrorAnalysisResult:
        """
        Load benchmark results from JSON and analyze.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruct PageLevelBenchmarkResult
        result = PageLevelBenchmarkResult(
            timestamp=data.get("timestamp", ""),
            ocr_engine=data.get("ocr_engine", "unknown"),
            dpi=data.get("dpi", 300),
            total_companies=data.get("total_companies", 0),
            total_pages=data.get("total_pages", 0),
            successful_pages=data.get("successful_pages", 0),
            overall_avg_format_agnostic_cer=data.get("overall_avg_format_agnostic_cer", 0),
            overall_avg_content_word_recall=data.get("overall_avg_content_word_recall", 0),
            overall_avg_number_f1=data.get("overall_avg_number_f1", 0),
            overall_std_format_agnostic_cer=data.get("overall_std_format_agnostic_cer", 0),
            overall_std_content_word_recall=data.get("overall_std_content_word_recall", 0),
            overall_std_number_f1=data.get("overall_std_number_f1", 0),
            total_time_seconds=data.get("total_time_seconds", 0),
        )
        
        # Reconstruct company results
        for cr_data in data.get("company_results", []):
            company_result = CompanyResult(
                company=cr_data.get("company", ""),
                pdf_path=cr_data.get("pdf_path", ""),
                total_pages=cr_data.get("total_pages", 0),
                successful_pages=cr_data.get("successful_pages", 0),
                avg_format_agnostic_cer=cr_data.get("avg_format_agnostic_cer", 0),
                avg_content_word_recall=cr_data.get("avg_content_word_recall", 0),
                avg_number_f1=cr_data.get("avg_number_f1", 0),
                std_format_agnostic_cer=cr_data.get("std_format_agnostic_cer", 0),
                std_content_word_recall=cr_data.get("std_content_word_recall", 0),
                std_number_f1=cr_data.get("std_number_f1", 0),
                total_time_seconds=cr_data.get("total_time_seconds", 0),
            )
            
            # Reconstruct page results
            for pr_data in cr_data.get("page_results", []):
                page_result = PageResult(
                    company=pr_data.get("company", ""),
                    page_number=pr_data.get("page_number", 0),
                    format_agnostic_cer=pr_data.get("format_agnostic_cer", 0),
                    content_word_recall=pr_data.get("content_word_recall", 0),
                    number_f1=pr_data.get("number_f1", 0),
                    number_precision=pr_data.get("number_precision", 0),
                    number_recall=pr_data.get("number_recall", 0),
                    ocr_text_length=pr_data.get("ocr_text_length", 0),
                    gt_text_length=pr_data.get("gt_text_length", 0),
                    processing_time_ms=pr_data.get("processing_time_ms", 0),
                    success=pr_data.get("success", True),
                    error=pr_data.get("error"),
                )
                company_result.page_results.append(page_result)
            
            result.company_results.append(company_result)
        
        return self.analyze(result)
    
    def save_analysis(self, analysis: ErrorAnalysisResult, output_path: str) -> None:
        """Save analysis results to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error analysis saved to {output_path}")
    
    def print_summary(self, analysis: ErrorAnalysisResult) -> None:
        """Print a human-readable summary of the analysis."""
        print("ERROR ANALYSIS SUMMARY")
        
        print(f"\nTotal Pages: {analysis.total_pages}")
        print(f"Successful: {analysis.successful_pages}")
        print(f"Failed: {analysis.failed_pages}")
        
        print("\n--- Problem Pages ---")
        print(f"High FA-CER (>30%): {analysis.high_cer_pages} pages")
        print(f"Low Number F1 (<70%): {analysis.low_number_f1_pages} pages")
        
        print("\n--- Per-Company Stats ---")
        print(f"{'Company':<10} {'FA-CER':<10} {'NumF1':<10} {'WordRecall':<12} {'Pages':<8}")
        print("-" * 50)
        for company, stats in sorted(analysis.company_stats.items()):
            print(f"{company:<10} {stats['avg_format_agnostic_cer']:.2%}     {stats['avg_number_f1']:.2%}     {stats['avg_word_recall']:.2%}       {stats['pages']}")
        
        if analysis.worst_pages:
            print("\n--- Worst Performing Pages (by FA-CER) ---")
            print(f"{'Company':<10} {'Page':<6} {'FA-CER':<10} {'NumF1':<10}")
            print("-" * 40)
            for page in analysis.worst_pages[:5]:
                print(f"{page['company']:<10} {page['page']:<6} {page['format_agnostic_cer']:.2%}     {page['number_f1']:.2%}")


def main():
    """Run error analysis on existing benchmark results."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze OCR benchmark errors")
    parser.add_argument("--input", type=str, required=True, help="Path to benchmark results JSON")
    parser.add_argument("--output", type=str, default=None, help="Output path for analysis JSON")
    
    args = parser.parse_args()
    
    analyzer = ErrorAnalyzer()
    analysis = analyzer.analyze_from_file(args.input)
    
    # Print summary
    analyzer.print_summary(analysis)
    
    # Save if output specified
    if args.output:
        analyzer.save_analysis(analysis, args.output)


if __name__ == "__main__":
    main()
