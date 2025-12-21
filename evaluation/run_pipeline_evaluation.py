"""
Pipeline Evaluation Script

Tests the new extraction pipeline against vnstock ground truth.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger
from services.pipeline import create_pipeline
from services.parser import ParsedReport
from services.ocr import get_ocr_service
from evaluation.simple_evaluator import SimpleEvaluator, EvaluationResult

logger = get_logger(__name__)


# Evaluation configurations
EVAL_CONFIGS = [
    {
        "report_id": "FPT_2024_Q4",
        "stock_code": "FPT",
        "year": 2024,
        "quarter": 4,
        "pdf_path": "data/pdfs/FPT_2024_Q4.pdf",
    },
    {
        "report_id": "DBC_2022_Q1",
        "stock_code": "DBC",
        "year": 2022,
        "quarter": 1,
        "pdf_path": "data/pdfs/DBC_2022_Q1.pdf",
    },
    {
        "report_id": "VCB_2023_Q2",
        "stock_code": "VCB",
        "year": 2023,
        "quarter": 2,
        "pdf_path": "data/pdfs/VCB_2023_Q2.pdf",
    },
    {
        "report_id": "VIC_2024_Q3",
        "stock_code": "VIC",
        "year": 2024,
        "quarter": 3,
        "pdf_path": "data/pdfs/VIC_2024_Q3.pdf",
    },
]


def run_pipeline_evaluation(
    mode: str = "separate",
    skip_ocr: bool = False,
    output_dir: str = "evaluation_results_pipeline",
):
    """
    Run the pipeline evaluation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("PIPELINE EVALUATION")
    print(f"Mode: {mode}")
    print("=" * 60)
    
    # Initialize components
    pipeline = create_pipeline(
        mode=mode,
        extract_notes=True,
        extract_metadata=True,
    )
    
    evaluator = SimpleEvaluator()
    
    if not skip_ocr:
        ocr_service = get_ocr_service("marker")
        print("OCR service initialized")
    
    print("Pipeline initialized")
    print()
    
    results: List[EvaluationResult] = []
    
    for config in EVAL_CONFIGS:
        report_id = config["report_id"]
        print("=" * 60)
        print(f"PROCESSING: {report_id}")
        print("=" * 60)
        
        report_dir = output_path / report_id
        report_dir.mkdir(exist_ok=True)
        
        ocr_path = report_dir / "ocr_output.md"
        
        # Step 1: OCR
        if skip_ocr and ocr_path.exists():
            print("  ⏭️  Skipping OCR (using existing file)")
            markdown = ocr_path.read_text(encoding="utf-8")
        elif skip_ocr and not ocr_path.exists():
            print("No existing OCR file and --skip-ocr is set. Run without --skip-ocr first.")
            continue
        else:
            pdf_path = Path(config["pdf_path"])
            if not pdf_path.exists():
                print(f"PDF not found: {pdf_path}")
                continue
            
            print(f"Running OCR on {pdf_path.name}...")
            start_time = time.time()
            markdown = ocr_service.process_pdf(str(pdf_path))
            ocr_time = time.time() - start_time
            print(f"OCR complete: {len(markdown):,} chars in {ocr_time:.1f}s")
            
            ocr_path.write_text(markdown, encoding="utf-8")
        
        # Step 2: Pipeline processing
        print("Running pipeline...")
        start_time = time.time()
        
        try:
            parsed: ParsedReport = pipeline.process(markdown)
            pipeline_time = time.time() - start_time
            
            print(f"Parsed in {pipeline_time:.1f}s:")
            print(f"     BS={len(parsed.balance_sheet.items)}, "
                  f"PL={len(parsed.income_statement.items)}, "
                  f"CF={len(parsed.cash_flow.items)} items")
            print(f"     Unit: {parsed.unit}, YTD: {parsed.is_ytd}")
            
            # Save parsed output
            parsed_dict = pipeline.to_dict(parsed)
            with open(report_dir / "parsed_output.json", "w", encoding="utf-8") as f:
                json.dump(parsed_dict, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Pipeline error: {e}")
            continue
        
        # Step 3: Evaluation
        print("Evaluating...")
        
        try:
            eval_result = evaluator.evaluate(
                parsed,
                config["stock_code"],
                config["year"],
                config["quarter"],
            )
            results.append(eval_result)
            
            # Print summary
            print(f"Match Rate: {eval_result.overall_match_rate:.1f}%")
            print(f"Value Accuracy: {eval_result.overall_value_accuracy:.1f}%")
            print(f"MAPE: {eval_result.overall_mape:.2f}%")
            
            # Save evaluation
            with open(report_dir / "evaluation.json", "w", encoding="utf-8") as f:
                json.dump(eval_result.to_dict(), f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"Evaluation error: {e}")
            continue
        
        print()
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if results:
        avg_match = sum(r.overall_match_rate for r in results) / len(results)
        avg_value_acc = sum(r.overall_value_accuracy for r in results) / len(results)
        avg_mape = sum(r.overall_mape for r in results) / len(results)
        
        print(f"\nSuccessful: {len(results)}/{len(EVAL_CONFIGS)}")
        print("\nAverage Metrics:")
        print(f"  Match Rate:      {avg_match:.1f}%")
        print(f"  Value Accuracy:  {avg_value_acc:.1f}%")
        print(f"  MAPE:            {avg_mape:.2f}%")
        
        print("\nPer-Report Results:")
        print("-" * 70)
        print(f"{'Report':<20} {'Match':>10} {'Val Acc':>10} {'MAPE':>10}")
        print("-" * 70)
        
        for r in results:
            print(f"{r.report_id:<20} "
                  f"{r.overall_match_rate:>9.1f}% "
                  f"{r.overall_value_accuracy:>9.1f}% "
                  f"{r.overall_mape:>9.2f}%")
        print("-" * 70)


def compare_modes():
    """Run evaluation in both modes and compare."""
    print("\n" + "=" * 60)
    print("COMPARING EXTRACTION MODES")
    print("=" * 60 + "\n")
    
    for mode in ["separate", "combined"]:
        print(f"\n{'='*30} MODE: {mode.upper()} {'='*30}\n")
        run_pipeline_evaluation(
            mode=mode,
            skip_ocr=True,
            output_dir=f"evaluation_results_pipeline_{mode}",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline evaluation")
    parser.add_argument(
        "--mode", 
        choices=["separate", "combined", "compare"],
        default="separate",
        help="Extraction mode: separate (3 extractors), combined (1 extractor), or compare (both)"
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip OCR and use existing markdown files"
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation_results_pipeline",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_modes()
    else:
        run_pipeline_evaluation(
            mode=args.mode,
            skip_ocr=args.skip_ocr,
            output_dir=args.output_dir,
        )
