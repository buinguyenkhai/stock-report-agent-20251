import json
from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.data_transformers import (
    VnstockTransformer, 
    OCRTransformer, 
    detect_unit_from_markdown
)
from evaluation.metrics import evaluate_report, print_evaluation_summary
from logger import get_logger

logger = get_logger(__name__)


def load_ground_truth(stock_code: str, year: int, quarter: int):
    """Load ground truth from saved CSV files."""
    gt_dir = Path(f"data/ground_truth/{stock_code}_{year}_Q{quarter}")
    
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth not found: {gt_dir}")
    
    bs_df = pd.read_csv(gt_dir / "balance_sheet.csv", encoding='utf-8')
    is_df = pd.read_csv(gt_dir / "income_statement.csv", encoding='utf-8')
    cf_df = pd.read_csv(gt_dir / "cash_flow.csv", encoding='utf-8')
    
    transformer = VnstockTransformer()
    return transformer.transform_quarterly_report(
        stock_code=stock_code,
        year=year,
        quarter=quarter,
        balance_sheet_df=bs_df,
        income_statement_df=is_df,
        cash_flow_df=cf_df,
        lang='vi'
    )

def re_evaluate_report(report_dir: Path):
    """
    Re-evaluate a single report using existing files.
    """
    report_id = report_dir.name
    parts = report_id.split('_')
    
    if len(parts) != 3:
        logger.error(f"Invalid report directory name: {report_id}")
        return None
    
    stock_code = parts[0]
    year = int(parts[1])
    quarter = int(parts[2].replace('Q', ''))
    
    print(f"\n{'='*60}")
    print(f"RE-EVALUATING: {report_id}")
    print(f"{'='*60}")
    
    # Check required files
    ocr_path = report_dir / "ocr_output.md"
    parsed_path = report_dir / "parsed_output.json"
    
    if not ocr_path.exists():
        print("  ✗ OCR output not found")
        return None
    
    if not parsed_path.exists():
        print("  ✗ Parsed output not found")
        return None
    
    # Load files
    with open(ocr_path, 'r', encoding='utf-8') as f:
        markdown = f.read()
    
    with open(parsed_path, 'r', encoding='utf-8') as f:
        parsed = json.load(f)
    
    # Check if parsing was successful
    if not parsed.get('BS') and not parsed.get('PL') and not parsed.get('CF'):
        print("  ✗ Parsing failed (no data)")
        return {"report_id": report_id, "error": "Parsing failed"}
    
    # Detect unit from markdown
    detected_unit = detect_unit_from_markdown(markdown)
    parsed['unit'] = detected_unit  # Override
    
    print(f"  Unit detected: {detected_unit}")
    print(f"  BS: {len(parsed.get('BS', []))} items")
    print(f"  PL: {len(parsed.get('PL', []))} items")
    print(f"  CF: {len(parsed.get('CF', []))} items")
    
    # Load ground truth
    try:
        gt_report = load_ground_truth(stock_code, year, quarter)
        print(f"  Ground truth: BS={len(gt_report.balance_sheet.items)}, "
              f"PL={len(gt_report.income_statement.items)}, "
              f"CF={len(gt_report.cash_flow.items)}")
    except Exception as e:
        print(f"  ✗ Ground truth error: {e}")
        return {"report_id": report_id, "error": str(e)}
    
    # Transform OCR output
    ocr_transformer = OCRTransformer()
    ocr_report = ocr_transformer.transform_report(
        ocr_output=parsed,
        stock_code=stock_code,
        year=year,
        quarter=quarter
    )
    
    # Run evaluation
    evaluation = evaluate_report(
        ocr_report=ocr_report,
        vnstock_report=gt_report,
        tolerance=0.02
    )
    
    # Print results
    print_evaluation_summary(evaluation)
    
    # Save updated results
    eval_summary = {
        "report_id": report_id,
        "stock_code": stock_code,
        "year": year,
        "quarter": quarter,
        "ocr_method": "docling",
        "detected_unit": detected_unit,
        "evaluated_at": datetime.now().isoformat(),
        "overall_match_rate": evaluation.overall_match_rate,
        "overall_value_accuracy": evaluation.overall_value_accuracy,
        "overall_mape": evaluation.overall_mape,
        "sections": {
            "balance_sheet": {
                "match_rate": evaluation.balance_sheet.match_rate,
                "value_accuracy": evaluation.balance_sheet.value_accuracy,
                "mape": evaluation.balance_sheet.mape,
                "total_vnstock": evaluation.balance_sheet.total_vnstock_items,
                "total_ocr": evaluation.balance_sheet.total_ocr_items,
                "matched": evaluation.balance_sheet.matched_items
            },
            "income_statement": {
                "match_rate": evaluation.income_statement.match_rate,
                "value_accuracy": evaluation.income_statement.value_accuracy,
                "mape": evaluation.income_statement.mape,
                "total_vnstock": evaluation.income_statement.total_vnstock_items,
                "total_ocr": evaluation.income_statement.total_ocr_items,
                "matched": evaluation.income_statement.matched_items
            },
            "cash_flow": {
                "match_rate": evaluation.cash_flow.match_rate,
                "value_accuracy": evaluation.cash_flow.value_accuracy,
                "mape": evaluation.cash_flow.mape,
                "total_vnstock": evaluation.cash_flow.total_vnstock_items,
                "total_ocr": evaluation.cash_flow.total_ocr_items,
                "matched": evaluation.cash_flow.matched_items
            }
        }
    }
    
    # Save updated results
    with open(report_dir / "evaluation_results_v2.json", 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)
    
    return {
        "report_id": report_id,
        "unit": detected_unit,
        "match_rate": evaluation.overall_match_rate,
        "value_accuracy": evaluation.overall_value_accuracy,
        "mape": evaluation.overall_mape,
    }


def main():
    """Re-evaluate all existing results."""
    
    print("=" * 60)
    print("RE-EVALUATION WITH CORRECTED UNIT DETECTION")
    print("=" * 60)
    
    results_dir = Path("evaluation_results")
    
    if not results_dir.exists():
        print("No evaluation_results directory found!")
        return
    
    results = []
    
    for report_dir in sorted(results_dir.iterdir()):
        if not report_dir.is_dir() or report_dir.name.endswith('.json'):
            continue
        
        result = re_evaluate_report(report_dir)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    
    if successful:
        avg_match = sum(r["match_rate"] for r in successful) / len(successful)
        avg_accuracy = sum(r["value_accuracy"] for r in successful) / len(successful)
        avg_mape = sum(r["mape"] for r in successful) / len(successful)
        
        print(f"\nSuccessful: {len(successful)}/{len(results)}")
        print("\nAverage Metrics:")
        print(f"  Match Rate:       {avg_match:.1%}")
        print(f"  Value Accuracy:   {avg_accuracy:.1%}")
        print(f"  MAPE:             {avg_mape:.2f}%")
        
        print("\nPer-Report Results:")
        print("-" * 70)
        print(f"{'Report':<20} {'Unit':<12} {'Match':>10} {'Val Acc':>10} {'MAPE':>10}")
        print("-" * 70)
        for r in successful:
            print(f"{r['report_id']:<20} {r['unit']:<12} {r['match_rate']:>9.1%} "
                  f"{r['value_accuracy']:>9.1%} {r['mape']:>9.2f}%")
        print("-" * 70)
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  - {r['report_id']}: {r['error']}")


if __name__ == "__main__":
    main()
