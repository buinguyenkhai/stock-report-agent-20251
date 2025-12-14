from pathlib import Path
from vnstock import Vnstock
from typing import List, Tuple, Optional
import json
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import get_logger

logger = get_logger(__name__)


class GroundTruthCollector:
    """Collect financial data from vnstock for evaluation."""
    
    def __init__(self, output_dir: str = "data/ground_truth"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vnstock = Vnstock()
    
    def collect_quarterly_report(
        self,
        stock_code: str,
        year: int,
        quarter: int,
        save: bool = True
    ) -> Optional[dict]:
        """
        Collect ground truth for a quarterly report.
        """
        logger.info(f"Collecting ground truth: {stock_code} Q{quarter}/{year}")
        
        try:
            stock = self.vnstock.stock(symbol=stock_code, source='VCI')
            
            # Get financial statements (quarterly)
            bs_df = stock.finance.balance_sheet(period='quarter', lang='vi', dropna=True)
            is_df = stock.finance.income_statement(period='quarter', lang='vi', dropna=True)
            cf_df = stock.finance.cash_flow(period='quarter', lang='vi', dropna=True)
            
            logger.info(f"Balance Sheet columns: {bs_df.columns.tolist()[:10] if bs_df is not None else 'None'}")
            
            result = {
                "stock_code": stock_code,
                "year": year,
                "quarter": quarter,
                "balance_sheet": bs_df,
                "income_statement": is_df,
                "cash_flow": cf_df,
                "collected_at": datetime.now().isoformat()
            }
            
            if save and bs_df is not None:
                # Save to CSV
                report_dir = self.output_dir / f"{stock_code}_{year}_Q{quarter}"
                report_dir.mkdir(exist_ok=True)
                
                if bs_df is not None and not bs_df.empty:
                    bs_df.to_csv(report_dir / "balance_sheet.csv", index=False, encoding='utf-8')
                    logger.info(f"  Saved balance_sheet.csv: {len(bs_df)} rows")
                    
                if is_df is not None and not is_df.empty:
                    is_df.to_csv(report_dir / "income_statement.csv", index=False, encoding='utf-8')
                    logger.info(f"  Saved income_statement.csv: {len(is_df)} rows")
                    
                if cf_df is not None and not cf_df.empty:
                    cf_df.to_csv(report_dir / "cash_flow.csv", index=False, encoding='utf-8')
                    logger.info(f"  Saved cash_flow.csv: {len(cf_df)} rows")
                
                # Save metadata
                metadata = {
                    "stock_code": stock_code,
                    "year": year,
                    "quarter": quarter,
                    "period": f"{year}Q{quarter}",
                    "collected_at": result["collected_at"],
                    "bs_rows": len(bs_df) if bs_df is not None else 0,
                    "is_rows": len(is_df) if is_df is not None else 0,
                    "cf_rows": len(cf_df) if cf_df is not None else 0
                }
                with open(report_dir / "metadata.json", 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                logger.info(f"✓ Saved ground truth to {report_dir}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error collecting {stock_code} Q{quarter}/{year}: {e}", exc_info=True)
            return None
    
    def collect_batch(
        self,
        reports: List[Tuple[str, int, int]]  # (stock_code, year, quarter)
    ) -> dict:
        """
        Collect ground truth for multiple quarterly reports.
        """
        results = {"collected": [], "failed": []}
        total = len(reports)
        
        for idx, (stock_code, year, quarter) in enumerate(reports, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Progress: {idx}/{total}")
            logger.info(f"{'='*60}")
            
            result = self.collect_quarterly_report(stock_code, year, quarter, save=True)
            
            if result and result.get("balance_sheet") is not None:
                results["collected"].append((stock_code, year, quarter))
            else:
                results["failed"].append((stock_code, year, quarter, "No data or error"))
        
        return results


def main():
    """Collect ground truth for specified reports."""
    
    # Target reports (Consolidated/Hợp nhất)
    # Format: (stock_code, year, quarter)
    target_reports = [
        ("FPT", 2024, 4),   # FPT Q4 2024
        ("VIC", 2024, 3),   # VIC Q3 2024
        ("VCB", 2023, 2),   # VCB Q2 2023
        ("DBC", 2022, 1),   # DBC Q1 2022
    ]
    
    print("="*60)
    print("GROUND TRUTH COLLECTION")
    print("="*60)
    print("\nTarget reports:")
    for stock, year, q in target_reports:
        print(f"  - {stock} Q{q}/{year}")
    print()
    
    collector = GroundTruthCollector()
    results = collector.collect_batch(target_reports)
    
    print("\n" + "="*60)
    print("COLLECTION COMPLETE")
    print("="*60)
    print(f"Successfully collected: {len(results['collected'])} reports")
    print(f"Failed: {len(results['failed'])} reports")
    
    if results['collected']:
        print("\nCollected reports:")
        for stock, year, quarter in results['collected']:
            print(f"  - {stock} Q{quarter}/{year}")
    
    if results['failed']:
        print("\nFailed reports:")
        for stock, year, quarter, error in results['failed']:
            print(f"  - {stock} Q{quarter}/{year}: {error}")
    
    print("\nGround truth saved to: data/ground_truth/")


if __name__ == "__main__":
    main()
