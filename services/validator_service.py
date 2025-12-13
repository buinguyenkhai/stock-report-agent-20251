from config import ReportConstants
from logger import get_logger

logger = get_logger(__name__)

class FinancialValidator:
    """
    Validates structured financial data against accounting rules.
    """
    def validate(self, data: dict) -> list:
        """
        Returns a list of validation errors/warnings. Empty list means valid.
        data structure: {'BS': [...], 'PL': [...], 'CF': [...]}
        """
        if not data:
            logger.warning("Empty data received for validation")
            return ["Dữ liệu trống, không thể kiểm tra."]
            
        errors = []
        
        bs = {item['item_code']: item['value'] for item in data.get('BS', []) if item.get('item_code')}
        pl = {item['item_code']: item['value'] for item in data.get('PL', []) if item.get('item_code')}
        cf = {item['item_code']: item['value'] for item in data.get('CF', []) if item.get('item_code')}

        def get_val(source, code, default=0.0):
            val = source.get(code, default)
            return float(val) if val is not None else 0.0

        tolerance = ReportConstants.VALIDATION_TOLERANCE

        # 1. Check Balance Sheet: Assets = Resources
        total_assets = get_val(bs, ReportConstants.BS_TOTAL_ASSETS_CODE)
        total_resources = get_val(bs, ReportConstants.BS_TOTAL_RESOURCES_CODE)
        
        # If codes are missing, try to infer from max values (heuristic)
        if total_assets == 0 and bs:
            total_assets = max((v for v in bs.values() if v is not None), default=0)
        if total_resources == 0 and bs:
            # Try alternative codes or heuristics
            pass

        if total_assets > 0 and total_resources > 0:
            if abs(total_assets - total_resources) > tolerance: 
                errors.append(f"Cảnh báo Bảng CĐKT: Tổng tài sản ({total_assets:,.0f}) khác Tổng nguồn vốn ({total_resources:,.0f})")

        # 2. Check Profit: Net Profit = Profit Before Tax - Tax
        net_profit = get_val(pl, ReportConstants.PL_NET_PROFIT_CODE)
        pbt = get_val(pl, ReportConstants.PL_PROFIT_BEFORE_TAX_CODE)
        tax_current = get_val(pl, ReportConstants.PL_TAX_CURRENT_CODE)
        tax_deferred = get_val(pl, ReportConstants.PL_TAX_DEFERRED_CODE)
        
        calculated_net_profit = pbt - tax_current - tax_deferred
        
        if net_profit != 0 and pbt != 0:
            if abs(net_profit - calculated_net_profit) > tolerance:
                errors.append(f"Cảnh báo KQKD: LN sau thuế ({net_profit:,.0f}) không khớp tính toán ({calculated_net_profit:,.0f})")

        # 3. Check Cash Flow: Cash End = Cash Begin + Net Cash Flow
        cash_end = get_val(cf, ReportConstants.CF_CASH_END_CODE)
        cash_begin = get_val(cf, ReportConstants.CF_CASH_BEGIN_CODE)
        net_cash_flow = get_val(cf, ReportConstants.CF_NET_CASH_FLOW_CODE)
        exchange_diff = get_val(cf, ReportConstants.CF_EXCHANGE_DIFF_CODE)
        
        calculated_cash_end = cash_begin + net_cash_flow + exchange_diff
        
        if cash_end != 0 and cash_begin != 0:
            if abs(cash_end - calculated_cash_end) > tolerance:
                errors.append(f"Cảnh báo LCTT: Tiền cuối kỳ ({cash_end:,.0f}) không khớp tính toán ({calculated_cash_end:,.0f})")

        logger.info(f"Validation completed with {len(errors)} issues found")
        return errors
