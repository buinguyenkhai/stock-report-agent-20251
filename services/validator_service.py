class FinancialValidator:
    """
    Validates structured financial data against accounting rules.
    """
    def validate(self, data: dict) -> list:
        """
        Returns a list of validation errors/warnings. Empty list means valid.
        data structure: {'BS': [...], 'PL': [...], 'CF': [...]}
        """
        errors = []
        
        bs = {item['item_code']: item['value'] for item in data.get('BS', []) if item.get('item_code')}
        pl = {item['item_code']: item['value'] for item in data.get('PL', []) if item.get('item_code')}
        cf = {item['item_code']: item['value'] for item in data.get('CF', []) if item.get('item_code')}

        def get_val(source, code, default=0.0):
            return float(source.get(code, default) or 0.0)

        # 1. Check Balance Sheet: Assets = Resources
        # Total Assets should equal Total Resources (Liabilities + Equity)
        # We look for the largest values in the Balance Sheet which typically represent the Totals.
        # Or we can try to find items with "Tổng cộng tài sản" and "Tổng cộng nguồn vốn" in their names.
        
        total_assets = get_val(bs, "270") # Common code for Assets
        total_resources = get_val(bs, "440") # Common code for Resources
        
        # If codes are missing, try to infer from max values (heuristic)
        if total_assets == 0 and bs:
             total_assets = max(bs.values())
        if total_resources == 0 and bs:
             pass

        if total_assets > 0 and total_resources > 0:
            if abs(total_assets - total_resources) > 1.0: 
                errors.append(f"Cảnh báo Bảng CĐKT: Tổng tài sản ({total_assets:,.0f}) khác Tổng nguồn vốn ({total_resources:,.0f})")

        # 2. Check Profit: Net Profit = Profit Before Tax - Tax
        # General check.
        net_profit = get_val(pl, "60")
        pbt = get_val(pl, "50")
        tax_current = get_val(pl, "51")
        tax_deferred = get_val(pl, "52")
        
        calculated_net_profit = pbt - tax_current - tax_deferred
        
        if net_profit != 0 and pbt != 0:
             if abs(net_profit - calculated_net_profit) > 1.0:
                 errors.append(f"Cảnh báo KQKD: LN sau thuế ({net_profit:,.0f}) không khớp tính toán ({calculated_net_profit:,.0f})")

        # 3. Check Cash Flow: Cash End = Cash Begin + Net Cash Flow
        cash_end = get_val(cf, "70")
        cash_begin = get_val(cf, "60")
        net_cash_flow = get_val(cf, "50")
        exchange_diff = get_val(cf, "61")
        
        calculated_cash_end = cash_begin + net_cash_flow + exchange_diff
        
        if cash_end != 0 and cash_begin != 0:
            if abs(cash_end - calculated_cash_end) > 1.0:
                 errors.append(f"Cảnh báo LCTT: Tiền cuối kỳ ({cash_end:,.0f}) không khớp tính toán ({calculated_cash_end:,.0f})")

        return errors
