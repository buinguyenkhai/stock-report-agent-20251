"""
Export utilities for financial tables.
Provides functions to export table data to CSV, Excel, and JSON formats.
"""

import io
import json
from typing import List, Dict, Any
import pandas as pd


def items_to_dataframe(items: List[Dict[str, Any]], table_name: str) -> pd.DataFrame:
    """Convert financial items to a pandas DataFrame."""
    if not items:
        return pd.DataFrame()
    
    df = pd.DataFrame(items)
    
    # Reorder columns for better readability
    column_order = ["item_code", "item_name", "value", "notes_ref"]
    existing_cols = [c for c in column_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + other_cols]
    
    # Rename columns to Vietnamese
    column_names = {
        "item_code": "Mã số",
        "item_name": "Chỉ tiêu",
        "value": "Giá trị",
        "notes_ref": "Thuyết minh",
    }
    df = df.rename(columns=column_names)
    
    return df


def export_to_csv(items: List[Dict[str, Any]], table_name: str) -> bytes:
    """
    Export financial items to CSV format.
    """
    df = items_to_dataframe(items, table_name)
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def export_to_excel(items: List[Dict[str, Any]], table_name: str) -> bytes:
    """
    Export financial items to Excel format.
    """
    df = items_to_dataframe(items, table_name)
    
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=table_name[:31], index=False)  # Excel sheet name max 31 chars
    buffer.seek(0)
    return buffer.getvalue()


def export_to_json(items: List[Dict[str, Any]], table_name: str) -> bytes:
    """
    Export financial items to JSON format.
    """
    data = {
        "table_name": table_name,
        "items": items,
    }
    return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


def export_all_tables(parsed_data: Dict[str, Any], format: str = "json") -> bytes:
    """
    Export all financial tables to a single file.
    """
    if format == "json":
        return json.dumps(parsed_data, ensure_ascii=False, indent=2).encode("utf-8")
    
    elif format == "excel":
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            # Balance Sheet
            if "balance_sheet" in parsed_data:
                bs_items = parsed_data["balance_sheet"].get("items", [])
                df_bs = items_to_dataframe(bs_items, "Bảng cân đối")
                df_bs.to_excel(writer, sheet_name="Bảng cân đối", index=False)
            
            # Income Statement
            if "income_statement" in parsed_data:
                is_items = parsed_data["income_statement"].get("items", [])
                df_is = items_to_dataframe(is_items, "Kết quả HĐKD")
                df_is.to_excel(writer, sheet_name="Kết quả HĐKD", index=False)
            
            # Cash Flow
            if "cash_flow" in parsed_data:
                cf_items = parsed_data["cash_flow"].get("items", [])
                df_cf = items_to_dataframe(cf_items, "Lưu chuyển tiền tệ")
                df_cf.to_excel(writer, sheet_name="Lưu chuyển tiền tệ", index=False)
        
        buffer.seek(0)
        return buffer.getvalue()
    
    else:
        raise ValueError(f"Unsupported export format: {format}")
