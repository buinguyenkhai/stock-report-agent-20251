"""
Multi-Report Comparison Components
Provides UI components for comparing multiple financial reports.
- Same company, different periods → Merged comparison table
- Different companies → Side-by-side panels
"""

import streamlit as st
from typing import Dict, Any, List, Optional
import pandas as pd


def detect_comparison_type(results: List[Dict[str, Any]]) -> str:
    """
    Detect the type of comparison based on results.
    """
    if not results or len(results) < 2:
        return "single"
    
    stock_codes = set()
    for result in results:
        state = result.get("state", {})
        stock_codes.add(state.get("stock_code", ""))
    
    if len(stock_codes) == 1:
        return "same_company"
    else:
        return "different_companies"


def build_period_label(state: Dict[str, Any]) -> str:
    """Build a label for the report period."""
    stock_code = state.get("stock_code", "")
    period = state.get("period", "")
    year = state.get("year", "")
    quarter = state.get("quarter")
    
    if quarter:
        return f"{stock_code} Q{quarter}/{year}"
    elif period == "Cả năm":
        return f"{stock_code} Năm {year}"
    else:
        return f"{stock_code} {period} {year}"


def render_merged_comparison(
    results: List[Dict[str, Any]],
    table_key: str = "income_statement"
) -> None:
    """
    Render merged comparison table for same company across periods.
    """
    if len(results) < 2:
        st.warning("Cần ít nhất 2 báo cáo để so sánh")
        return
    
    # Build comparison DataFrame
    all_items = {}  # item_name -> {period1: value, period2: value, ...}
    period_labels = []
    
    for result in results:
        state = result.get("state", {})
        parsed_data = result.get("parsed_data", {})
        
        period_label = build_period_label(state)
        period_labels.append(period_label)
        
        table_data = parsed_data.get(table_key, {})
        items = table_data.get("items", [])
        
        for item in items:
            item_name = item.get("item_name", "")
            item_code = item.get("item_code", "")
            value = item.get("value")
            
            key = f"{item_code}|{item_name}" if item_code else item_name
            
            if key not in all_items:
                all_items[key] = {
                    "Mã": item_code,
                    "Chỉ tiêu": item_name,
                }
            
            all_items[key][period_label] = value
    
    # Convert to DataFrame
    df = pd.DataFrame(list(all_items.values()))
    
    # Reorder columns
    base_cols = ["Mã", "Chỉ tiêu"]
    period_cols = [c for c in df.columns if c not in base_cols]
    df = df[base_cols + period_cols]
    
    # Format numeric columns
    for col in period_cols:
        df[col] = df[col].apply(
            lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else "-"
        )
    
    # Calculate difference if exactly 2 periods
    if len(period_labels) == 2:
        df["Chênh lệch"] = ""  # Placeholder - would need numeric values
    
    # Display
    st.dataframe(df, use_container_width=True, hide_index=True, height=500)


def render_sidebyside_comparison(results: List[Dict[str, Any]]) -> None:
    """
    Render side-by-side comparison panels for different companies.
    """
    if len(results) < 2:
        st.warning("Cần ít nhất 2 báo cáo để so sánh")
        return
    
    # Limit to max 4 columns for readability
    results = results[:4]
    
    # Create columns
    cols = st.columns(len(results))
    
    for i, (col, result) in enumerate(zip(cols, results)):
        with col:
            state = result.get("state", {})
            parsed_data = result.get("parsed_data", {})
            
            # Header
            period_label = build_period_label(state)
            st.markdown(f"**{period_label}**")
            
            # Tabs for each table
            tabs = st.tabs(["📊 CĐKT", "📈 KQKD", "💰 LCTT"])
            
            table_configs = [
                ("balance_sheet", tabs[0]),
                ("income_statement", tabs[1]),
                ("cash_flow", tabs[2]),
            ]
            
            for key, tab in table_configs:
                with tab:
                    table_data = parsed_data.get(key, {})
                    items = table_data.get("items", [])
                    
                    if not items:
                        st.info("Không có dữ liệu")
                        continue
                    
                    # Simple table view
                    df = pd.DataFrame(items)
                    if "item_name" in df.columns and "value" in df.columns:
                        df = df[["item_name", "value"]]
                        df.columns = ["Chỉ tiêu", "Giá trị"]
                        df["Giá trị"] = df["Giá trị"].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else "-"
                        )
                    
                    st.dataframe(df, use_container_width=True, hide_index=True, height=300)


def render_comparison_view(results: List[Dict[str, Any]]) -> None:
    """
    Render appropriate comparison view based on report types.
    """
    comparison_type = detect_comparison_type(results)
    
    if comparison_type == "single":
        st.info("Chỉ có 1 báo cáo. Thêm báo cáo khác để so sánh.")
        return
    
    st.subheader("📊 So sánh báo cáo")
    
    if comparison_type == "same_company":
        # Merged comparison for same company
        st.caption("So sánh các kỳ của cùng công ty")
        
        # Select which table to compare
        table_option = st.selectbox(
            "Chọn báo cáo",
            ["Kết quả HĐKD", "Bảng cân đối KT", "Lưu chuyển tiền tệ"],
            key="comparison_table_select"
        )
        
        table_map = {
            "Kết quả HĐKD": "income_statement",
            "Bảng cân đối KT": "balance_sheet",
            "Lưu chuyển tiền tệ": "cash_flow",
        }
        
        render_merged_comparison(results, table_map[table_option])
        
    else:
        # Side-by-side for different companies
        st.caption("So sánh các công ty khác nhau")
        render_sidebyside_comparison(results)
