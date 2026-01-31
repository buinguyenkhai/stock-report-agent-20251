"""
UI Components for Stock Report Agent
Provides Streamlit components for the chat interface and financial table display.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable, Tuple
import pandas as pd

from config import OCR_ENGINE_OPTIONS, LLM_MODEL_OPTIONS, DEFAULT_LLM_MODEL


def render_model_selector() -> Tuple[str, str]:
    """
    Render the OCR engine and LLM model selector dropdowns.
    Returns tuple of (ocr_engine, llm_model).
    """
    col1, col2 = st.columns(2)
    
    with col1:
        ocr_options = {label: value for label, value in OCR_ENGINE_OPTIONS}
        ocr_labels = list(ocr_options.keys())
        
        selected_ocr = st.selectbox(
            "OCR Engine",
            ocr_labels,
            index=0,
            help="Chọn công cụ OCR để xử lý báo cáo PDF"
        )
        ocr_engine = ocr_options[selected_ocr]
    
    with col2:
        llm_options = {label: value for label, value in LLM_MODEL_OPTIONS}
        llm_labels = list(llm_options.keys())
        
        selected_llm = st.selectbox(
            "LLM Model",
            llm_labels,
            index=0,
            help="Chọn model LLM để trích xuất dữ liệu"
        )
        llm_model = llm_options[selected_llm]
    
    return ocr_engine, llm_model


def render_progress_step(
    step_name: str,
    status: str,  # "pending", "running", "done", "error"
    details: Optional[str] = None
) -> None:
    """
    Render a single progress step with status icon.
    """
    icons = {
        "pending": "⏳",
        "running": "🔄",
        "done": "✅",
        "error": "❌",
    }
    icon = icons.get(status, "⏳")
    
    text = f"{icon} {step_name}"
    if details:
        text += f" ({details})"
    
    st.markdown(text)


def render_progress_stream(steps: List[Dict[str, Any]]) -> None:
    """
    Render the full progress stream with all steps.
    """
    for step in steps:
        render_progress_step(
            step.get("name", ""),
            step.get("status", "pending"),
            step.get("details")
        )


def render_report_header(
    stock_code: str,
    period: str,
    year: int,
    quarter: Optional[int] = None,
    consolidation: Optional[str] = None,
    unit: Optional[str] = None,
    pdf_link: Optional[str] = None,
) -> None:
    """
    Render the report metadata header.
    """
    # Build period string
    period_str = f"{period}"
    if quarter:
        period_str = f"Q{quarter}/{year}"
    elif period == "Cả năm":
        period_str = f"Năm {year}"
    else:
        period_str = f"{period} {year}"
    
    # Build header
    header_parts = [f"📄 **{stock_code}** - {period_str}"]
    if consolidation:
        header_parts.append(f"*{consolidation}*")
    if unit:
        header_parts.append(f"Đơn vị: {unit}")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(" | ".join(header_parts))
    with col2:
        if pdf_link:
            st.markdown(f"[📥 PDF]({pdf_link})")


def render_financial_tables(
    parsed_data: Dict[str, Any],
    on_notes_click: Optional[Callable[[str], None]] = None
) -> None:
    """
    Render tabbed financial tables with export buttons.
    """
    from .export import export_to_csv, export_to_excel, export_to_json
    
    tabs = st.tabs(["📊 Bảng cân đối", "📈 Kết quả HĐKD", "💰 Lưu chuyển tiền tệ"])
    
    table_configs = [
        ("balance_sheet", "Bảng cân đối kế toán", tabs[0]),
        ("income_statement", "Kết quả hoạt động kinh doanh", tabs[1]),
        ("cash_flow", "Lưu chuyển tiền tệ", tabs[2]),
    ]
    
    for key, name, tab in table_configs:
        with tab:
            table_data = parsed_data.get(key, {})
            items = table_data.get("items", [])
            
            if not items:
                st.info(f"Không có dữ liệu {name}")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame(items)
            
            # Reorder and rename columns
            column_order = ["item_code", "item_name", "value", "notes_ref"]
            existing_cols = [c for c in column_order if c in df.columns]
            df = df[existing_cols]
            
            column_names = {
                "item_code": "Mã",
                "item_name": "Chỉ tiêu",
                "value": "Giá trị",
                "notes_ref": "TM",
            }
            df = df.rename(columns=column_names)
            
            # Format value column with thousand separators
            if "Giá trị" in df.columns:
                df["Giá trị"] = df["Giá trị"].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                )
            
            # Display table
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=400,
            )
            
            # Export buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                csv_data = export_to_csv(items, name)
                st.download_button(
                    "⬇️ CSV",
                    data=csv_data,
                    file_name=f"{key}.csv",
                    mime="text/csv",
                    key=f"csv_{key}",
                )
            with col2:
                excel_data = export_to_excel(items, name)
                st.download_button(
                    "⬇️ Excel",
                    data=excel_data,
                    file_name=f"{key}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"excel_{key}",
                )
            with col3:
                json_data = export_to_json(items, name)
                st.download_button(
                    "⬇️ JSON",
                    data=json_data,
                    file_name=f"{key}.json",
                    mime="application/json",
                    key=f"json_{key}",
                )


def render_notes_modal(notes: List[Dict[str, Any]]) -> None:
    """
    Render notes in an expandable section.
    
    Args:
        notes: List of financial notes
    """
    if not notes:
        return
    
    with st.expander("📝 Thuyết minh báo cáo tài chính", expanded=False):
        for note in notes:
            note_num = note.get("note_number", "")
            note_title = note.get("note_title", "")
            content = note.get("content", "")
            
            if note_num or note_title:
                st.markdown(f"**{note_num}. {note_title}**")
            if content:
                st.markdown(content)
            st.divider()


def render_error_with_retry(
    error_message: str,
    on_retry: Optional[Callable[[], None]] = None
) -> None:
    """
    Render an error message with retry button.
    
    Args:
        error_message: Error message to display
        on_retry: Callback for retry button
    """
    st.error(f"❌ {error_message}")
    
    if on_retry:
        if st.button("🔄 Thử lại", key="retry_button"):
            on_retry()


def render_raw_ocr(markdown_content: str) -> None:
    """
    Render raw OCR markdown in a collapsible section.
    
    Args:
        markdown_content: Raw OCR markdown output
    """
    with st.expander("📄 Raw OCR Output (Markdown)", expanded=False):
        st.code(markdown_content, language="markdown")
