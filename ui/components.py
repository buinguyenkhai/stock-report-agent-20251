"""
UI Components for Stock Report Agent
Provides Streamlit components for the chat interface and financial table display.
"""

import streamlit as st
from typing import Dict, Any, List, Optional, Callable, Tuple
import pandas as pd

from config import OCR_ENGINE_OPTIONS, LLM_MODEL_OPTIONS, DEFAULT_LLM_MODEL
from logger import get_logger

logger = get_logger(__name__)


def render_chat_input(is_processing: bool = False) -> Tuple[str, str, Optional[Any], str, bool, bool]:
    """
    Render a modern AI chatbox-style input with inline controls.
    Layout: Text input on top, controls + send/stop button on bottom.
    """
    # Initialize session state
    if "ocr_engine_idx" not in st.session_state:
        st.session_state.ocr_engine_idx = 0
    if "llm_model_idx" not in st.session_state:
        st.session_state.llm_model_idx = 0
    
    # CSS for modern chatbox styling - tighter layout
    st.markdown("""
    <style>
    /* Chat input container styling */
    .stForm {
        background: transparent !important;
        border: none !important;
    }
    
    /* Make form inputs look inline with minimal gaps */
    [data-testid="stForm"] > div {
        background: rgba(38, 39, 48, 0.95);
        border-radius: 16px;
        padding: 8px 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Remove vertical gaps between elements */
    [data-testid="stForm"] > div > div {
        gap: 0 !important;
        margin: 0 !important;
    }
    
    [data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: 0.25rem !important;
    }
    
    /* Style selectboxes to be compact */
    .chat-controls .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        min-height: 34px;
    }
    
    /* Style text input */
    .chat-input-area .stTextInput > div > div > input {
        background: transparent !important;
        border: none !important;
        font-size: 15px;
        padding: 4px 0;
    }
    
    /* Popover button styling for PDF upload */
    .chat-controls [data-testid="stPopover"] > button {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 4px 8px;
        font-size: 14px;
        min-height: 34px;
    }
    
    .chat-controls [data-testid="stPopover"] > button:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.25);
    }
    
    /* Submit button styling */
    .chat-controls .stFormSubmitButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 50%;
        padding: 0;
        font-weight: 500;
        width: 34px;
        height: 34px;
        min-width: 34px;
        max-width: 34px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .chat-controls .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Build options
    ocr_options = {label: value for label, value in OCR_ENGINE_OPTIONS}
    ocr_labels = list(ocr_options.keys())
    llm_options = {label: value for label, value in LLM_MODEL_OPTIONS}
    llm_labels = list(llm_options.keys())
    
    # Default return values
    ocr_engine = ocr_options[ocr_labels[st.session_state.ocr_engine_idx]]
    llm_model = DEFAULT_LLM_MODEL
    uploaded_pdf: Optional[Any] = None
    query_text = ""
    submitted = False
    stop_requested = False
    
    # Check if a PDF is already uploaded from previous interaction
    has_uploaded_pdf = bool(st.session_state.get("chat_pdf_upload"))
    
    # Show processing indicator
    if is_processing:
        st.info(
            "**Đang xử lý...** OCR và trích xuất có thể mất 5-10 phút tùy độ dài PDF.",
            icon="⏳"
        )
    
    # Create form for submit behavior
    with st.form("chat_input_form", clear_on_submit=False):
        # ROW 1: Text input
        st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
        
        # Determine if text input should be disabled
        text_input_disabled = is_processing or has_uploaded_pdf
        text_placeholder = "PDF đã được chọn - nhấn ▶ để xử lý" if has_uploaded_pdf else "Nhập mã chứng khoán và kỳ báo cáo (VD: FPT Q3 2024)"
        
        query_text = st.text_input(
            "Query",
            placeholder=text_placeholder,
            label_visibility="collapsed",
            key="chat_query_input",
            disabled=text_input_disabled
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ROW 2: Controls (OCR, LLM, PDF, Send)
        st.markdown('<div class="chat-controls">', unsafe_allow_html=True)
        ctrl_cols = st.columns([3, 3, 0.7, 3.4, 0.5])
        
        with ctrl_cols[0]:
            selected_ocr_idx = st.selectbox(
                "OCR",
                range(len(ocr_labels)),
                index=st.session_state.ocr_engine_idx,
                format_func=lambda i: f"{ocr_labels[i]}",
                label_visibility="collapsed",
                key="chat_ocr_select",
                disabled=is_processing
            )
            st.session_state.ocr_engine_idx = selected_ocr_idx
            ocr_engine = ocr_options[ocr_labels[selected_ocr_idx]]
        
        with ctrl_cols[1]:
            # Ensure index is within bounds
            current_llm_idx = min(st.session_state.llm_model_idx, len(llm_labels) - 1)
            selected_llm_idx = st.selectbox(
                "LLM",
                range(len(llm_labels)),
                index=current_llm_idx,
                format_func=lambda i: f"{llm_labels[i]}",
                label_visibility="collapsed",
                key="chat_llm_select",
                disabled=is_processing
            )
            st.session_state.llm_model_idx = selected_llm_idx
            llm_model = llm_options.get(llm_labels[selected_llm_idx], DEFAULT_LLM_MODEL)
        
        with ctrl_cols[2]:
            # PDF upload indicator when files are selected
            pdf_button_label = "📎 1" if has_uploaded_pdf else "📎"
            with st.popover(pdf_button_label, width='stretch', disabled=is_processing):
                uploaded_pdf = st.file_uploader(
                    "Upload PDF",
                    type=["pdf"],
                    accept_multiple_files=False,
                    key="chat_pdf_upload",
                    help="Upload 1 PDF file (30MB max)"
                )
                if uploaded_pdf:
                    st.caption("1 file selected")
                    st.caption("💡 Xóa các file ở trên để nhập truy vấn văn bản")
        
        # ctrl_cols[3] is spacer
        
        with ctrl_cols[4]:
            # Send button
            submitted = st.form_submit_button("▶", width='stretch', disabled=is_processing)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return ocr_engine, llm_model, uploaded_pdf, query_text, submitted, stop_requested

def render_model_selector() -> Tuple[str, str, Optional[Any]]:
    ocr_engine, llm_model, pdf, _, _, _ = render_chat_input()
    return ocr_engine, llm_model, pdf


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
    year_val = int(year) if isinstance(year, int) else 0
    period_str = f"{period}".strip()
    if quarter and year_val:
        period_str = f"Q{quarter}/{year_val}"
    elif period == "Cả năm" and year_val:
        period_str = f"Năm {year_val}"
    elif year_val and period_str:
        period_str = f"{period_str} {year_val}"

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
            st.link_button("📥 PDF", pdf_link)


def render_financial_tables(
    parsed_data: Dict[str, Any],
    on_notes_click: Optional[Callable[[str], None]] = None
) -> None:
    """
    Render tabbed financial tables with export buttons and clickable notes.
    """
    from .export import export_to_csv, export_to_excel, export_to_json
    
    # Get structured notes from parsed_data
    notes_by_ref: Dict[str, Dict] = parsed_data.get("notes_by_ref", {})
    notes_content = parsed_data.get("notes_content", "")
    
    # Store in session state for access by other components
    st.session_state.notes_by_ref = notes_by_ref
    st.session_state.notes_content = notes_content
    
    def normalize_ref(ref: str) -> str:
        """Normalize note reference for lookup."""
        import re
        if not ref:
            return ""
        ref = str(ref).strip().rstrip('.')
        # Fix OCR error: S/s at start -> 5.
        ref = re.sub(r'^[sS](\d)', r'5.\1', ref)
        # Fix missing decimal: 53 -> 5.3
        if re.match(r'^5\d{1,2}$', ref) and '.' not in ref:
            ref = '5.' + ref[1:]
        # Remove leading zeros: 5.01 -> 5.1
        ref = re.sub(r'\.0+(\d)', r'.\1', ref)
        return re.sub(r'\s+', '', ref).upper()
    
    def lookup_note_content(note_ref: str) -> Tuple[str, bool]:
        """
        Look up note content using the LLM-indexed notes_by_ref dictionary.
        """
        if not note_ref or not notes_by_ref:
            return "", False
        
        norm_ref = normalize_ref(note_ref)
        
        # Direct match
        if norm_ref in notes_by_ref:
            note = notes_by_ref[norm_ref]
            return f"**{note['title']}**\n\n{note['content']}", True
        
        # Fallback: try partial matches
        for key, note in notes_by_ref.items():
            # Match without dots
            if key.replace('.', '') == norm_ref.replace('.', ''):
                return f"**{note['title']}**\n\n{note['content']}", True
            # Match parent section (5.1 -> 5)
            if norm_ref.startswith(key + '.') or key.startswith(norm_ref + '.'):
                return f"**{note['title']}**\n\n{note['content']}", True
        
        return "", False
    
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
            
            # Convert to DataFrame for display
            df = pd.DataFrame(items)
            
            # Reorder columns
            column_order = ["item_code", "item_name", "value", "notes_ref"]
            existing_cols = [c for c in column_order if c in df.columns]
            df = df[existing_cols]
            
            # Rename columns for display
            column_names = {
                "item_code": "Mã",
                "item_name": "Chỉ tiêu",
                "value": "Giá trị",
                "notes_ref": "TM",
            }
            df = df.rename(columns=column_names)  # type: ignore
            
            # Format value column with thousand separators
            if "Giá trị" in df.columns:
                df["Giá trị"] = df["Giá trị"].apply(
                    lambda x: f"{x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else x
                )
            
            # Collect unique notes refs for this table
            notes_refs = []
            if "TM" in df.columns:
                notes_refs = [str(ref) for ref in df["TM"].dropna().unique() if ref and str(ref).strip()]
            
            # Display table without row selection
            st.dataframe(
                df,
                width='stretch',
                hide_index=True,
                height=400,
            )
            
            # TM lookup section - prominently displayed if there are notes
            if notes_refs:
                st.markdown("##### 📝 Xem Thuyết minh")
                
                # Show available notes count
                available_count = sum(1 for ref in notes_refs if lookup_note_content(ref)[1])
                if available_count < len(notes_refs):
                    st.caption(f"📊 Tìm thấy {available_count}/{len(notes_refs)} thuyết minh")
                
                # Selectbox to choose which TM to view
                tm_col1, tm_col2 = st.columns([1, 3])
                with tm_col1:
                    selected_tm = st.selectbox(
                        "Chọn TM",
                        options=[""] + notes_refs,
                        format_func=lambda x: f"TM {x}" if x else "-- Chọn thuyết minh --",
                        key=f"tm_select_{key}",
                        label_visibility="collapsed",
                    )
                
                # Show selected TM content
                if selected_tm:
                    with st.container(border=True):
                        note_content, found = lookup_note_content(selected_tm)
                        
                        st.markdown(f"#### Thuyết minh {selected_tm}")
                        
                        if found and note_content:
                            st.markdown(note_content[:5000])
                            if len(note_content) > 5000:
                                st.caption("_(Nội dung đã được rút gọn)_")
                        else:
                            st.warning(f"Không tìm thấy nội dung chi tiết cho TM {selected_tm}")
                            st.caption("Thử xem toàn bộ thuyết minh trong phần mở rộng bên dưới.")
            
            # Export buttons
            st.markdown("---")
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


def render_notes_modal(notes_content: str) -> None:
    """
    Render notes in an expandable section.
    """
    if not notes_content:
        return
    
    preview_limit = 50_000

    with st.expander("📝 Thuyết minh báo cáo tài chính", expanded=False):
        # Show as raw markdown for debugging
        st.code(notes_content[:preview_limit], language="markdown")
        if len(notes_content) > preview_limit:
            st.caption(f"Showing first {preview_limit} characters")


def render_error_with_retry(
    error_message: str,
    on_retry: Optional[Callable[[], None]] = None
) -> None:
    """
    Render an error message with retry button.
    """
    st.error(f"❌ {error_message}")
    
    if on_retry:
        if st.button("🔄 Thử lại", key="retry_button"):
            on_retry()


def render_raw_ocr(markdown_content: Optional[str] = None, markdown_path: Optional[str] = None) -> None:
    """
    Render raw OCR markdown in a collapsible section.
    """
    preview_limit = 20_000

    if not markdown_content and markdown_path:
        try:
            with open(markdown_path, "r", encoding="utf-8") as f:
                markdown_content = f.read(preview_limit)
        except Exception:
            markdown_content = None

    with st.expander("📄 Raw OCR Output (Markdown)", expanded=False):
        if markdown_path:
            st.caption(f"Saved at: {markdown_path}")
        if markdown_content:
            st.code(markdown_content, language="markdown")
            if len(markdown_content) >= preview_limit:
                st.caption(f"Showing first {preview_limit} characters")
        else:
            st.info("Không có OCR markdown để hiển thị.")