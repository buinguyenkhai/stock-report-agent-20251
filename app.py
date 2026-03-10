"""
Stock Report Agent - Streamlit UI
"""

import streamlit as st
from typing import Dict, Any, Optional, cast
import time
import json

from dotenv import load_dotenv
import runtime_env  # noqa: F401
from langgraph.graph import StateGraph, START, END

from state import StockReportState
from logger import setup_logging, get_logger
from config import settings
from nodes import (
    process_query_node,
    extract_report_link_node,
    prepare_next_extraction_node,
    check_extraction_result,
    should_continue_extraction,
    collect_result_node,
    generate_final_response_node,
    ocr_report_node,
    parse_report_node
)
from ui.components import (
    render_chat_input,
    render_progress_step,
    render_financial_tables,
    render_notes_modal,
    render_report_header,
    render_error_with_retry,
    render_raw_ocr,
)

load_dotenv()
setup_logging()
logger = get_logger(__name__)

# Page config
st.set_page_config(
    page_title="Stock Report Agent",
    page_icon="🤖",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .step-done { color: #28a745; }
    .step-running { color: #ffc107; }
    .step-error { color: #dc3545; }
</style>
""", unsafe_allow_html=True)


def build_agent():
    """Build and compile the LangGraph agent."""
    graph_builder = StateGraph(StockReportState)
    
    # Nodes
    graph_builder.add_node("process_query", process_query_node)
    graph_builder.add_node("prepare_next_extraction", prepare_next_extraction_node)
    graph_builder.add_node("extract_report_link", extract_report_link_node)
    graph_builder.add_node("ocr_report", ocr_report_node)
    graph_builder.add_node("parse_report", parse_report_node)
    graph_builder.add_node("collect_result", collect_result_node)
    graph_builder.add_node("generate_final_response", generate_final_response_node)
    
    # Edges
    graph_builder.add_edge(START, "process_query")
    graph_builder.add_conditional_edges(
        "process_query",
        should_continue_extraction,
        {"continue": "prepare_next_extraction", "end_extraction": "generate_final_response"}
    )
    graph_builder.add_conditional_edges(
        "collect_result",
        should_continue_extraction,
        {"continue": "prepare_next_extraction", "end_extraction": "generate_final_response"}
    )
    graph_builder.add_edge("generate_final_response", END)
    graph_builder.add_edge("prepare_next_extraction", "extract_report_link")
    graph_builder.add_edge("ocr_report", "parse_report")
    graph_builder.add_edge("parse_report", "collect_result")
    graph_builder.add_conditional_edges(
        "extract_report_link",
        check_extraction_result,
        {"ocr_report": "ocr_report", "collect": "collect_result"}
    )
    
    return graph_builder.compile()


def run_agent_with_progress(query: str, ocr_engine: str, llm_model: str, progress_container):
    """
    Run the agent with progress updates displayed in the UI.
    """
    steps = [
        {"name": "Đang phân tích yêu cầu...", "status": "pending"},
        {"name": "Đang tìm link báo cáo trên Vietstock...", "status": "pending"},
        {"name": "Đang xử lý OCR...", "status": "pending"},
        {"name": "Đang trích xuất dữ liệu...", "status": "pending"},
    ]
    
    def update_progress(step_index: int, status: str, details: Optional[str] = None):
        """Update progress display."""
        steps[step_index]["status"] = status
        if details:
            steps[step_index]["details"] = details
        
        with progress_container:
            for step in steps:
                render_progress_step(step["name"], step["status"], step.get("details"))
    
    try:
        # Build agent
        agent = build_agent()
        
        # Step 1: Process query
        update_progress(0, "running")
        time.sleep(0.3)  # Small delay for visual feedback
        
        # Run the full agent
        initial_state = cast(
            StockReportState,
            {
                "query": query,
                "ocr_engine": ocr_engine,
                "llm_model": llm_model,
            },
        )
        
        # Simulate step-by-step progress
        update_progress(0, "done")
        update_progress(1, "running")
        
        # Run agent
        final_state = agent.invoke(initial_state)
        
        # Update remaining steps based on result
        if final_state.get("error_message"):
            for i in range(1, 4):
                if steps[i]["status"] == "pending":
                    steps[i]["status"] = "error"
            update_progress(1, "error")
        else:
            update_progress(1, "done")
            update_progress(2, "done", f"Engine: {ocr_engine}")
            update_progress(3, "done")
        
        return final_state
        
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        for i, step in enumerate(steps):
            if step["status"] in ["pending", "running"]:
                steps[i]["status"] = "error"
        update_progress(0, "error")
        return {"error_message": str(e)}


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("Stock Report Agent")
    
    # Intro text
    st.markdown("""
    Hệ thống AI tự động trích xuất dữ liệu từ báo cáo tài chính doanh nghiệp Việt Nam.
    
    **Bạn có thể:**
    - Trích xuất BCTC theo mã cổ phiếu và kỳ báo cáo
    - Xem Bảng cân đối, Kết quả HĐKD, Lưu chuyển tiền tệ
    - Xuất dữ liệu ra CSV, Excel, JSON
    """)
    
    st.divider()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "is_processing" not in st.session_state:
        st.session_state.is_processing = False
    if "cancel_requested" not in st.session_state:
        st.session_state.cancel_requested = False
    
    # Model selectors, PDF uploader, and chat input
    ocr_engine, llm_model, uploaded_pdf, query, submitted, stop_requested = render_chat_input(
        is_processing=st.session_state.is_processing
    )
    
    # Handle stop request
    if stop_requested:
        st.session_state.cancel_requested = True
        st.session_state.is_processing = False
        st.rerun()
    
    # Handle form submission
    if submitted:
        # If both PDF and query provided, prioritize PDF and notify user
        if uploaded_pdf and query:
            st.info("📎 PDF được ưu tiên xử lý. Truy vấn văn bản sẽ bị bỏ qua.")
         
        # Process PDF uploads if any (takes priority over text query)
        if uploaded_pdf:
            with st.spinner("Đang xử lý PDF..."):
                import tempfile
                import os
                import shutil
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    try:
                        try:
                            uploaded_pdf.seek(0)
                        except Exception:
                            pass
                        shutil.copyfileobj(uploaded_pdf, tmp)
                        tmp_path = tmp.name
                    finally:
                        try:
                            tmp.flush()
                        except Exception:
                            pass

                try:
                    from services.ocr import get_ocr_service
                    from services.pipeline import create_pipeline
                    
                    ocr_service = get_ocr_service(ocr_engine)
                    markdown_content = ocr_service.process_pdf(tmp_path)

                    os.makedirs(settings.reports_output_dir, exist_ok=True)
                    safe_name = uploaded_pdf.name.replace(" ", "_").replace(".pdf", "")
                    ocr_md_path = os.path.join(settings.reports_output_dir, f"UPLOAD_{safe_name}_{int(time.time())}.md")
                    with open(ocr_md_path, "w", encoding="utf-8") as f:
                        f.write(markdown_content)
                    ocr_preview = markdown_content[:20_000]
                    
                    pipeline = create_pipeline(
                        mode="separate",
                        extract_notes=True,
                        extract_metadata=True,
                        extractor_model=llm_model,
                        parser_model=llm_model,
                    )
                    parsed_report = pipeline.process(markdown_content)
                    parsed_data = pipeline.to_dict(parsed_report)

                    st.session_state.current_result = {
                        "state": {
                            "stock_code": uploaded_pdf.name.replace(".pdf", ""),
                            "period": "Tải lên",
                            "year": 0,
                            "quarter": None,
                            "consolidation_status": None,
                            "report_link": None,
                            "ocr_markdown_path": ocr_md_path,
                            "ocr_markdown_preview": ocr_preview,
                            "ocr_engine": ocr_engine,
                            "llm_model": llm_model,
                        },
                        "parsed_data": parsed_data,
                    }
                    st.success("✅ Trích xuất thành công 1 báo cáo!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Lỗi xử lý {getattr(uploaded_pdf, 'name', 'PDF')}: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
        
        # Process text query if provided
        elif query:
            # Store query for processing and set processing flag
            st.session_state.pending_query = query
            st.session_state.pending_ocr_engine = ocr_engine
            st.session_state.pending_llm_model = llm_model
            st.session_state.is_processing = True
            st.rerun()
    
    # Handle pending query processing
    if st.session_state.is_processing and "pending_query" in st.session_state:
        query = st.session_state.pending_query
        ocr_engine = st.session_state.pending_ocr_engine
        llm_model = st.session_state.pending_llm_model
        
        # Add user message if not already added
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != query:
            st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Process with agent
        with st.chat_message("assistant"):
            progress_container = st.empty()
            result_container = st.container()
            
            with progress_container.container():
                st.markdown("**Đang xử lý...**")
            
            # Check if cancelled
            if st.session_state.cancel_requested:
                st.session_state.cancel_requested = False
                st.session_state.is_processing = False
                del st.session_state.pending_query
                st.warning("Đã hủy xử lý.")
            else:
                # Run agent
                final_state = run_agent_with_progress(query, ocr_engine, llm_model, progress_container)
                st.session_state.is_processing = False
                del st.session_state.pending_query
                
                # Display results
                with result_container:
                    if final_state.get("error_message"):
                        render_error_with_retry(
                            final_state["error_message"],
                            on_retry=lambda: st.rerun()
                        )
                    else:
                        # Get parsed data from saved file or state
                        parsed_data_any = final_state.get("parsed_data")
                        parsed_data: Optional[Dict[str, Any]] = parsed_data_any if isinstance(parsed_data_any, dict) else None
                        
                        # If not in state, try to load from file
                        if not parsed_data:
                            stock_code = final_state.get("stock_code", "UNKNOWN")
                            year = final_state.get("year", "NA")
                            period = final_state.get("period", "NA")
                            
                            import os
                            filename = f"{stock_code}_{year}_{period}.json"
                            filepath = os.path.join(settings.parsed_output_dir, filename)
                            
                            if os.path.exists(filepath):
                                with open(filepath, "r", encoding="utf-8") as f:
                                    loaded = json.load(f)
                                    if isinstance(loaded, dict):
                                        parsed_data = loaded
                        
                        if parsed_data:
                            warnings = parsed_data.get("status", {}).get("warnings") or []
                            if warnings:
                                st.warning("\n".join([str(w) for w in warnings]))

                            # Show any agent notification
                            notification = final_state.get("notification")
                            if notification:
                                text = str(notification).strip()
                                if text:
                                    if ("⚠️" in text) or ("bỏ qua" in text.lower()):
                                        st.warning(text)
                                    else:
                                        st.info(text)

                            # Show report header
                            year_val = final_state.get("year")
                            quarter_val = final_state.get("quarter")
                            year_int = int(year_val) if isinstance(year_val, int) or (isinstance(year_val, str) and str(year_val).isdigit()) else 0
                            quarter_int = int(quarter_val) if isinstance(quarter_val, int) or (isinstance(quarter_val, str) and str(quarter_val).isdigit()) else None

                            render_report_header(
                                stock_code=str(final_state.get("stock_code", "")),
                                period=str(final_state.get("period", "")),
                                year=year_int,
                                quarter=quarter_int,
                                consolidation=final_state.get("consolidation_status"),
                                unit=parsed_data.get("metadata", {}).get("unit", "VND"),
                                pdf_link=final_state.get("report_link"),
                            )
                            
                            st.divider()
                            
                            # Show financial tables
                            render_financial_tables(parsed_data)
                            
                            # Show notes if available
                            notes_content = parsed_data.get("notes_content", "")
                            if notes_content:
                                render_notes_modal(notes_content)
                            
                            # Show raw OCR (collapsible)
                            ocr_preview = final_state.get("ocr_markdown_preview") or final_state.get("ocr_markdown_content")
                            ocr_path = final_state.get("ocr_markdown_path")
                            if ocr_preview or ocr_path:
                                render_raw_ocr(markdown_content=ocr_preview, markdown_path=ocr_path)
                            
                            # Store result
                            st.session_state.current_result = {
                                "state": final_state,
                                "parsed_data": parsed_data,
                            }
                        else:
                            st.warning("Không thể tải dữ liệu đã trích xuất.")
                            if final_state.get("notification"):
                                st.info(final_state["notification"])
    
    # Show previous results if no new query
    elif st.session_state.current_result:
        result = st.session_state.current_result
        parsed_data = result.get("parsed_data", {})
        state = result.get("state", {})
        
        if parsed_data:
            render_report_header(
                stock_code=state.get("stock_code", ""),
                period=state.get("period", ""),
                year=state.get("year", 0),
                quarter=state.get("quarter"),
                consolidation=state.get("consolidation_status"),
                unit=parsed_data.get("metadata", {}).get("unit", "VND"),
                pdf_link=state.get("report_link"),
            )
            
            st.divider()
            render_financial_tables(parsed_data)
            
            # Show notes if available
            notes_content = parsed_data.get("notes_content", "")
            if notes_content:
                render_notes_modal(notes_content)
            
            # Show raw OCR (collapsible)
            ocr_preview = state.get("ocr_markdown_preview") or state.get("ocr_markdown_content")
            ocr_path = state.get("ocr_markdown_path")
            if ocr_preview or ocr_path:
                render_raw_ocr(markdown_content=ocr_preview, markdown_path=ocr_path)


if __name__ == "__main__":
    main()
