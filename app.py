"""
Stock Report Agent - Streamlit UI
"""

import streamlit as st
from typing import Dict, Any, Optional
import time
import json

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from state import StockReportState
from logger import setup_logging, get_logger
from config import settings, OCR_ENGINE_OPTIONS
from nodes import (
    process_query_node,
    extract_report_link_node,
    prepare_next_extraction_node,
    check_extraction_result,
    should_continue_extraction,
    collect_result_node,
    ask_user_for_clarification_node,
    generate_final_response_node,
    ocr_report_node,
    parse_report_node
)
from ui.components import (
    render_model_selector,
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
    page_icon="📊",
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
    graph_builder.add_node("ask_user", ask_user_for_clarification_node)
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
    graph_builder.add_edge("ask_user", "ocr_report")
    graph_builder.add_edge("ocr_report", "parse_report")
    graph_builder.add_edge("parse_report", "collect_result")
    graph_builder.add_conditional_edges(
        "extract_report_link",
        check_extraction_result,
        {"ask_user": "ask_user", "collect": "ocr_report"}
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
        initial_state = {
            "query": query,
            "ocr_engine": ocr_engine,
            "llm_model": llm_model,
        }
        
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
    st.title("📊 Stock Report Agent")
    st.caption("Trích xuất và hiển thị báo cáo tài chính Việt Nam")
    
    # Model selectors
    ocr_engine, llm_model = render_model_selector()
    
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
    
    # Chat input
    query = st.chat_input("Nhập mã chứng khoán và kỳ báo cáo (VD: FPT Q3 2024)")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Process with agent
        with st.chat_message("assistant"):
            progress_container = st.empty()
            result_container = st.container()
            
            with progress_container.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown("**Đang xử lý...**")
                with col2:
                    if st.button("❌ Hủy", key="cancel_btn", type="secondary"):
                        st.session_state.cancel_requested = True
                        st.session_state.is_processing = False
                        st.rerun()
            
            # Check if cancelled before running
            if st.session_state.cancel_requested:
                st.session_state.cancel_requested = False
                st.warning("Đã hủy xử lý.")
            else:
                # Run agent
                st.session_state.is_processing = True
                final_state = run_agent_with_progress(query, ocr_engine, llm_model, progress_container)
                st.session_state.is_processing = False
                
                # Display results
                with result_container:
                    if final_state.get("error_message"):
                        render_error_with_retry(
                            final_state["error_message"],
                            on_retry=lambda: st.rerun()
                        )
                    else:
                        # Get parsed data from saved file or state
                        parsed_data = final_state.get("parsed_data")
                        
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
                                    parsed_data = json.load(f)
                        
                        if parsed_data:
                            # Show report header
                            render_report_header(
                                stock_code=final_state.get("stock_code", ""),
                                period=final_state.get("period", ""),
                                year=final_state.get("year", 0),
                                quarter=final_state.get("quarter"),
                                consolidation=final_state.get("consolidation_status"),
                                unit=parsed_data.get("metadata", {}).get("unit", "VND"),
                                pdf_link=final_state.get("report_link"),
                            )
                            
                            st.divider()
                            
                            # Show financial tables
                            render_financial_tables(parsed_data)
                            
                            # Show notes if available
                            notes = parsed_data.get("notes", [])
                            if notes:
                                render_notes_modal(notes)
                            
                            # Show raw OCR (collapsible)
                            ocr_content = final_state.get("ocr_markdown_content")
                            if ocr_content:
                                render_raw_ocr(ocr_content)
                            
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
            
            notes = parsed_data.get("notes", [])
            if notes:
                render_notes_modal(notes)


if __name__ == "__main__":
    main()
