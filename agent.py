import json
from dotenv import load_dotenv
import runtime_env  # noqa: F401
from langgraph.graph import StateGraph, START, END

from state import StockReportState
from logger import setup_logging, get_logger
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

load_dotenv()
setup_logging()
logger = get_logger(__name__)

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

agent = graph_builder.compile()

logger.info("Agent compiled successfully")
print('Xin chào, tôi là trợ lý báo cáo tài chính cổ phiếu Việt Nam. Hãy nhập truy vấn của bạn!')
query = input("Truy vấn: ")

logger.info(f"Received query: {query}")
final_state = agent.invoke({"query": query})

logger.info("AGENT ĐÃ HOÀN TẤT")
print("AGENT ĐÃ HOÀN TẤT")
with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(final_state, f, ensure_ascii=False, indent=4)

# Lưu graph
with open("readme_img/graph_v2.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())
