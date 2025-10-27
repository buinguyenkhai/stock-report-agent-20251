import json
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from state import StockReportState
from nodes import (
    process_query_node,
    extract_report_link_node,
    prepare_next_extraction_node,
    check_extraction_result,
    should_continue_extraction,
    collect_result_node,
    ask_user_for_clarification_node,
)

load_dotenv()

graph_builder = StateGraph(StockReportState)

# Nodes

graph_builder.add_node("process_query", process_query_node)
graph_builder.add_node("prepare_next_extraction", prepare_next_extraction_node)
graph_builder.add_node("extract_report_link", extract_report_link_node)
graph_builder.add_node("ask_user", ask_user_for_clarification_node)
graph_builder.add_node("collect_result", collect_result_node)

# Edges

graph_builder.add_edge(START, "process_query")
graph_builder.add_conditional_edges(
    "process_query",
    should_continue_extraction,
    {"continue": "prepare_next_extraction", "end_extraction": END}
)
graph_builder.add_conditional_edges(
    "collect_result",
    should_continue_extraction,
    {"continue": "prepare_next_extraction", "end_extraction": END}
)
graph_builder.add_edge("prepare_next_extraction", "extract_report_link")
graph_builder.add_edge("ask_user", "collect_result")
graph_builder.add_conditional_edges(
    "extract_report_link",
    check_extraction_result,
    {"ask_user": "ask_user", "collect": "collect_result"}
)

agent = graph_builder.compile()

query = "so sánh fpt quý 2 và quý 3 năm 2024"
final_state = agent.invoke({"query": query})

print("AGENT ĐÃ HOÀN TẤT")
with open("result.json", 'w', encoding='utf-8') as f:
    json.dump(final_state, f, ensure_ascii=False, indent=4)

# Lưu graph
with open("graph_v2.png", "wb") as f:
    f.write(agent.get_graph().draw_mermaid_png())