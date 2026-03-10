import os


def disable_langsmith_tracing() -> None:
    """Force-disable LangSmith/LangChain tracing for this process."""
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_CALLBACKS_BACKGROUND"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGSMITH_API_KEY"] = ""
    os.environ["LANGCHAIN_ENDPOINT"] = ""
    os.environ["LANGSMITH_ENDPOINT"] = ""


disable_langsmith_tracing()
