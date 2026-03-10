from __future__ import annotations

from typing import Any, Dict


DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-lite-preview-09-2025"
DEFAULT_MARKER_LLM_MODEL = DEFAULT_LLM_MODEL


LLM_MODEL_OPTIONS = [
    ("Qwen3 235B A22B", "qwen/qwen3-235b-a22b-2507"),
    ("Gemini 2.5 Flash Lite", DEFAULT_LLM_MODEL),
]


TASK_LLM_SETTINGS: Dict[str, Dict[str, Any]] = {
    "item_matching": {
        "temperature": 0.0,
        "max_tokens": 500,
        "timeout": 60,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.0,
        "max_retries": 3,
    },
    "unit_detection": {
        "temperature": 0.0,
        "max_tokens": 200,
        "timeout": 60,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_retries": 3,
    },
    "parsing": {
        "temperature": 0.0,
        "max_tokens": 64000,
        "timeout": 600,
        "top_p": 0.95,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.1,
        "max_retries": 3,
    },
    "query_processing": {
        "temperature": 0.0,
        "max_tokens": 500,
        "timeout": 60,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_retries": 3,
    },
    "notes_tables_by_ref": {
        "temperature": 0.0,
        "max_tokens": 64000,
        "timeout": 300,
        "top_p": 0.9,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_retries": 3,
    },
}


def get_task_llm_settings(task: str) -> Dict[str, Any]:
    return dict(TASK_LLM_SETTINGS.get(task, {}))
