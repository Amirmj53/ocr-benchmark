"""Prompt builders for the three Darsio study modes.

Public surface:

    from darsio_ingest.prompts import build_messages, pack_context

    messages = build_messages(question, chunks, history, study_mode="exam")
"""

from darsio_ingest.prompts.base import (
    StudyMode,
    build_messages,
    chunk_sort_key,
    pack_context,
    render_chunk,
)
from darsio_ingest.prompts.exam import SYSTEM_PROMPT as EXAM_SYSTEM_PROMPT
from darsio_ingest.prompts.normal import SYSTEM_PROMPT as NORMAL_SYSTEM_PROMPT
from darsio_ingest.prompts.research import SYSTEM_PROMPT as RESEARCH_SYSTEM_PROMPT

__all__ = [
    "StudyMode",
    "build_messages",
    "pack_context",
    "render_chunk",
    "chunk_sort_key",
    "NORMAL_SYSTEM_PROMPT",
    "EXAM_SYSTEM_PROMPT",
    "RESEARCH_SYSTEM_PROMPT",
]
