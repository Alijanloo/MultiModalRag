import os
from pathlib import Path

from multimodal_rag.frameworks.logging_config import setup_logging, get_logger

setup_logging(
    log_level=os.getenv("MULTIMODAL_RAG_LOG_LEVEL", "INFO"),
    console_output=True,
    file_output=True,
    format_type="detailed",
    use_colors=os.getenv("MULTIMODAL_RAG_USE_COLORS", "true").lower() == "true",
)

DEFAULT_PATH = Path(os.path.realpath(__file__)).parents[1]

__all__ = ["get_logger", "setup_logging", "DEFAULT_PATH"]
