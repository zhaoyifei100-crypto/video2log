"""video2log - 定时拍照 + LLM 描述"""
__version__ = "0.1.0"

from .config import config
from .logger import logger
from .llm_client import LLMClient, get_llm_client
from .capture_timer import CaptureTimer, main
from .detector import BlackScreenDetector, ScreenDetectResult, MultiScreenResult

__all__ = [
    'config',
    'logger',
    'LLMClient',
    'get_llm_client',
    'CaptureTimer',
    'main',
    'BlackScreenDetector',
    'DetectResult'
]
