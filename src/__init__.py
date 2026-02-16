"""video2log - 视觉监控 + LLM 分析"""

__version__ = "0.1.0"

from .config import config
from .logger import logger
from .llm_client import LLMClient, get_llm_client
from .vision import (
    VisionProcessor,
    VisionMode,
    VisionState,
    VisionResult,
    VisionContext,
)
from .detectors import (
    get_detector,
    get_available_detectors,
    build_llm_menu,
    BaseDetector,
    DetectionResult,
)

__all__ = [
    # 配置与工具
    "config",
    "logger",
    "LLMClient",
    "get_llm_client",
    # 视觉处理
    "VisionProcessor",
    "VisionMode",
    "VisionState",
    "VisionResult",
    "VisionContext",
    # 检测器
    "get_detector",
    "get_available_detectors",
    "build_llm_menu",
    "BaseDetector",
    "DetectionResult",
]
