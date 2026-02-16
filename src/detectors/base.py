"""
探测器基类模块

所有 CV 检测器必须继承 BaseDetector
CV 描述使用 [LLM_DESC] 标记写在类 docstring 中
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class DetectionResult:
    """检测结果统一格式"""

    is_suspicious: bool  # 是否可疑
    confidence: float  # 置信度 0.0-1.0
    description: str  # 文字描述（供 LLM 阅读）
    metadata: Dict[str, Any]  # 原始数据（供二次分析）
    alert_reason: Optional[str] = None  # 触发警报的原因


class BaseDetector(ABC):
    """
    CV 探测器基类

    子类必须在 docstring 中包含 [LLM_DESC] 标记，供 LLM 选择

    Example:
        \"\"\"
        [LLM_DESC]
        能力：检测屏幕是否黑屏
        场景：电视关闭、显示器断电、信号丢失
        参数：
          - threshold: 亮度阈值 (0-255, 默认30)
          - dark_ratio: 暗像素比例 (0.0-1.0, 默认0.9)
        [/LLM_DESC]
        \"\"\"
    """

    def __init__(self, params: Dict[str, Any] = None):
        """
        Args:
            params: LLM 生成的参数配置
        """
        self.params = params or {}
        self.name = self.__class__.__name__.replace("Detector", "").lower()

    @abstractmethod
    def detect(
        self, frame: np.ndarray, prev_frame: np.ndarray = None
    ) -> DetectionResult:
        """
        执行检测

        Args:
            frame: 当前帧 (BGR 格式)
            prev_frame: 前一帧（可选，用于运动检测）

        Returns:
            DetectionResult
        """
        pass

    def get_llm_description(self) -> str:
        """提取 docstring 中的 [LLM_DESC] 内容"""
        doc = self.__doc__ or ""
        import re

        match = re.search(r"\[LLM_DESC\](.*?)\[/LLM_DESC\]", doc, re.DOTALL)
        if match:
            return match.group(1).strip()
        return f"{self.name} 检测器（未提供描述）"
