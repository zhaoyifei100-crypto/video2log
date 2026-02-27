"""
探测器基类模块

所有 CV 检测器必须继承 BaseDetector
CV 描述使用 [LLM_DESC] 标记写在类 docstring 中
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np


@dataclass
class DetectionResult:
    """检测结果统一格式"""

    is_suspicious: bool  # 是否可疑
    confidence: float  # 置信度 0.0-1.0
    description: str  # 文字描述（供 LLM 阅读）
    metadata: Dict[str, Any]  # 原始数据（供二次分析）
    alert_reason: Optional[str] = None  # 触发警报的原因


def normalize_region(
    region: Optional[list], frame_height: int, frame_width: int
) -> Optional[Tuple[int, int, int, int]]:
    """
    将归一化坐标转换为像素坐标

    Args:
        region: [x1, y1, x2, y2] 归一化坐标(0.0-1.0)
        frame_height: 帧高度
        frame_width: 帧宽度

    Returns:
        (x1, y1, x2, y2) 像素坐标
    """
    if not region or len(region) != 4:
        return None

    x1, y1, x2, y2 = region

    # 归一化坐标转换为像素
    x1_px = int(x1 * frame_width)
    y1_px = int(y1 * frame_height)
    x2_px = int(x2 * frame_width)
    y2_px = int(y2 * frame_height)

    # 确保坐标在有效范围内
    x1_px = max(0, min(x1_px, frame_width))
    y1_px = max(0, min(y1_px, frame_height))
    x2_px = max(0, min(x2_px, frame_width))
    y2_px = max(0, min(y2_px, frame_height))

    # 确保 x1 <= x2, y1 <= y2
    if x1_px > x2_px:
        x1_px, x2_px = x2_px, x1_px
    if y1_px > y2_px:
        y1_px, y2_px = y2_px, y1_px

    # 确保至少有 1 像素宽度/高度
    if x1_px == x2_px:
        if x1_px > 0:
            x1_px -= 1
        elif x2_px < frame_width:
            x2_px += 1
    if y1_px == y2_px:
        if y1_px > 0:
            y1_px -= 1
        elif y2_px < frame_height:
            y2_px += 1

    return (x1_px, y1_px, x2_px, y2_px)


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

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Args:
            params: LLM 生成的参数配置
        """
        self.params = params or {}
        self.name = self.__class__.__name__.replace("Detector", "").lower()

    @abstractmethod
    def detect(
        self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None
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
