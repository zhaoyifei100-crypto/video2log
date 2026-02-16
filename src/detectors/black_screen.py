"""
黑屏检测器 - 相对亮度变化检测
以 INIT 阶段为基准，检测亮度下降比例
支持归一化区域坐标
"""

import cv2
import numpy as np
from .base import BaseDetector, DetectionResult


def normalize_region(region, frame_height, frame_width):
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
    x1 = int(x1 * frame_width)
    y1 = int(y1 * frame_height)
    x2 = int(x2 * frame_width)
    y2 = int(y2 * frame_height)

    # 确保坐标在有效范围内
    x1 = max(0, min(x1, frame_width))
    y1 = max(0, min(y1, frame_height))
    x2 = max(0, min(x2, frame_width))
    y2 = max(0, min(y2, frame_height))

    # 确保 x1 < x2, y1 < y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return (x1, y1, x2, y2)


class BlackScreenDetector(BaseDetector):
    """
    [LLM_DESC]
    能力：检测屏幕是否黑屏、关闭或信号丢失
    原理：以初始画面为基准，检测亮度下降比例
    场景：
      - 电视关闭（亮度骤降）
      - 显示器断电
      - 监控摄像头被遮挡
      - 画面信号中断（无输入源）
    参数：
      - brightness_drop: 亮度下降阈值 (0.0-1.0, 默认0.5)
        说明：当前亮度比基准亮度下降超过此比例视为黑屏
        示例：0.5 表示亮度下降 50% 触发警报
      - dark_ratio: 暗像素比例 (0.0-1.0, 默认0.8)
        说明：暗像素占比超过此值视为黑屏（辅助判断）
      - region: 检测区域 [x1, y1, x2, y2] (可选)
        说明：只检测画面中的特定区域。使用归一化坐标(0.0-1.0)
        示例: [0.3, 0.2, 0.7, 0.8] 表示画面中间区域（从左30%，上20%，到从左70%，上80%）
    输出：
      - current_brightness: 当前亮度
      - baseline_brightness: 基准亮度
      - brightness_drop_ratio: 亮度下降比例
      - is_significant_drop: 是否显著下降
    [/LLM_DESC]
    """

    DEFAULT_PARAMS = {
        "brightness_drop": 0.5,  # 亮度下降 50% 视为黑屏
        "dark_ratio": 0.8,
        "region": None,
    }

    def __init__(self, params=None):
        super().__init__(params)
        self._baseline_brightness = None  # 基准亮度（INIT 阶段）

    def detect(
        self, frame: np.ndarray, prev_frame: np.ndarray = None
    ) -> DetectionResult:
        # 获取参数
        brightness_drop_threshold = float(
            self.params.get("brightness_drop", self.DEFAULT_PARAMS["brightness_drop"])
        )
        dark_ratio_threshold = float(
            self.params.get("dark_ratio", self.DEFAULT_PARAMS["dark_ratio"])
        )
        region = self.params.get("region", self.DEFAULT_PARAMS["region"])

        # 裁剪区域（如果指定）- 支持归一化坐标
        if region and len(region) == 4:
            h, w = frame.shape[:2]
            coords = normalize_region(region, h, w)
            if coords:
                x1, y1, x2, y2 = coords
                frame = frame[y1:y2, x1:x2]

        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 计算当前亮度
        current_brightness = float(np.mean(gray))

        # 如果还没有基准亮度，设为当前亮度（INIT 阶段）
        if self._baseline_brightness is None:
            self._baseline_brightness = current_brightness
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description=f"初始化完成，基准亮度: {current_brightness:.1f}",
                metadata={
                    "current_brightness": current_brightness,
                    "baseline_brightness": current_brightness,
                    "brightness_drop_ratio": 0.0,
                    "is_significant_drop": False,
                },
                alert_reason=None,
            )

        # 计算亮度下降比例
        # drop_ratio = (baseline - current) / baseline
        if self._baseline_brightness > 0:
            brightness_drop_ratio = (
                self._baseline_brightness - current_brightness
            ) / self._baseline_brightness
        else:
            brightness_drop_ratio = 0.0

        # 计算暗像素比例（辅助判断）
        # 使用绝对阈值判断暗像素（30 是经验值）
        dark_pixels = np.sum(gray < 30)
        dark_ratio = dark_pixels / gray.size

        # 判断是否可疑：亮度显著下降 或 暗像素过多
        is_significant_drop = brightness_drop_ratio > brightness_drop_threshold
        is_dark = dark_ratio > dark_ratio_threshold

        is_suspicious = is_significant_drop or is_dark

        # 计算置信度
        if is_suspicious:
            if is_significant_drop:
                confidence = min(brightness_drop_ratio, 1.0)
                reason = f"亮度下降 {brightness_drop_ratio:.1%} (基准: {self._baseline_brightness:.1f} → 当前: {current_brightness:.1f})"
            else:
                confidence = dark_ratio
                reason = f"暗像素比例 {dark_ratio:.1%} 超过阈值"
        else:
            confidence = 0.0
            reason = None

        # 构建描述
        description = (
            f"亮度: {current_brightness:.1f} / {self._baseline_brightness:.1f} "
            f"(下降 {brightness_drop_ratio:.1%}), 暗像素: {dark_ratio:.1%}"
        )

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            description=description,
            metadata={
                "current_brightness": current_brightness,
                "baseline_brightness": self._baseline_brightness,
                "brightness_drop_ratio": brightness_drop_ratio,
                "is_significant_drop": is_significant_drop,
                "dark_ratio": dark_ratio,
                "brightness_drop_threshold": brightness_drop_threshold,
            },
            alert_reason=reason,
        )

    def reset_baseline(self):
        """重置基准亮度（用于重新开始监控）"""
        self._baseline_brightness = None
