"""
OpenCV 预处理模块 - 动态视觉用
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


def capture_screen(output_path: str) -> bool:
    """
    捕获桌面截图并保存

    Args:
        output_path: 保存路径

    Returns:
        bool: 是否成功
    """
    try:
        import pyautogui

        screenshot = pyautogui.screenshot()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True
    except Exception as e:
        print(f"截图失败: {e}")
        return False


@dataclass
class FrameAnalysis:
    """帧分析结果"""

    avg_brightness: float  # 平均亮度
    dark_ratio: float  # 暗像素比例
    motion_score: float  # 运动得分 (帧差)
    is_anomaly: bool  # 是否异常


class OpenCVHelper:
    """OpenCV 预处理工具"""

    def __init__(
        self,
        brightness_threshold: int = 30,
        dark_ratio_threshold: float = 0.9,
        motion_threshold: float = 1000,
    ):
        """
        Args:
            brightness_threshold: 亮度阈值
            dark_ratio_threshold: 暗像素比例阈值
            motion_threshold: 运动检测阈值
        """
        self.brightness_threshold = brightness_threshold
        self.dark_ratio_threshold = dark_ratio_threshold
        self.motion_threshold = motion_threshold
        self._prev_frame = None

    def analyze_frame(self, frame: np.ndarray) -> FrameAnalysis:
        """分析单帧

        Returns:
            FrameAnalysis
        """
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. 亮度计算
        avg_brightness = np.mean(gray)

        # 2. 暗像素比例
        dark_pixels = np.sum(gray < self.brightness_threshold)
        dark_ratio = dark_pixels / gray.size

        # 3. 运动检测 (帧差)
        motion_score = 0
        if self._prev_frame is not None:
            if self._prev_frame.shape != gray.shape:
                self._prev_frame = gray
            else:
                diff = cv2.absdiff(gray, self._prev_frame)
                motion_score = np.sum(diff)

        self._prev_frame = gray.copy()

        # 4. 判定异常
        is_anomaly = (
            avg_brightness < self.brightness_threshold
            or dark_ratio > self.dark_ratio_threshold
        )

        return FrameAnalysis(
            avg_brightness=avg_brightness,
            dark_ratio=dark_ratio,
            motion_score=motion_score,
            is_anomaly=is_anomaly,
        )

    def detect_screen_regions(self, frame: np.ndarray) -> list:
        """简单边缘检测找屏幕区域 (备用)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 高斯模糊 + 边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 找轮廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:  # 过滤小区域
                x, y, w, h = cv2.boundingRect(cnt)
                regions.append({"x": x, "y": y, "w": w, "h": h})

        return regions

    def reset(self):
        """重置状态"""
        self._prev_frame = None
