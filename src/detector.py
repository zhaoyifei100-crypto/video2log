"""
黑屏检测模块
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class DetectResult:
    """检测结果"""
    is_black: bool          # 当前帧是否黑屏
    avg_brightness: float   # 平均亮度 (0-255)
    black_ratio: float      # 暗像素比例


class BlackScreenDetector:
    """黑屏检测器"""
    
    def __init__(self, threshold: int = 30, dark_pixel_ratio: float = 0.9):
        """
        Args:
            threshold: 亮度阈值，低于此值认为暗 (0-255)
            dark_pixel_ratio: 暗像素判定比例
        """
        self.threshold = threshold
        self.dark_pixel_ratio = dark_pixel_ratio
    
    def detect(self, frame: np.ndarray) -> DetectResult:
        """
        检测单帧是否黑屏
        
        Args:
            frame: BGR 图像 (OpenCV 格式)
            
        Returns:
            DetectResult
        """
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算平均亮度
        avg_brightness = np.mean(gray)
        
        # 计算暗像素比例 (亮度 < threshold)
        dark_pixels = np.sum(gray < self.threshold)
        total_pixels = gray.size
        black_ratio = dark_pixels / total_pixels
        
        # 判定黑屏: 平均亮度低 或 暗像素比例高
        is_black = (avg_brightness < self.threshold) or (black_ratio > self.dark_pixel_ratio)
        
        return DetectResult(
            is_black=is_black,
            avg_brightness=avg_brightness,
            black_ratio=black_ratio
        )
    
    def detect_from_file(self, image_path: str) -> DetectResult:
        """从文件检测"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(frame)
