"""
黑屏检测模块 - 支持多屏幕检测
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .screen_detector import ScreenDetector, ScreenRegion
from .logger import logger


@dataclass
class ScreenDetectResult:
    """单屏幕检测结果"""
    name: str
    is_black: bool
    avg_brightness: float
    black_ratio: float
    region: ScreenRegion


@dataclass
class MultiScreenResult:
    """多屏幕检测结果汇总"""
    results: list[ScreenDetectResult]
    all_pass: bool
    
    def summary(self) -> str:
        """生成汇总文字"""
        parts = []
        for r in self.results:
            status = "❌ 黑屏" if r.is_black else "✅ 正常"
            parts.append(f"{r.name}: {status} (亮度:{r.avg_brightness:.1f})")
        return " | ".join(parts)


class BlackScreenDetector:
    """黑屏检测器 - 支持多屏幕"""
    
    def __init__(
        self, 
        threshold: int = 30, 
        dark_pixel_ratio: float = 0.9,
        auto_detect_screens: bool = True,
        manual_regions: list = None
    ):
        """
        Args:
            threshold: 亮度阈值，低于此值认为暗 (0-255)
            dark_pixel_ratio: 暗像素判定比例
            auto_detect_screens: 是否自动检测屏幕边界 (调用 Qwen)
            manual_regions: 手动指定的屏幕区域 [{"name": "TV1", "x1":..., "y1":..., "x2":..., "y2":...}]
        """
        self.threshold = threshold
        self.dark_pixel_ratio = dark_pixel_ratio
        self.auto_detect_screens = auto_detect_screens
        self.manual_regions = manual_regions or []
        
        # 屏幕检测器
        self.screen_detector = ScreenDetector()
        
        # 缓存的屏幕区域
        self._cached_regions: list[ScreenRegion] = None
    
    def detect(self, frame: np.ndarray, image_path: str = None) -> MultiScreenResult:
        """
        检测多屏幕是否黑屏
        
        Args:
            frame: BGR 图像 (OpenCV 格式)
            image_path: 图像路径 (用于 LLM 检测屏幕边界)
            
        Returns:
            MultiScreenResult
        """
        # 获取屏幕区域
        regions = self._get_regions(frame, image_path)
        
        if not regions:
            # 没有检测到屏幕，做全屏检测
            logger.warning("未检测到屏幕区域，执行全屏检测")
            result = self._detect_single(frame, "FullScreen")
            return MultiScreenResult(results=[result], all_pass=not result.is_black)
        
        # 对每个屏幕区域做检测
        results = []
        for region in regions:
            screen_frame = region.crop(frame)
            result = self._detect_single(screen_frame, region.name)
            result.region = region
            results.append(result)
        
        all_pass = all(not r.is_black for r in results)
        
        return MultiScreenResult(results=results, all_pass=all_pass)
    
    def _get_regions(self, frame: np.ndarray, image_path: str = None) -> list[ScreenRegion]:
        """获取屏幕区域"""
        # 优先使用手动配置
        if self.manual_regions:
            return [
                ScreenRegion(r["name"], r["x1"], r["y1"], r["x2"], r["y2"])
                for r in self.manual_regions
            ]
        
        # 自动检测
        if self.auto_detect_screens and image_path:
            if self._cached_regions is None:
                self._cached_regions = self.screen_detector.detect_screens(image_path)
            return self._cached_regions
        
        return []
    
    def _detect_single(self, frame: np.ndarray, name: str) -> ScreenDetectResult:
        """检测单帧是否黑屏"""
        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 计算平均亮度
        avg_brightness = np.mean(gray)
        
        # 计算暗像素比例
        dark_pixels = np.sum(gray < self.threshold)
        total_pixels = gray.size
        black_ratio = dark_pixels / total_pixels
        
        # 判定黑屏
        is_black = (avg_brightness < self.threshold) or (black_ratio > self.dark_pixel_ratio)
        
        if is_black:
            logger.warning(f"{name}: 黑屏! 亮度={avg_brightness:.1f}, 暗像素={black_ratio:.1%}")
        
        return ScreenDetectResult(
            name=name,
            is_black=is_black,
            avg_brightness=avg_brightness,
            black_ratio=black_ratio,
            region=None
        )
    
    def detect_from_file(self, image_path: str) -> MultiScreenResult:
        """从文件检测"""
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(frame, image_path)
    
    def clear_cache(self):
        """清除屏幕区域缓存"""
        self._cached_regions = None
        self.screen_detector.clear_cache()
