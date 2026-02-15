"""
屏幕检测模块 - 用 Qwen VL 识别屏幕边界
"""

import re
import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .llm_client import get_llm_client
from .logger import logger


@dataclass
class ScreenRegion:
    """屏幕区域"""
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    
    @property
    def roi(self):
        """返回 ROI 区域 (y1:y2, x1:x2)"""
        return (self.y1, self.y2, self.x1, self.x2)
    
    def crop(self, frame: np.ndarray) -> np.ndarray:
        """从帧中裁剪屏幕区域"""
        return frame[self.y1:self.y2, self.x1:self.x2]


class ScreenDetector:
    """屏幕检测器 - 用 LLM 识别屏幕边界"""
    
    def __init__(self, client=None):
        self.client = client or get_llm_client()
        self._cached_regions = None
        self._cache_image = None
    
    PROMPT = """请检测图中所有电视/显示器屏幕的位置。
要求：
1. 只返回屏幕显示区域（不包括黑色边框）
2. 按顺序命名为 TV1, TV2, TV3 等
3. 如果只有一个屏幕，也返回 TV1

返回格式（每行一个屏幕）：
TV1: x1,y1,x2,y2
TV2: x1,y1,x2,y2

例如：
TV1: 100,50,500,400
TV2: 600,50,1000,400

只返回屏幕名称和坐标，不要其他说明文字。"""

    def detect_screens(self, image_path: str, use_cache: bool = True) -> list[ScreenRegion]:
        """
        检测图像中所有屏幕的位置
        
        Args:
            image_path: 图像路径
            use_cache: 是否使用缓存（同一张图不重复检测）
            
        Returns:
            ScreenRegion 列表
        """
        # 检查缓存
        if use_cache and self._cached_regions and self._cache_image == image_path:
            logger.info(f"使用缓存的屏幕区域: {len(self._cached_regions)} 个")
            return self._cached_regions
        
        logger.info("正在调用 LLM 检测屏幕区域...")
        
        # 调用 LLM
        result = self.client.describe_image(image_path, self.PROMPT)
        
        if not result:
            logger.error("LLM 调用失败，无法检测屏幕")
            return []
        
        # 解析结果
        regions = self._parse_response(result)
        
        if not regions:
            logger.warning("未能解析出屏幕区域")
            return []
        
        logger.info(f"检测到 {len(regions)} 个屏幕: {[r.name for r in regions]}")
        
        # 更新缓存
        self._cached_regions = regions
        self._cache_image = image_path
        
        return regions
    
    def _parse_response(self, response: str) -> list[ScreenRegion]:
        """解析 LLM 返回的坐标"""
        regions = []
        
        # 匹配格式: TV1: 100,50,500,400
        pattern = r'(TV\d+):\s*(\d+),(\d+),(\d+),(\d+)'
        matches = re.findall(pattern, response, re.IGNORECASE)
        
        for name, x1, y1, x2, y2 in matches:
            try:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # 确保坐标正确 (x1 < x2, y1 < y2)
                if x1 > x2:
                    x1, x2 = x2, x1
                if y1 > y2:
                    y1, y2 = y2, y1
                
                regions.append(ScreenRegion(name, x1, y1, x2, y2))
            except ValueError:
                continue
        
        return regions
    
    def clear_cache(self):
        """清除缓存"""
        self._cached_regions = None
        self._cache_image = None
        logger.debug("屏幕区域缓存已清除")
