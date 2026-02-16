"""
运动检测器 - 简单全局变化检测
计算画面中发生变化的像素占比，适合作为通用的运动/变化模版
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseDetector, DetectionResult, normalize_region
from ..logger import logger


class MotionDetector(BaseDetector):
    """
    [LLM_DESC]
    能力：检测画面中的运动、物体移动或任何显著变化
    原理：计算前后两帧之间发生变化的像素占总像素的比例 (change_ratio)
    场景：
      - 检测是否有人进入房间
      - 检测屏幕内容是否发生了跳转或切换
      - 检测摄像头是否被移动
    参数：
      - threshold: 变化比例阈值 (0.0-1.0, 默认0.05)
        说明：画面中超过此比例的像素发生变化时触发警报
        示例：0.05 表示画面 5% 发生变化即报警
      - region: 检测区域 [x1, y1, x2, y2] (可选)
        说明：只检测特定区域的变化。使用归一化坐标(0.0-1.0)
    输出：
      - change_ratio: 变化像素占比
    [/LLM_DESC]
    """

    DEFAULT_PARAMS = {"threshold": 0.05, "region": None}

    def __init__(self, params=None):
        super().__init__(params)
        self.prev_gray = None

    def detect(
        self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None
    ) -> DetectionResult:
        # 获取参数
        threshold = float(
            self.params.get("threshold", self.DEFAULT_PARAMS["threshold"])
        )
        region = self.params.get("region", self.DEFAULT_PARAMS["region"])

        # 裁剪区域
        if region and len(region) == 4:
            h, w = frame.shape[:2]
            coords = normalize_region(region, h, w)
            if coords:
                x1, y1, x2, y2 = coords
                frame = frame[y1:y2, x1:x2]

        if frame is None or frame.size == 0:
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="无效帧",
                metadata={},
            )

        # 转灰度并轻微模糊以去噪
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # 初始化基准
        if self.prev_gray is None:
            self.prev_gray = gray
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="初始化第一帧",
                metadata={"change_ratio": 0.0},
            )

        # 尺寸检查
        if self.prev_gray.shape != gray.shape:
            logger.warning(
                f"MotionDetector: 帧大小不一致 ({self.prev_gray.shape} vs {gray.shape})，重置基准"
            )
            self.prev_gray = gray
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="帧尺寸变更，重置基准",
                metadata={"change_ratio": 0.0},
            )

        # 计算帧差
        frame_delta = cv2.absdiff(self.prev_gray, gray)
        # 二值化：差异超过 25 的像素视为变化
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # 计算变化像素占比
        change_count = np.count_nonzero(thresh)
        change_ratio = float(change_count) / gray.size

        # 更新前一帧
        self.prev_gray = gray.copy()

        # 判断是否超过阈值
        is_suspicious = change_ratio >= threshold

        if is_suspicious:
            confidence = min(change_ratio / (threshold * 2), 1.0)
            reason = (
                f"检测到画面变化: 变化比例 {change_ratio:.1%} (阈值 {threshold:.1%})"
            )
        else:
            confidence = 0.0
            reason = None

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            description=f"画面变化比例: {change_ratio:.1%}",
            metadata={
                "change_ratio": change_ratio,
                "threshold": threshold,
                "change_pixel_count": int(change_count),
            },
            alert_reason=reason,
        )
