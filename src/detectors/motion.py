"""
运动检测器
检测画面运动、物体移动
支持归一化区域坐标
"""

import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from .base import BaseDetector, DetectionResult, normalize_region
from ..logger import logger


class MotionDetector(BaseDetector):
    """
    [LLM_DESC]
    能力：检测画面运动、物体移动、有人出现
    场景：
      - 有人进入监控区域
      - 物体被移动
      - 画面内容发生变化
      - 摄像头被触碰导致抖动
    参数：
      - sensitivity: 灵敏度 (1000-50000, 默认5000)
        说明：帧差值超过此值视为运动
      - min_area: 最小运动区域面积 (默认500)
        说明：过滤小的噪声变化
      - region: 检测区域 [x1, y1, x2, y2] (可选)
        说明：只检测画面中的特定区域。使用归一化坐标(0.0-1.0)
        示例: [0.3, 0.2, 0.7, 0.8] 表示画面中间区域
    输出：
      - motion_score: 运动得分（帧差值）
      - motion_areas: 运动区域数量和位置
    [/LLM_DESC]
    """

    DEFAULT_PARAMS = {"sensitivity": 20000, "min_area": 2000, "region": None}

    def __init__(self, params=None):
        super().__init__(params)
        self.prev_gray = None

    def detect(
        self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None
    ) -> DetectionResult:
        # 获取参数
        sensitivity = self.params.get("sensitivity", self.DEFAULT_PARAMS["sensitivity"])
        min_area = self.params.get("min_area", self.DEFAULT_PARAMS["min_area"])
        region = self.params.get("region", self.DEFAULT_PARAMS["region"])

        # 裁剪区域（如果指定）- 支持归一化坐标
        if region and len(region) == 4:
            h, w = frame.shape[:2]
            coords = normalize_region(region, h, w)
            if coords:
                x1, y1, x2, y2 = coords
                frame = frame[y1:y2, x1:x2]

        # 转灰度
        if frame is None or frame.size == 0:
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="无效帧",
                metadata={},
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 高斯模糊减少噪声
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # 第一帧初始化
        if self.prev_gray is None:
            self.prev_gray = gray
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="初始化完成，等待下一帧",
                metadata={"motion_score": 0, "areas": []},
            )

        # 帧差
        if self.prev_gray.shape != gray.shape:
            logger.warning(
                f"MotionDetector: 帧大小不一致 ({self.prev_gray.shape} vs {gray.shape})，重置基准"
            )
            self.prev_gray = gray
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="帧大小变化，已重置基准",
                metadata={"motion_score": 0, "areas": []},
            )

        frame_delta = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

        # 膨胀处理
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # 找轮廓
        contours, _ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 过滤小区域
        motion_areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                motion_areas.append({"x": x, "y": y, "w": w, "h": h, "area": area})

        # 计算运动得分
        motion_score = int(np.sum(frame_delta))

        # 更新前一帧
        self.prev_gray = gray.copy()

        # 判断是否可疑
        is_suspicious = motion_score > sensitivity or len(motion_areas) > 0

        # 计算置信度
        if is_suspicious:
            confidence = min(motion_score / (sensitivity * 2), 1.0)
            if len(motion_areas) > 0:
                reason = (
                    f"检测到 {len(motion_areas)} 个运动区域，运动得分 {motion_score}"
                )
            else:
                reason = f"运动得分 {motion_score} > {sensitivity}"
        else:
            confidence = 0.0
            reason = None

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            description=f"运动得分: {motion_score}, 检测区域数: {len(motion_areas)}",
            metadata={
                "motion_score": motion_score,
                "sensitivity": sensitivity,
                "areas": motion_areas,
                "area_count": len(motion_areas),
            },
            alert_reason=reason,
        )
