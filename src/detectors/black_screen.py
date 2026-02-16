"""
黑屏检测器 - 特征丢失/平坦度检测 (AGC 抗性增强)
不依赖绝对亮度，而是检测画面是否失去细节、对比度和结构
支持归一化区域坐标
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any
from .base import BaseDetector, DetectionResult, normalize_region


class BlackScreenDetector(BaseDetector):
    """
    [LLM_DESC]
    能力：检测屏幕是否黑屏、关闭、无信号或被遮挡
    原理：检测画面特征密度（Laplacian 梯度）和对比度。
         相比于亮度检测，该方法对摄像头的自动增益(AGC)有更强的抗性。
    场景：
      - 电视关闭（画面变平整，即使被 AGC 提亮也无细节）
      - 显示器断电
      - 监控摄像头被遮挡
      - 画面信号中断（无输入源，画面变纯色或仅有噪点）
    参数：
      - threshold: 特征丢失比例 (0.0-1.0, 默认0.8)
        说明：画面特征（Laplacian 响应）比基准下降超过此比例视为黑屏
        示例：0.8 表示丢失了 80% 的画面细节
      - region: 检测区域 [x1, y1, x2, y2] (可选)
        说明：只检测画面中的特定区域。使用归一化坐标(0.0-1.0)
        示例: [0.3, 0.2, 0.7, 0.8] 表示画面中间区域
    输出：
      - feature_drop_ratio: 特征丢失比例
      - is_flat: 画面是否变平整
      - current_variance: 当前 Laplacian 方差
    [/LLM_DESC]
    """

    DEFAULT_PARAMS = {
        "threshold": 0.8,  # 特征丢失 80% 视为黑屏
        "region": None,
    }

    def __init__(self, params=None):
        super().__init__(params)
        self._baseline_var: Optional[float] = None  # 基准 Laplacian 方差
        self._baseline_std: Optional[float] = None  # 基准标准差
        self._baseline_brightness: Optional[float] = None  # 基准亮度

    def detect(
        self, frame: np.ndarray, prev_frame: Optional[np.ndarray] = None
    ) -> DetectionResult:
        # 获取参数
        threshold = float(
            self.params.get("threshold", self.DEFAULT_PARAMS["threshold"])
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
        if frame is None or frame.size == 0:
            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description="无效帧",
                metadata={},
            )
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 降噪（避免噪点干扰 Laplacian）
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 1. 计算 Laplacian 方差（衡量画面特征/细节丰富度）
        laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
        current_var = float(laplacian.var())

        # 2. 计算标准差（衡量全局对比度）
        current_std = float(np.std(gray))

        # 3. 计算亮度（辅助参考，不作为主要判定）
        current_brightness = float(np.mean(gray))

        # 如果还没有基准，设为当前（INIT 阶段）
        if (
            self._baseline_var is None
            or self._baseline_std is None
            or self._baseline_brightness is None
        ):
            # 避免除以 0
            self._baseline_var = max(current_var, 0.1)
            self._baseline_std = max(current_std, 0.1)
            self._baseline_brightness = current_brightness

            return DetectionResult(
                is_suspicious=False,
                confidence=0.0,
                description=f"初始化基准: 特征值={self._baseline_var:.1f}, 对比度={self._baseline_std:.1f}",
                metadata={
                    "current_var": current_var,
                    "baseline_var": self._baseline_var,
                    "current_std": current_std,
                    "baseline_std": self._baseline_std,
                    "feature_drop_ratio": 0.0,
                },
                alert_reason=None,
            )

        # 计算下降比例
        var_drop_ratio = (self._baseline_var - current_var) / self._baseline_var
        std_drop_ratio = (self._baseline_std - current_std) / self._baseline_std

        # 判定：特征和对比度都显著下降
        # 即使 AGC 提亮了画面，Laplacian 方差在纯色/无信号画面上依然会极低
        is_flat = var_drop_ratio >= threshold
        is_low_contrast = std_drop_ratio >= (threshold * 0.8)  # 对比度下降稍宽容一点

        # 如果是极度黑暗（亮度 < 10），直接判定为黑屏（处理非 AGC 场景）
        is_pure_dark = current_brightness < 10

        is_suspicious = bool(is_flat or is_pure_dark)

        # 计算置信度
        if is_suspicious:
            confidence = max(var_drop_ratio, 0.5) if not is_pure_dark else 1.0
            if is_pure_dark:
                reason = f"画面极暗 (亮度 {current_brightness:.1f})"
            else:
                reason = (
                    f"画面丢失细节 (特征值下降 {var_drop_ratio:.1%}, 阈值 {threshold})"
                )
        else:
            confidence = 0.0
            reason = None

        # 构建描述
        description = (
            f"特征值: {current_var:.1f}/{self._baseline_var:.1f} (-{var_drop_ratio:.1%}), "
            f"对比度: {current_std:.1f}/{self._baseline_std:.1f}, "
            f"亮度: {current_brightness:.1f}"
        )

        return DetectionResult(
            is_suspicious=is_suspicious,
            confidence=confidence,
            description=description,
            metadata={
                "current_var": current_var,
                "baseline_var": self._baseline_var,
                "current_std": current_std,
                "baseline_std": self._baseline_std,
                "var_drop_ratio": var_drop_ratio,
                "std_drop_ratio": std_drop_ratio,
                "current_brightness": current_brightness,
            },
            alert_reason=reason,
        )

    def reset_baseline(self):
        """重置基准"""
        self._baseline_var = None
        self._baseline_std = None
        self._baseline_brightness = None
