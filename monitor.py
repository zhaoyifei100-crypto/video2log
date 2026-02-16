#!/usr/bin/env python3
"""
Video2Log Monitor - 独立后台监控脚本
特点：
- 独立进程，不阻塞主 Agent
- CV 检测 + VLLM 二次确认
- 确认异常后写入 VISION_ALERT.md 并退出
"""

import sys
import os
import json
import time
import base64
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import cv2
import numpy as np
import requests

# 确保可以导入 src 模块
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from src.opencv_helper import OpenCVHelper
from src.logger import logger


class VLLMClient:
    """独立的 VLLM 客户端 - 用于异常确认"""

    def __init__(self, config: Dict[str, Any]):
        self.provider = config.get("provider", "siliconflow")
        self.api_key = config.get("api_key")
        self.model = config.get("model", "Qwen/Qwen2.5-VL-72B-Instruct")
        self.base_url = config.get("base_url", "https://api.siliconflow.cn/v1")

        if not self.api_key:
            raise ValueError("VLLM API key 未配置")

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def confirm_anomaly(
        self, image_path: str, cv_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        调用 VLLM 确认是否为真正的异常

        Args:
            image_path: 当前帧路径
            cv_result: CV 算法检测结果

        Returns:
            {
                "is_confirmed_anomaly": bool,
                "confidence": float,
                "reason": str
            }
        """
        base64_image = self._encode_image(image_path)

        # 构建确认 Prompt
        prompt = f"""你是一位专业的视觉分析师。请分析这张图片，判断以下 CV 算法检测到的异常是否真实存在。

CV 算法检测结果：
```json
{json.dumps(cv_result, indent=2, ensure_ascii=False)}
```

请仔细分析图片内容，回答：
1. 这是否是真正的异常？（考虑可能的误报情况）
2. 如果是误报，说明原因
3. 如果是真实异常，描述具体异常内容

返回格式（JSON）：
```json
{{
  "is_confirmed_anomaly": true/false,
  "confidence": 0.0-1.0,
  "reason": "详细说明"
}}
```"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            "max_tokens": 500,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]

            # 解析 JSON
            import re

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "is_confirmed_anomaly": data.get("is_confirmed_anomaly", False),
                    "confidence": data.get("confidence", 0.5),
                    "reason": data.get("reason", "未提供原因"),
                }
            else:
                # 如果没有 JSON，基于内容判断
                is_anomaly = (
                    "true" in content.lower() and "false" not in content.lower()
                )
                return {
                    "is_confirmed_anomaly": is_anomaly,
                    "confidence": 0.5,
                    "reason": content[:200],
                }

        except Exception as e:
            logger.error(f"VLLM 确认失败: {e}")
            # 默认保守处理：认为是可疑异常
            return {
                "is_confirmed_anomaly": True,
                "confidence": 0.3,
                "reason": f"VLLM 调用失败: {e}",
            }


class MonitorLogger:
    """独立监控日志记录器"""

    def __init__(self, log_file: Path):
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_detection(self, result: Dict[str, Any]):
        """记录检测结果"""
        entry = {"timestamp": datetime.now().isoformat(), **result}
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class VideoMonitor:
    """视频监控器 - 后台独立运行"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor_type = config.get("monitor_type", "black_screen")
        self.interval = config.get("interval", 5)
        self.stream_url = config.get("stream_url", "desktop")
        self.threshold = config.get("threshold", 30)
        self.output_dir = Path(config.get("output_dir", "monitor_output"))
        self.alert_file = Path(config.get("alert_file", "VISION_ALERT.md"))

        # 初始化目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # CV 检测器
        self.cv_helper = OpenCVHelper(
            brightness_threshold=self.threshold, dark_ratio_threshold=0.9
        )

        # VLLM 客户端
        vllm_config = config.get("vllm", {})
        self.vllm = VLLMClient(vllm_config)

        # 日志记录器
        self.logger = MonitorLogger(self.output_dir / "monitor_logs.jsonl")

        # 视频捕获
        self.video_capture = None

        # 连续可疑计数
        self.suspicious_count = 0
        self.max_suspicious = 2  # 连续可疑次数阈值

        print(f"🎥 监控器初始化完成")
        print(f"   类型: {self.monitor_type}")
        print(f"   间隔: {self.interval}秒")
        print(f"   输出: {self.output_dir}")

    def capture_frame(self) -> Optional[np.ndarray]:
        """捕获一帧"""
        if self.stream_url == "desktop":
            # 桌面截图
            try:
                import pyautogui

                screenshot = pyautogui.screenshot()
                frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                return frame
            except Exception as e:
                print(f"截图失败: {e}", file=sys.stderr)
                return None
        else:
            # 视频流
            if self.video_capture is None or not self.video_capture.isOpened():
                self.video_capture = cv2.VideoCapture(self.stream_url)

            if not self.video_capture.isOpened():
                print(f"无法打开视频流: {self.stream_url}", file=sys.stderr)
                return None

            ret, frame = self.video_capture.read()
            return frame if ret else None

    def save_frame(self, frame: np.ndarray, prefix: str) -> Path:
        """保存帧"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return filepath

    def detect_suspicious(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        CV 算法检测可疑情况
        返回可疑检测结果，不直接判定异常
        """
        analysis = self.cv_helper.analyze_frame(frame)

        result = {
            "avg_brightness": analysis.avg_brightness,
            "dark_ratio": analysis.dark_ratio,
            "motion_score": analysis.motion_score,
            "is_suspicious": False,
            "suspicious_type": None,
            "confidence": 0.0,
        }

        # 根据监控类型判断可疑情况
        if self.monitor_type == "black_screen":
            # 黑屏检测
            if analysis.avg_brightness < self.threshold:
                result["is_suspicious"] = True
                result["suspicious_type"] = "low_brightness"
                result["confidence"] = 1.0 - (analysis.avg_brightness / self.threshold)
            elif analysis.dark_ratio > 0.9:
                result["is_suspicious"] = True
                result["suspicious_type"] = "high_dark_ratio"
                result["confidence"] = analysis.dark_ratio

        elif self.monitor_type == "motion":
            # 运动检测
            if analysis.motion_score > 5000:  # 运动阈值
                result["is_suspicious"] = True
                result["suspicious_type"] = "motion_detected"
                result["confidence"] = min(analysis.motion_score / 10000, 1.0)

        return result

    def write_alert(self, cv_result: Dict, vllm_result: Dict, image_path: Path):
        """写入异常警报到文件"""
        alert_content = f"""# 🚨 视频监控异常报告

**检测时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**监控类型**: {self.monitor_type}

## CV 检测结果

```json
{json.dumps(cv_result, indent=2, ensure_ascii=False)}
```

## VLLM 确认结果

- **确认异常**: {"✅ 是" if vllm_result.get("is_confirmed_anomaly") else "❌ 否"}
- **置信度**: {vllm_result.get("confidence", 0):.2f}
- **原因**: {vllm_result.get("reason", "N/A")}

## 截图

**文件**: `{image_path.name}`

---

**下一步操作**:
- **continue**: 继续监控（清空此文件后重新启动）
- **adjust**: 调整监控参数
- **stop**: 停止监控任务
"""
        self.alert_file.write_text(alert_content, encoding="utf-8")
        print(f"   警报已写入: {self.alert_file}")

    def run(self):
        """主监控循环"""
        print(f"\n🔴 开始监控...")
        print(f"   按 Ctrl+C 停止\n")

        check_count = 0

        try:
            while True:
                check_count += 1

                # 1. 截图
                frame = self.capture_frame()
                if frame is None:
                    print(f"[{check_count}] 截图失败，跳过")
                    time.sleep(self.interval)
                    continue

                # 2. CV 检测可疑情况
                cv_result = self.detect_suspicious(frame)

                if cv_result["is_suspicious"]:
                    print(
                        f"[{check_count}] ⚠️ 检测到可疑情况: {cv_result['suspicious_type']}"
                    )
                    self.suspicious_count += 1

                    # 3. 连续可疑达到阈值，调用 VLLM 确认
                    if self.suspicious_count >= self.max_suspicious:
                        print(
                            f"   连续 {self.suspicious_count} 次可疑，调用 VLLM 确认..."
                        )

                        # 保存当前帧
                        image_path = self.save_frame(frame, "suspicious")

                        # 调用 VLLM 确认
                        vllm_result = self.vllm.confirm_anomaly(
                            str(image_path), cv_result
                        )

                        print(
                            f"   VLLM 确认结果: {'异常' if vllm_result['is_confirmed_anomaly'] else '正常'}"
                        )
                        print(f"   置信度: {vllm_result['confidence']:.2f}")
                        print(f"   原因: {vllm_result['reason'][:100]}...")

                        # 记录到日志
                        self.logger.log_detection(
                            {
                                "check_count": check_count,
                                "cv_result": cv_result,
                                "vllm_result": vllm_result,
                                "image_path": str(image_path),
                                "action": "confirmed"
                                if vllm_result["is_confirmed_anomaly"]
                                else "filtered",
                            }
                        )

                        # 4. VLLM 确认是异常，写入警报并退出
                        if vllm_result["is_confirmed_anomaly"]:
                            # 保存为异常截图
                            alert_image = self.save_frame(frame, "alert")
                            self.write_alert(cv_result, vllm_result, alert_image)
                            print(f"\n🚨 已确认异常！监控退出，等待 Agent 处理...")
                            sys.exit(0)
                        else:
                            # 误报，重置计数
                            print(f"   误报过滤，继续监控")
                            self.suspicious_count = 0
                else:
                    # 正常情况
                    if self.suspicious_count > 0:
                        print(f"[{check_count}] ✅ 恢复正常")
                        self.suspicious_count = 0
                    elif check_count % 10 == 0:
                        print(
                            f"[{check_count}] 监控正常 (亮度: {cv_result['avg_brightness']:.1f})"
                        )

                # 5. 正常等待
                time.sleep(self.interval)

        except KeyboardInterrupt:
            print(f"\n⏹️ 监控已停止，共检测 {check_count} 次")
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="Video2Log Monitor - 独立监控脚本")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径 (JSON)")
    parser.add_argument(
        "--dry-run", action="store_true", help="只运行一次检测并退出（用于预检查）"
    )

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"配置文件不存在: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 启动监控
    monitor = VideoMonitor(config)

    if args.dry_run:
        # 预检查模式：只运行一次
        print("🧪 Dry Run 模式：执行一次检测并退出\n")

        # 1. 截图
        frame = monitor.capture_frame()
        if frame is None:
            print("❌ 预检查失败：无法获取画面")
            sys.exit(1)

        print(f"✅ 截图成功，帧大小: {frame.shape}")

        # 2. CV 检测
        cv_result = monitor.detect_suspicious(frame)
        print(f"✅ CV 检测成功: {cv_result}")

        # 3. 尝试 VLLM 连接（可选，跳过以加快速度）
        # vllm_result = monitor.vllm.confirm_anomaly(...)

        print("\n✅ 预检查通过，monitor 可正常启动")
        sys.exit(0)

    monitor.run()


if __name__ == "__main__":
    main()
