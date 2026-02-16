"""
视觉处理核心模块 - 动态/静态模式
使用 CV 模板系统，LLM 选择模板而非生成代码
"""

import cv2
import time
import json
import base64
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any
from enum import Enum

from .config import config
from .llm_client import get_llm_client
from .logger import logger
from .detectors import get_detector, build_llm_menu, DetectionResult as DetectorResult


class VisionMode(Enum):
    """视觉处理模式"""

    DYNAMIC = "dynamic"  # 动态: 选择 CV 模板 + 循环监控
    STATIC = "static"  # 静态: 直接 LLM


class VisionState(Enum):
    """动态视觉状态机"""

    INIT = "init"  # 初始：截帧 + LLM 选择检测器
    MONITOR = "monitor"  # 监控：执行 CV 模板检测
    ALERT = "alert"  # 异常：LLM 决策下一步
    DONE = "done"  # 完成


@dataclass
class VisionResult:
    """视觉处理结果"""

    mode: VisionMode
    state: VisionState = VisionState.INIT
    frame: Optional[cv2.Mat] = None
    image_path: Optional[Path] = None
    llm_response: Optional[str] = None
    cv_result: Optional[Dict[str, Any]] = None
    is_anomaly: bool = False
    trigger_reason: Optional[str] = None
    detector_config: Optional[Dict[str, Any]] = None  # LLM 选择的检测器配置


@dataclass
class VisionContext:
    """动态视觉上下文 - 供 LLM 决策用"""

    state: VisionState = VisionState.INIT
    baseline_description: str = ""  # INIT 阶段 LLM 的静态分析
    detector_name: str = ""  # 选择的检测器名称
    detector_params: Dict[str, Any] = field(default_factory=dict)  # 检测器参数
    cv_results_history: list = field(default_factory=list)  # 历次 CV 结果

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "baseline_description": self.baseline_description,
            "detector": self.detector_name,
            "params": self.detector_params,
            "cv_results_history": self.cv_results_history[-5:],  # 最近5条
        }


class VisionProcessor:
    """视觉处理器 - 使用 CV 模板系统"""

    def __init__(
        self,
        mode: VisionMode = None,
        interval: int = None,
        llm_config: dict = None,
        stream_url: str = None,
        output_dir: str = None,
    ):
        self.mode = mode or VisionMode(config.get("mode", "static"))
        self.interval = interval or config.get("interval", 60)
        self.stream_url = stream_url or config.get("stream_url", "")
        output_dir = output_dir or config.get("output_dir", "photos")

        # LLM 客户端
        self.llm_client = get_llm_client(llm_config) if llm_config else get_llm_client()

        # 视频捕获
        self.video_capture = None

        # 输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 动态模式状态
        self.vision_state = VisionState.INIT
        self.context = VisionContext()
        self.prev_frame = None  # 用于帧差对比
        self.detector = None  # 当前使用的检测器
        self.user_prompt = "监控异常"  # 用户监控意图

        logger.info(
            f"VisionProcessor: mode={self.mode.value}, interval={self.interval}s"
        )

    def set_stream(self, url: str):
        """设置视频流"""
        self.stream_url = url

    def capture_frame(self) -> Optional[cv2.Mat]:
        """从视频流捕获一帧"""
        if self.video_capture is None or not self.video_capture.isOpened():
            if not self.stream_url:
                logger.error("未设置视频流地址")
                return None
            logger.info(f"打开视频流: {self.stream_url}")
            self.video_capture = cv2.VideoCapture(self.stream_url)

        if not self.video_capture.isOpened():
            logger.error(f"无法打开视频流: {self.stream_url}")
            return None

        ret, frame = self.video_capture.read()
        if not ret:
            logger.warning("读取帧失败")
            return None

        return frame

    def save_frame(self, frame: cv2.Mat, prefix: str = "frame") -> Path:
        """保存帧为图片"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = self.output_dir / filename

        # 降分辨率 (减少 LLM token)
        resized = cv2.resize(frame, (1280, 720))
        cv2.imwrite(str(filepath), resized, [cv2.IMWRITE_JPEG_QUALITY, 85])

        return filepath

    def frame_to_base64(self, frame: cv2.Mat) -> str:
        """帧转 Base64"""
        resized = cv2.resize(frame, (640, 360))  # 进一步压缩
        _, buffer = cv2.imencode(".jpg", resized)
        return base64.b64encode(buffer).decode("utf-8")

    # ========== 动态视觉核心流程 ==========

    def process_dynamic_init(self, frame: cv2.Mat) -> VisionResult:
        """INIT 阶段：LLM 静态分析 + 选择检测器模板"""
        result = VisionResult(mode=self.mode, state=VisionState.INIT)
        result.frame = frame
        result.image_path = self.save_frame(frame, "init")

        # 获取检测器菜单
        detector_menu = build_llm_menu()

        # 调用 LLM 分析画面并生成监控方案
        prompt = f"""你是一个视觉监控系统初始化专家。用户的监控目标是："{self.user_prompt}"

画面尺寸：640x480 像素

请分析这张初始图片，并制定监控方案：
1. 描述画面内容
2. 识别监控主体（如 iPad 屏幕、人、门口等）在画面中的位置，返回归一化边界框坐标 [x1, y1, x2, y2]
3. 从以下检测器菜单中选择最合适的一个：
{detector_menu}
4. 制定“VLLM 二次确认逻辑”：当 CV 检测器发现可疑变化时，VLLM 应该如何判断？请编写一段给 VLLM 的指令（vllm_prompt）。

坐标说明：
- 使用归一化坐标 (0.0-1.0)，相对于画面尺寸的比例
- x1, y1 是左上角，x2, y2 是右下角

返回格式（纯 JSON，不要注释）：
```json
{{
  "description": "画面内容描述",
  "monitor_region": [x1, y1, x2, y2],
  "detector": "检测器名称",
  "params": {{ "检测器参数" }},
  "vllm_prompt": "你是一位视觉分析师。用户的目标是：{user_goal}。CV检测器({detector_name})发现了可疑变化：{cv_result}。请对比基准图和当前图，判断是否发生了..."
}}
```"""

        try:
            response = self.llm_client.describe_image(str(result.image_path), prompt)

            # 解析 JSON
            import re
            json_str = None
            json_block_match = re.search(r"```json\s*\n(.*?)\n```", response, re.DOTALL)
            if json_block_match:
                json_str = json_block_match.group(1)
            else:
                json_match = re.search(r"\{[\s\S]*?\}", response)
                if json_match:
                    json_str = json_match.group()

            if json_str:
                data = json.loads(json_str)
                result.llm_response = data.get("description", "")

                detector_name = data.get("detector", "black_screen")
                detector_params = data.get("params", {})
                
                # 如果有 region 且没有在 params 中设置，则设置它
                if "region" not in detector_params and "monitor_region" in data:
                    detector_params["region"] = data["monitor_region"]

                result.detector_config = {
                    "detector": detector_name,
                    "params": detector_params,
                    "vllm_prompt": data.get("vllm_prompt")
                }

                # 保存到上下文
                self.context.baseline_description = result.llm_response
                self.context.detector_name = detector_name
                self.context.detector_params = detector_params
                # 扩展上下文以存储 vllm_prompt
                self.context.vllm_prompt = data.get("vllm_prompt")

                # 初始化检测器
                self.detector = get_detector(detector_name, detector_params)

                logger.info(f"INIT 完成: {detector_name}, params={detector_params}")
            else:
                logger.error("无法从 LLM 响应中解析 JSON")
                # 默认回退
                self.detector = get_detector("black_screen", {})

        except Exception as e:
            logger.error(f"INIT LLM 调用失败: {e}")
            # 出错时使用默认检测器
            self.context.detector_name = "black_screen"
            self.detector = get_detector("black_screen", {})

        # 切换到监控状态
        self.vision_state = VisionState.MONITOR
        result.state = VisionState.INIT

        return result

    def process_dynamic_monitor(self, frame: cv2.Mat) -> VisionResult:
        """MONITOR 阶段：执行 CV 模板检测"""
        result = VisionResult(mode=self.mode, state=VisionState.MONITOR)
        result.frame = frame
        result.detector_config = {
            "detector": self.context.detector_name,
            "params": self.context.detector_params,
        }

        if self.detector is None:
            logger.error("检测器未初始化")
            return result

        # 执行检测
        try:
            detector_result = self.detector.detect(frame, self.prev_frame)

            # 转换为字典
            cv_result = {
                "is_suspicious": detector_result.is_suspicious,
                "confidence": detector_result.confidence,
                "description": detector_result.description,
                "metadata": detector_result.metadata,
                "alert_reason": detector_result.alert_reason,
            }

            result.cv_result = cv_result
            result.is_anomaly = detector_result.is_suspicious
            result.trigger_reason = detector_result.alert_reason

            # 记录历史
            self.context.cv_results_history.append(
                {"timestamp": time.time(), "result": cv_result}
            )

            # 如果检测到异常，切换到 ALERT 状态
            if detector_result.is_suspicious:
                self.vision_state = VisionState.ALERT
                result.state = VisionState.ALERT
                logger.warning(f"MONITOR 检测到异常: {detector_result.alert_reason}")

            # 更新前一帧
            self.prev_frame = frame.copy()

        except Exception as e:
            logger.error(f"检测执行失败: {e}")
            result.cv_result = {"error": str(e)}

        return result

    def process_dynamic_alert(self, frame: cv2.Mat) -> VisionResult:
        """ALERT 阶段：LLM 决策下一步"""
        result = VisionResult(mode=self.mode, state=VisionState.ALERT)
        result.frame = frame
        result.image_path = self.save_frame(frame, "alert")

        # 获取检测器菜单（用于调整时参考）
        detector_menu = build_llm_menu()

        # 准备上下文给 LLM
        context_prompt = f"""当前监控状态：
- 基准描述: {self.context.baseline_description}
- 当前检测器: {self.context.detector_name}
- 当前参数: {json.dumps(self.context.detector_params, ensure_ascii=False)}
- 最近 CV 结果: {json.dumps(self.context.cv_results_history[-3:], ensure_ascii=False)}

{detector_menu}

检测到异常！请决策：
1. 继续监控 (continue) - 可能是临时异常，用相同参数继续
2. 调整参数 (adjust) - 选择新检测器或修改参数
3. 停止监控 (stop) - 已确定问题，停止监控

请返回 JSON:
```json
{{
  "decision": "continue/adjust/stop",
  "reason": "决策原因",
  "detector": "如果选择 adjust，填写检测器名称",
  "params": {{"如果选择 adjust，填写新参数"}}
}}
```"""

        try:
            response = self.llm_client.describe_image(
                str(result.image_path), context_prompt
            )

            import re

            json_match = re.search(r"\{{[\s\S]*\}}", response)
            if json_match:
                data = json.loads(json_match.group())
                decision = data.get("decision", "continue")

                if decision == "stop":
                    self.vision_state = VisionState.DONE
                    result.llm_response = "监控停止"

                elif decision == "adjust":
                    # 更新检测器
                    new_detector = data.get("detector", self.context.detector_name)
                    new_params = data.get("params", self.context.detector_params)

                    self.context.detector_name = new_detector
                    self.context.detector_params = new_params
                    self.detector = get_detector(new_detector, new_params)

                    self.vision_state = VisionState.MONITOR
                    result.llm_response = f"已切换到 {new_detector} 检测器，继续监控"
                    result.detector_config = {
                        "detector": new_detector,
                        "params": new_params,
                    }

                else:  # continue
                    self.vision_state = VisionState.MONITOR
                    result.llm_response = "继续监控"
            else:
                # 默认继续
                self.vision_state = VisionState.MONITOR
                result.llm_response = "继续监控"

        except Exception as e:
            logger.error(f"ALERT LLM 调用失败: {e}")
            self.vision_state = VisionState.MONITOR
            result.llm_response = "继续监控"

        return result

    # ========== 主流程 ==========

    def process_frame(self, frame: cv2.Mat) -> VisionResult:
        """处理单帧 - 根据状态机分发"""

        if self.mode == VisionMode.STATIC:
            # 静态模式：直接 LLM
            result = VisionResult(mode=self.mode, state=VisionState.DONE)
            result.image_path = self.save_frame(frame)

            try:
                result.llm_response = self.llm_client.describe_image(
                    str(result.image_path)
                )
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")

            return result

        elif self.mode == VisionMode.DYNAMIC:
            # 动态模式：状态机
            if self.vision_state == VisionState.INIT:
                return self.process_dynamic_init(frame)
            elif self.vision_state == VisionState.MONITOR:
                return self.process_dynamic_monitor(frame)
            elif self.vision_state == VisionState.ALERT:
                return self.process_dynamic_alert(frame)
            else:
                # DONE 状态
                result = VisionResult(mode=self.mode, state=VisionState.DONE)
                result.llm_response = "监控已完成"
                return result

    def process_once(self) -> Optional[VisionResult]:
        """处理一次"""
        frame = self.capture_frame()
        if frame is None:
            return None
        return self.process_frame(frame)

    def process_loop(self, callback: Callable[[VisionResult], None] = None):
        """循环处理"""
        logger.info(f"开始视觉处理循环 (mode={self.mode.value})")

        while True:
            try:
                result = self.process_once()

                if result and callback:
                    callback(result)

                # 监控完成或出错则退出
                if self.vision_state == VisionState.DONE:
                    logger.info("监控完成")
                    break

                time.sleep(self.interval)

            except KeyboardInterrupt:
                logger.info("收到中断信号，停止")
                break
            except Exception as e:
                logger.error(f"处理异常: {e}")
                time.sleep(5)

        self.release()

    def release(self):
        """释放资源"""
        if self.video_capture:
            self.video_capture.release()
        logger.info("资源已释放")
