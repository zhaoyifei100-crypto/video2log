"""
视觉处理核心模块 - 动态/静态模式
支持 LLM 自己生成 CV 代码进行监控
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


class VisionMode(Enum):
    """视觉处理模式"""
    DYNAMIC = "dynamic"  # 动态: LLM 生成 CV 代码 + 循环监控
    STATIC = "static"   # 静态: 直接 LLM


class VisionState(Enum):
    """动态视觉状态机"""
    INIT = "init"       # 初始：截帧 + LLM 静态分析
    MONITOR = "monitor" # 监控：执行 LLM 生成的 CV 代码
    ALERT = "alert"     # 异常：LLM 决策下一步
    DONE = "done"       # 完成


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
    generated_code: Optional[str] = None  # LLM 生成的 CV 代码


@dataclass
class VisionContext:
    """动态视觉上下文 - 供 LLM 决策用"""
    state: VisionState = VisionState.INIT
    baseline_description: str = ""    # INIT 阶段 LLM 的静态分析
    generated_code: str = ""          # LLM 生成的 CV 代码
    cv_results_history: list = field(default_factory=list)  # 历次 CV 结果
    last_frame_base64: str = ""       # 最近一帧 (Base64)
    
    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "baseline_description": self.baseline_description,
            "generated_code": self.generated_code,
            "cv_results_history": self.cv_results_history[-5:],  # 最近5条
        }


class VisionProcessor:
    """视觉处理器"""
    
    # LLM 可用的 CV 函数 (供生成代码时参考)
    CV_FUNCTIONS = """
你可以通过以下 OpenCV 函数来编写监控代码：

def detect_brightness(frame, region=None) -> float:
    '''检测画面亮度，返回平均亮度值 (0-255)'''
    
def detect_dark_ratio(frame, threshold=30) -> float:
    '''检测暗像素比例，返回 0.0-1.0'''
    
def detect_motion(frame, prev_frame) -> float:
    '''检测运动变化，返回帧差分数'''
    
def detect_edges(frame, low=50, high=150) -> dict:
    '''边缘检测，返回边缘数量和位置'''
    
def detect_color_region(frame, color_range={'lower': [0,0,200], 'upper': [100,100,255]}):
    '''检测指定颜色区域，返回匹配区域列表'''
    
def crop_region(frame, x1, y1, x2, y2):
    '''裁剪画面区域'''
    
def compare_frame(frame1, frame2) -> dict:
    '''对比两帧差异，返回差异分数和位置'''

注意：
- 所有函数第一个参数都是 frame (numpy array)
- 返回值必须是 JSON 格式的字典
- 代码要简洁，适合循环执行
"""
    
    def __init__(
        self,
        mode: VisionMode = None,
        interval: int = None,
        llm_config: dict = None,
        stream_url: str = None,
        output_dir: str = None
    ):
        self.mode = mode or VisionMode(config.get('mode', 'static'))
        self.interval = interval or config.get('interval', 60)
        self.stream_url = stream_url or config.get('stream_url', '')
        output_dir = output_dir or config.get('output_dir', 'photos')
        
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
        
        logger.info(f"VisionProcessor: mode={self.mode.value}, interval={self.interval}s")
    
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
        _, buffer = cv2.imencode('.jpg', resized)
        return base64.b64encode(buffer).decode('utf-8')
    
    # ========== CV 函数库 ==========
    
    def cv_detect_brightness(self, frame, region=None) -> float:
        """检测亮度"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if region:
            x1, y1, x2, y2 = region
            gray = gray[y1:y2, x1:x2]
        return float(np.mean(gray))
    
    def cv_detect_dark_ratio(self, frame, threshold=30) -> float:
        """检测暗像素比例"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dark_pixels = np.sum(gray < threshold)
        return dark_pixels / gray.size
    
    def cv_detect_motion(self, frame, prev_frame=None) -> float:
        """检测运动"""
        if prev_frame is None:
            return 0.0
        gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray1.shape != gray2.shape:
            return 0.0
        diff = cv2.absdiff(gray1, gray2)
        return float(np.sum(diff))
    
    def cv_detect_edges(self, frame, low=50, high=150) -> dict:
        """边缘检测"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low, high)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return {
            "edge_count": len(contours),
            "total_pixels": int(np.sum(edges > 0))
        }
    
    def cv_crop_region(self, frame, x1, y1, x2, y2):
        """裁剪区域"""
        return frame[y1:y2, x1:x2]
    
    def cv_compare_frames(self, frame1, frame2) -> dict:
        """对比两帧"""
        if frame1 is None or frame2 is None:
            return {"diff_score": 0, "is_same": True}
        
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        if gray1.shape != gray2.shape:
            return {"diff_score": 0, "is_same": False}
        
        diff = cv2.absdiff(gray1, gray2)
        diff_score = float(np.mean(diff))
        
        return {
            "diff_score": diff_score,
            "is_same": diff_score < 5
        }
    
    def run_generated_code(self, frame: cv2.Mat, code: str) -> Dict[str, Any]:
        """执行 LLM 生成的 CV 代码"""
        try:
            # 简单的代码执行 - 通过 exec
            # 提供可用的 CV 函数
            local_vars = {
                'frame': frame,
                'prev_frame': self.prev_frame,
                'cv2': cv2,
                'np': np,
                'detect_brightness': self.cv_detect_brightness,
                'detect_dark_ratio': self.cv_detect_dark_ratio,
                'detect_motion': self.cv_detect_motion,
                'detect_edges': self.cv_detect_edges,
                'crop_region': self.cv_crop_region,
                'compare_frame': self.cv_compare_frames,
            }
            
            # 执行代码，最后一行应该是返回值
            exec(code, {}, local_vars)
            
            # 获取 result 变量
            result = local_vars.get('result', {})
            
            # 更新 prev_frame
            self.prev_frame = frame.copy()
            
            return result
            
        except Exception as e:
            logger.error(f"执行 CV 代码失败: {e}")
            return {"error": str(e)}
    
    # ========== 动态视觉核心流程 ==========
    
    def process_dynamic_init(self, frame: cv2.Mat) -> VisionResult:
        """INIT 阶段：LLM 静态分析 + 生成监控代码"""
        result = VisionResult(mode=self.mode, state=VisionState.INIT)
        result.frame = frame
        result.image_path = self.save_frame(frame, "init")
        result.last_frame_base64 = self.frame_to_base64(frame)
        
        # 调用 LLM 静态分析
        prompt = f"""你是一个视觉监控系统。请分析这张图片：

1. 描述画面内容（有什么？是否正常？）
2. 如果要监控异常（比如黑屏、闪烁、物体移动），应该关注什么区域/指标？
3. 生成一段 Python OpenCV 代码来执行监控。

代码要求：
- 使用上面提供的函数
- 返回 result 字典，包含监控指标和阈值
- 代码要简洁，适合每秒执行一次

返回格式：
```json
{{
  "description": "画面描述",
  "monitor_focus": "监控关注点",
  "code": "生成的 Python 代码（不要包含注释）"
}}
```"""

        try:
            response = self.llm_client.chat([
                {"role": "user", "content": prompt},
                {"role": "system", "content": self.CV_FUNCTIONS}
            ])
            
            # 解析 JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                result.llm_response = data.get('description', '')
                result.generated_code = data.get('code', '')
                
                # 保存到上下文
                self.context.baseline_description = result.llm_response
                self.context.generated_code = result.generated_code
                
                logger.info(f"INIT 完成: {result.llm_response[:100]}")
                logger.info(f"生成的代码: {result.generated_code[:200]}")
            
        except Exception as e:
            logger.error(f"INIT LLM 调用失败: {e}")
        
        # 切换到监控状态
        self.vision_state = VisionState.MONITOR
        result.state = VisionState.INIT
        
        return result
    
    def process_dynamic_monitor(self, frame: cv2.Mat) -> VisionResult:
        """MONITOR 阶段：执行 LLM 生成的 CV 代码"""
        result = VisionResult(mode=self.mode, state=VisionState.MONITOR)
        result.frame = frame
        result.generated_code = self.context.generated_code
        
        # 执行生成的 CV 代码
        cv_result = self.run_generated_code(frame, self.context.generated_code)
        result.cv_result = cv_result
        
        # 记录历史
        self.context.cv_results_history.append({
            "timestamp": time.time(),
            "result": cv_result
        })
        
        # 判断是否异常 (LLM 生成的代码自己决定)
        is_anomaly = cv_result.get('is_anomaly', False)
        if 'error' not in cv_result and 'is_same' in cv_result:
            # 简单默认：如果帧完全相同，可能是卡住了
            if cv_result.get('is_same', False):
                is_anomaly = True
        
        result.is_anomaly = is_anomaly
        
        if is_anomaly:
            result.trigger_reason = cv_result.get('reason', '检测到异常')
            self.vision_state = VisionState.ALERT
            result.state = VisionState.ALERT
            logger.warning(f"MONITOR 检测到异常: {result.trigger_reason}")
        
        return result
    
    def process_dynamic_alert(self, frame: cv2.Mat) -> VisionResult:
        """ALERT 阶段：LLM 决策下一步"""
        result = VisionResult(mode=self.mode, state=VisionState.ALERT)
        result.frame = frame
        result.image_path = self.save_frame(frame, "alert")
        
        # 准备上下文给 LLM
        context_prompt = f"""当前监控状态：
- 基准描述: {self.context.baseline_description}
- 生成的代码: {self.context.generated_code}
- 最近 CV 结果: {self.context.cv_results_history[-3:]}

检测到异常！请决策：
1. 继续监控 (可能恢复正常)
2. 调整代码参数 (重新生成代码)
3. 停止监控 (已确定问题)

请返回 JSON:
```json
{{
  "decision": "continue/adjust/stop",
  "reason": "原因",
  "new_code": "如果选择 adjust，生成新代码"
}}
```"""

        try:
            response = self.llm_client.chat([
                {"role": "user", "content": context_prompt}
            ])
            
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
                decision = data.get('decision', 'continue')
                
                if decision == 'stop':
                    self.vision_state = VisionState.DONE
                    result.llm_response = "监控停止"
                elif decision == 'adjust':
                    self.context.generated_code = data.get('new_code', self.context.generated_code)
                    self.vision_state = VisionState.MONITOR
                    result.llm_response = "已调整代码，继续监控"
                else:
                    self.vision_state = VisionState.MONITOR
                    result.llm_response = "继续监控"
                
                result.generated_code = self.context.generated_code
                
        except Exception as e:
            logger.error(f"ALERT LLM 调用失败: {e}")
            self.vision_state = VisionState.MONITOR
        
        return result
    
    # ========== 主流程 ==========
    
    def process_frame(self, frame: cv2.Mat) -> VisionResult:
        """处理单帧 - 根据状态机分发"""
        
        if self.mode == VisionMode.STATIC:
            # 静态模式：直接 LLM
            result = VisionResult(mode=self.mode, state=VisionState.DONE)
            result.image_path = self.save_frame(frame)
            result.last_frame_base64 = self.frame_to_base64(frame)
            
            try:
                result.llm_response = self.llm_client.describe_image(str(result.image_path))
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
    
    def process_once(self) -> VisionResult:
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
