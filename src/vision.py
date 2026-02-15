"""
视觉处理核心模块 - 动态/静态模式
"""
import cv2
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum

from .config import config
from .opencv_helper import OpenCVHelper, FrameAnalysis
from .llm_client import get_llm_client
from .logger import logger


class VisionMode(Enum):
    """视觉处理模式"""
    DYNAMIC = "dynamic"  # 动态: OpenCV 预处理 + LLM (异常时)
    STATIC = "static"   # 静态: 直接 LLM


@dataclass
class VisionResult:
    """视觉处理结果"""
    mode: VisionMode
    frame: Optional[cv2.Mat] = None
    image_path: Optional[Path] = None
    llm_response: Optional[str] = None
    frame_analysis: Optional[FrameAnalysis] = None
    is_anomaly: bool = False
    trigger_reason: Optional[str] = None


class VisionProcessor:
    """视觉处理器"""
    
    def __init__(
        self,
        mode: VisionMode = None,
        interval: int = None,
        opencv_config: dict = None,
        llm_config: dict = None,
        stream_url: str = None,
        output_dir: str = None
    ):
        """
        Args:
            mode: 处理模式 (默认从配置读取)
            interval: 抓取间隔 (默认从配置读取)
            opencv_config: OpenCV 配置
            llm_config: LLM 配置
            stream_url: 视频流地址
            output_dir: 输出目录
        """
        # 从配置读取默认值
        self.mode = mode or VisionMode(config.get('mode', 'static'))
        self.interval = interval or config.get('interval', 60)
        self.stream_url = stream_url or config.get('stream_url', '')
        output_dir = output_dir or config.get('output_dir', 'photos')
        
        # OpenCV 预处理
        opencv_cfg = opencv_config or config.get('opencv', {})
        self.opencv = OpenCVHelper(
            brightness_threshold=opencv_cfg.get('brightness_threshold', 30),
            dark_ratio_threshold=opencv_cfg.get('dark_ratio_threshold', 0.9),
            motion_threshold=opencv_cfg.get('motion_threshold', 1000)
        ) if self.mode == VisionMode.DYNAMIC else None
        
        # LLM 客户端
        self.llm_client = get_llm_client(llm_config) if llm_config else get_llm_client()
        
        # 视频捕获
        self.video_capture = None
        
        # 输出目录
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def process_frame(self, frame: cv2.Mat) -> VisionResult:
        """处理单帧"""
        result = VisionResult(mode=self.mode)
        
        # 1. OpenCV 预处理 (动态模式)
        if self.mode == VisionMode.DYNAMIC:
            result.frame_analysis = self.opencv.analyze_frame(frame)
            result.is_anomaly = result.frame_analysis.is_anomaly
            
            if result.is_anomaly:
                reason = []
                if result.frame_analysis.avg_brightness < self.opencv.brightness_threshold:
                    reason.append(f"亮度低({result.frame_analysis.avg_brightness:.1f})")
                if result.frame_analysis.dark_ratio > self.opencv.dark_ratio_threshold:
                    reason.append(f"暗像素多({result.frame_analysis.dark_ratio:.1%})")
                result.trigger_reason = "; ".join(reason)
                logger.warning(f"检测到异常: {result.trigger_reason}")
        
        # 2. LLM 分析 (静态模式 或 动态模式异常时)
        need_llm = (self.mode == VisionMode.STATIC) or result.is_anomaly
        
        if need_llm and self.llm_client:
            # 保存图片
            result.image_path = self.save_frame(frame)
            
            # 调用 LLM
            try:
                result.llm_response = self.llm_client.describe_image(str(result.image_path))
                logger.info(f"LLM 响应: {result.llm_response[:100] if result.llm_response else 'None'}...")
            except Exception as e:
                logger.error(f"LLM 调用失败: {e}")
        
        result.frame = frame
        return result
    
    def process_once(self) -> VisionResult:
        """处理一次"""
        frame = self.capture_frame()
        if frame is None:
            return None
        return self.process_frame(frame)
    
    def process_loop(self, callback: Callable[[VisionResult], None] = None):
        """循环处理
        
        Args:
            callback: 每帧处理后的回调函数
        """
        logger.info("开始视觉处理循环")
        
        while True:
            try:
                result = self.process_once()
                
                if result and callback:
                    callback(result)
                
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
