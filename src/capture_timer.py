"""定时拍照循环模块"""
import time
import subprocess
import cv2
from pathlib import Path
from datetime import datetime
from typing import Optional

from .config import config
from .logger import logger
from .llm_client import get_llm_client
from .detector import BlackScreenDetector


class CaptureTimer:
    """定时拍照循环"""
    
    def __init__(self, interval: int = None, output_dir: str = None):
        self.interval = interval or config.interval
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.source_type = config.get('source.type', 'local')
        self.stream_url = config.get('source.stream_url', '')
        
        self.llm_client = get_llm_client()
        self.running = False
        self.video_capture = None
        
        # 黑屏检测初始化
        detection_config = config.get('detection', {})
        self.detection_enabled = detection_config.get('enabled', False)
        
        if self.detection_enabled:
            black_screen_config = detection_config.get('black_screen', {})
            self.black_screen_enabled = black_screen_config.get('enabled', True)
            
            if self.black_screen_enabled:
                self.detector = BlackScreenDetector(
                    threshold=black_screen_config.get('threshold', 30),
                    dark_pixel_ratio=black_screen_config.get('dark_pixel_ratio', 0.9),
                    auto_detect_screens=black_screen_config.get('auto_detect_screens', True),
                    manual_regions=black_screen_config.get('manual_regions', [])
                )
            else:
                self.detector = None
        else:
            self.black_screen_enabled = False
            self.detector = None
        
        logger.info(f"定时拍照初始化: 间隔={self.interval}秒, 输出目录={self.output_dir}")
        logger.info(f"输入源: {self.source_type}, 流地址: {self.stream_url}")
        logger.info(f"黑屏检测: {'启用' if self.black_screen_enabled else '关闭'}")
    
    def _capture_from_stream(self) -> Path:
        """从网络流捕获帧"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.{config.image_config.get('format', 'jpg')}"
        filepath = self.output_dir / filename
        
        if self.video_capture is None or not self.video_capture.isOpened():
            logger.info(f"打开网络流: {self.stream_url}")
            self.video_capture = cv2.VideoCapture(self.stream_url)
        
        if not self.video_capture.isOpened():
            logger.error(f"无法打开网络流: {self.stream_url}")
            return None
        
        ret, frame = self.video_capture.read()
        if not ret:
            logger.error("从网络流读取帧失败")
            return None
        
        cv2.imwrite(str(filepath), frame)
        logger.info(f"从网络流捕获成功: {filepath}")
        return filepath
    
    def capture_photo(self, device: str = None) -> Path:
        """拍照 - 支持本地摄像头或网络流"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.{config.image_config.get('format', 'jpg')}"
        filepath = self.output_dir / filename
        
        # 网络流模式
        if self.source_type == 'stream' and self.stream_url:
            return self._capture_from_stream()
        
        # 本地摄像头模式 (raspistill / fswebcam)
        cmd = [
            'raspistill',
            '-o', str(filepath),
            '-w', str(config.image_config.get('width', 1280)),
            '-h', str(config.image_config.get('height', 720)),
            '-q', str(config.image_config.get('quality', 85))
        ]
        
        if device:
            cmd = ['fswebcam', '-r', f"{config.image_config.get('width', 1280)}x{config.image_config.get('height', 720)}", str(filepath)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=10)
            logger.info(f"拍照成功: {filepath}")
            return filepath
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.error(f"拍照失败: {e}")
            return None
    
    def describe_photo(self, image_path: Path) -> str:
        """调用 LLM 描述图像"""
        try:
            description = self.llm_client.describe_image(str(image_path))
            if description:
                # 保存描述到文本文件
                txt_path = image_path.with_suffix('.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(description)
                logger.info(f"图像描述已保存: {txt_path}")
            return description
        except Exception as e:
            logger.error(f"图像描述失败: {e}")
            return None
    
    def send_to_telegram(self, image_path: Path, description: str = None):
        """发送照片到 Telegram"""
        telegram_config = config.telegram_config
        
        if not telegram_config.get('enabled'):
            return
        
        try:
            import requests
            
            bot_token = telegram_config.get('bot_token')
            chat_id = telegram_config.get('chat_id')
            
            # 发送图片
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {'chat_id': chat_id}
                if description:
                    data['caption'] = description[:1000]
                
                response = requests.post(
                    f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                    data=data,
                    files=files,
                    timeout=30
                )
                response.raise_for_status()
                logger.info(f"Telegram 发送成功: {image_path.name}")
                
        except Exception as e:
            logger.error(f"Telegram 发送失败: {e}")
    
    def detect_and_judge(self, image_path: Path) -> str:
        """黑屏检测并判定
        
        Returns:
            'PASS' - 所有屏幕正常
            'FAIL' - 任意屏幕黑屏/闪断
        """
        if not self.black_screen_enabled or not self.detector:
            return 'PASS'
        
        try:
            result = self.detector.detect_from_file(str(image_path))
            
            if not result.all_pass:
                # 有屏幕黑屏
                fail_screens = [r.name for r in result.results if r.is_black]
                logger.warning(f"⚠️ Link Test FAIL: {', '.join(fail_screens)} 黑屏")
                for r in result.results:
                    status = "❌ 黑屏" if r.is_black else "✅ 正常"
                    logger.warning(f"  {r.name}: {status} (亮度={r.avg_brightness:.1f})")
                return 'FAIL'
            else:
                # 全部正常
                summary = result.summary()
                logger.info(f"✓ Link Test PASS: {summary}")
                return 'PASS'
                
        except Exception as e:
            logger.error(f"黑屏检测失败: {e}")
            return 'PASS'  # 检测失败默认为 PASS
    
    def capture_and_describe(self):
        """拍照并描述"""
        logger.info("=" * 50)
        logger.info("开始拍照循环")
        
        while self.running:
            try:
                # 拍照
                image_path = self.capture_photo()
                
                if image_path:
                    # 黑屏检测
                    judge_result = self.detect_and_judge(image_path)
                    
                    # LLM 描述 (仅正常时)
                    description = None
                    if judge_result == 'PASS':
                        description = self.describe_photo(image_path)
                        if description:
                            logger.info(f"描述: {description[:100]}...")
                            # Telegram 发送
                            self.send_to_telegram(image_path, description)
                    else:
                        logger.warning(f"❌ Link Test FAIL - 黑屏检测未通过")
                        # 可选：FAIL 时也发送通知
                        detection_config = config.get('detection', {})
                        if detection_config.get('notify_on_anomaly'):
                            self.send_to_telegram(image_path, "⚠️ Link Test FAIL - 黑屏")
                
                # 等待下一个周期
                if self.running:
                    logger.info(f"等待 {self.interval} 秒...")
                    time.sleep(self.interval)
                    
            except KeyboardInterrupt:
                logger.info("收到中断信号，停止拍照")
                break
            except Exception as e:
                logger.error(f"拍照循环异常: {e}")
                time.sleep(5)  # 出错后等待5秒再重试
        
        # 清理资源
        if self.video_capture:
            self.video_capture.release()
        
        logger.info("拍照循环已停止")
    
    def start(self):
        """启动拍照循环"""
        self.running = True
        logger.info("定时拍照已启动")
        self.capture_and_describe()
    
    def stop(self):
        """停止拍照循环"""
        self.running = False
        logger.info("正在停止定时拍照...")


def main():
    """主函数"""
    import signal
    import sys
    
    timer = CaptureTimer()
    
    def signal_handler(sig, frame):
        logger.info("收到退出信号")
        timer.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    timer.start()


if __name__ == "__main__":
    main()
