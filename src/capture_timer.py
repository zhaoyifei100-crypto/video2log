"""定时拍照循环模块"""
import time
import subprocess
from pathlib import Path
from datetime import datetime

from .config import config
from .logger import logger
from .llm_client import get_llm_client


class CaptureTimer:
    """定时拍照循环"""
    
    def __init__(self, interval: int = None, output_dir: str = None):
        self.interval = interval or config.interval
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_client = get_llm_client()
        self.running = False
        
        logger.info(f"定时拍照初始化: 间隔={self.interval}秒, 输出目录={self.output_dir}")
    
    def capture_photo(self, device: str = None) -> Path:
        """使用 raspistill 或 fswebcam 拍照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.{config.image_config.get('format', 'jpg')}"
        filepath = self.output_dir / filename
        
        # 尝试使用 raspistill (树莓派官方摄像头)
        cmd = [
            'raspistill',
            '-o', str(filepath),
            '-w', str(config.image_config.get('width', 1280)),
            '-h', str(config.image_config.get('height', 720)),
            '-q', str(config.image_config.get('quality', 85))
        ]
        
        # 如果没有指定设备，尝试自动选择
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
    
    def capture_and_describe(self):
        """拍照并描述"""
        logger.info("=" * 50)
        logger.info("开始拍照循环")
        
        while self.running:
            try:
                # 拍照
                image_path = self.capture_photo()
                
                if image_path:
                    # LLM 描述
                    description = self.describe_photo(image_path)
                    
                    if description:
                        logger.info(f"描述: {description[:100]}...")
                        
                        # Telegram 发送
                        self.send_to_telegram(image_path, description)
                
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
