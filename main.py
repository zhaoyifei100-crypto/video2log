"""
视频视觉分析 - 主入口
"""
import argparse
import os
import sys

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.vision import VisionProcessor, VisionMode
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(description="视频视觉分析")
    parser.add_argument('--mode', choices=['dynamic', 'static'], default=None,
                        help='处理模式: dynamic(动态-OpenCV+LLM) 或 static(静态-直接LLM)')
    parser.add_argument('--interval', type=int, default=None,
                        help='抓取间隔(秒)')
    parser.add_argument('--stream', type=str, default=None,
                        help='视频流地址')
    parser.add_argument('--input', type=str, default=None,
                        help='本地图片路径 (静态模式用)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录')
    parser.add_argument('--once', action='store_true',
                        help='只运行一次')
    
    args = parser.parse_args()
    
    # 命令行参数优先，否则用配置
    if args.mode:
        mode = VisionMode.DYNAMIC if args.mode == 'dynamic' else VisionMode.STATIC
    else:
        mode = None  # 从配置读取
    
    processor = VisionProcessor(
        mode=mode,
        interval=args.interval,
        stream_url=args.stream,
        output_dir=args.output
    )
    
    # 支持本地图片输入
    if args.input:
        import cv2
        frame = cv2.imread(args.input)
        if frame is None:
            print(f"无法读取图片: {args.input}")
            return
        # 直接处理
        if processor.mode == VisionMode.STATIC:
            # 保存图片路径
            processor.output_dir = Path(args.input).parent
            result = processor.process_frame(frame)
        else:
            # 动态模式需要视频流，暂不支持本地图片
            print("动态模式暂不支持本地图片，请用 --stream")
            return
    elif args.once:
        # 运行一次
        logger.info(f"模式: {processor.mode.value}, 间隔: {processor.interval}s")
        result = processor.process_once()
        
        if result:
            logger.info("=" * 50)
            logger.info(f"模式: {result.mode.value}")
            logger.info(f"异常: {result.is_anomaly}")
            if result.trigger_reason:
                logger.info(f"原因: {result.trigger_reason}")
            if result.llm_response:
                logger.info(f"LLM: {result.llm_response[:200]}...")
            if result.image_path:
                logger.info(f"图片: {result.image_path}")
        else:
            logger.error("处理失败")
    else:
        # 循环运行
        def callback(result):
            logger.info("=" * 50)
            logger.info(f"模式: {result.mode.value}, 异常: {result.is_anomaly}")
            if result.trigger_reason:
                logger.info(f"原因: {result.trigger_reason}")
            if result.llm_response:
                logger.info(f"LLM: {result.llm_response[:100]}...")
        
        processor.process_loop(callback)


if __name__ == "__main__":
    main()
