#!/usr/bin/env python3
"""
运行真实Skill流程测试 - 保存结果到固定目录
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.person_detection.test_real_skill_flow import RealSkillMonitor


def run_test_with_preserved_results():
    """运行测试并保存所有结果"""

    # 固定输出目录
    output_dir = PROJECT_ROOT / "tests" / "fixtures" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建时间戳子目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_run_dir = output_dir / f"test_run_{timestamp}"
    test_run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("运行真实Skill流程测试 - 保存结果模式")
    print("=" * 70)
    print(f"视频: tests/04_020.avi")
    print(f"用户指令: 帮我盯着什么时候来人了")
    print(f"输出目录: {test_run_dir}")
    print("=" * 70)

    # 创建监控器并运行
    monitor = RealSkillMonitor(
        video_path=str(PROJECT_ROOT / "tests" / "04_020.avi"),
        user_prompt="帮我盯着什么时候来人了",
    )

    alert_file = monitor.run(test_run_dir)

    # 保存详细的测试结果日志
    log_file = test_run_dir / "test_log.json"
    test_log = {
        "timestamp": timestamp,
        "video_path": str(PROJECT_ROOT / "tests" / "04_020.avi"),
        "user_prompt": "帮我盯着什么时候来人了",
        "output_directory": str(test_run_dir),
        "test_results": {
            "processed_frames": monitor.processed_frames,
            "has_detection": monitor.alert_data is not None,
        },
    }

    if monitor.alert_data:
        test_log["test_results"]["detection"] = {
            "frame": monitor.alert_data["frame"],
            "video_position": monitor.alert_data["video_position"],
            "vllm_confidence": monitor.alert_data["vllm_result"].get("confidence", 0),
            "vllm_description": monitor.alert_data["vllm_result"].get(
                "description", ""
            ),
            "cv_motion_score": monitor.alert_data["cv_result"]["motion_score"],
            "cv_contour_count": monitor.alert_data["cv_result"]["contour_count"],
        }
        test_log["test_results"]["init_config"] = {
            "target": monitor.config.target_description if monitor.config else "",
            "strategy": monitor.config.detection_strategy if monitor.config else "",
            "threshold": monitor.config.confidence_threshold if monitor.config else 0.8,
        }

    log_file.write_text(
        json.dumps(test_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 打印摘要
    print("\n" + "=" * 70)
    print("测试完成!")
    print("=" * 70)
    print(f"\n生成的文件:")
    for f in sorted(test_run_dir.iterdir()):
        size = f.stat().st_size
        print(f"  ✓ {f.name}: {size:,} bytes")

    print(f"\n关键结果:")
    if monitor.alert_data:
        print(
            f"  🎯 检测帧: 第{monitor.alert_data['frame']}帧 ({monitor.alert_data['video_position']})"
        )
        print(
            f"  👥 检测结果: {monitor.alert_data['vllm_result'].get('description', 'N/A')}"
        )
        print(
            f"  📊 VLLM置信度: {monitor.alert_data['vllm_result'].get('confidence', 0):.1%}"
        )
    else:
        print(f"  ⚠️ 未检测到目标")

    print(f"\n📁 完整结果保存在: {test_run_dir}")
    print(f"📄 VISION_ALERT.md: {alert_file}")
    print(f"📝 测试日志: {log_file}")

    # 同时创建一个latest符号链接
    latest_link = output_dir / "latest"
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(test_run_dir, target_is_directory=True)
    print(f"🔗 快捷访问: {latest_link} -> {test_run_dir.name}")

    return test_run_dir


if __name__ == "__main__":
    run_test_with_preserved_results()
