#!/bin/bash
# Vision Skill 测试运行脚本
# 使用 venv 环境运行测试

set -e

echo "========================================="
echo "Vision Skill Test Suite"
echo "========================================="

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo "错误: 请在项目根目录运行此脚本"
    exit 1
fi

# 检查 venv 是否存在
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

# 激活虚拟环境
echo "激活虚拟环境..."
source .venv/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -q -r requirements.txt

# 生成测试资源
echo ""
echo "生成测试资源..."
python tests/utils/video_generator.py tests/fixtures/videos

# 运行测试
echo ""
echo "========================================="
echo "运行静态视觉测试 (VLLM)..."
echo "========================================="
pytest tests/static/test_vllm_vision.py -v --timeout=120 || true

echo ""
echo "========================================="
echo "运行动态视觉测试 (CV+VLLM)..."
echo "========================================="
pytest tests/dynamic/test_monitoring.py -v --timeout=120 || true

echo ""
echo "========================================="
echo "运行 CV 算法测试..."
echo "========================================="
pytest tests/static/test_image_analysis.py -v || true

echo ""
echo "========================================="
echo "测试完成!"
echo "========================================="
echo ""
echo "查看详细报告:"
echo "  静态测试: pytest tests/static/test_vllm_vision.py -v"
echo "  动态测试: pytest tests/dynamic/test_monitoring.py -v"
echo "  全部测试: pytest tests/ -v"
