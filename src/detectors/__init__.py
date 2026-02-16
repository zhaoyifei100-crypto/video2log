"""
探测器模块

自动发现和加载所有检测器
为 LLM 提供可用的 CV 模板菜单
"""

import importlib
import pkgutil
import re
from pathlib import Path
from typing import Dict, Type, Optional
from .base import BaseDetector, DetectionResult

# 自动发现所有检测器类
_DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def _discover_detectors():
    """自动发现 detectors 包中的所有检测器类"""
    global _DETECTOR_REGISTRY

    # 获取当前包路径
    package_path = Path(__file__).parent

    # 遍历所有模块
    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if module_name.startswith("_"):
            continue

        try:
            # 导入模块
            module = importlib.import_module(f".{module_name}", package=__name__)

            # 查找模块中的 Detector 类
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, BaseDetector)
                    and attr is not BaseDetector
                ):
                    # 注册检测器 - PascalCase 转 snake_case
                    import re

                    name = attr_name.replace("Detector", "")
                    # BlackScreen -> black_screen
                    detector_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
                    _DETECTOR_REGISTRY[detector_name] = attr

        except Exception as e:
            print(f"加载检测器 {module_name} 失败: {e}")


# 模块导入时自动发现
_discover_detectors()


def get_detector(name: str, params: Optional[dict] = None) -> BaseDetector:
    """
    获取检测器实例

    Args:
        name: 检测器名称 (black_screen, motion, ...)
        params: LLM 生成的参数

    Returns:
        BaseDetector 实例

    Raises:
        ValueError: 检测器不存在
    """
    if name not in _DETECTOR_REGISTRY:
        available = ", ".join(_DETECTOR_REGISTRY.keys())
        raise ValueError(f"未知检测器 '{name}'。可用: {available}")

    detector_class = _DETECTOR_REGISTRY[name]
    return detector_class(params)


def get_available_detectors() -> Dict[str, str]:
    """
    获取所有可用检测器及其描述

    Returns:
        {名称: LLM描述}
    """
    return {
        name: detector_class(None).get_llm_description()
        for name, detector_class in _DETECTOR_REGISTRY.items()
    }


def build_llm_menu() -> str:
    """
    为 LLM 构建检测器选择菜单

    Returns:
        格式化的菜单文本
    """
    menu_lines = ["## 可用 CV 检测器模板", ""]

    for name, detector_class in _DETECTOR_REGISTRY.items():
        desc = detector_class(None).get_llm_description()
        menu_lines.append(f"### {name}")
        menu_lines.append(desc)
        menu_lines.append("")

    menu_lines.append("---")
    menu_lines.append("请根据用户需求，选择一个检测器并配置参数。")
    menu_lines.append("返回格式:")
    menu_lines.append("```json")
    menu_lines.append("{")
    menu_lines.append('  "detector": "检测器名称",')
    menu_lines.append('  "params": {')
    menu_lines.append('    "参数名": 值')
    menu_lines.append("  }")
    menu_lines.append("}")
    menu_lines.append("```")

    return "\n".join(menu_lines)


# 导出
__all__ = [
    "BaseDetector",
    "DetectionResult",
    "get_detector",
    "get_available_detectors",
    "build_llm_menu",
]
