"""配置文件加载模块"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "config.yaml"

        self.config_path = Path(config_path)
        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 处理环境变量替换
        config = self._replace_env_vars(config)
        return config

    def _replace_env_vars(self, obj):
        """递归替换 ${VAR} 格式的环境变量"""
        if isinstance(obj, dict):
            return {k: self._replace_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_env_vars(item) for item in obj]
        elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
            env_var = obj[2:-1]
            return os.environ.get(env_var, obj)
        return obj

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项，支持点分隔的键"""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    @property
    def interval(self) -> int:
        return self.get("interval", 60)

    @property
    def output_dir(self) -> str:
        return self.get("output_dir", "photos")

    @property
    def log_dir(self) -> str:
        return self.get("log_dir", "logs")

    @property
    def llm_config(self) -> Dict[str, Any]:
        return self.get("llm", {})

    @property
    def image_config(self) -> Dict[str, Any]:
        return self.get("image", {})

    @property
    def telegram_config(self) -> Dict[str, Any]:
        return self.get("telegram", {})

    @property
    def detection_config(self) -> Dict[str, Any]:
        return self.get("detection", {})

    @property
    def vllm_config(self) -> Dict[str, Any]:
        return self.get("vllm", self.get("llm", {}))  # 默认使用 llm 配置


# 全局配置实例
config = Config()
