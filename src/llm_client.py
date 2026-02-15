"""LLM 客户端模块 - 调用 OpenAI/Claude API 描述图像"""
import base64
import json
from pathlib import Path
from typing import Optional
import requests

from .config import config
from .logger import logger


class LLMClient:
    """LLM API 客户端"""
    
    def __init__(self, provider: str = None, api_key: str = None, model: str = None, base_url: str = None):
        llm_config = config.llm_config
        
        self.provider = provider or llm_config.get('provider', 'openai')
        self.api_key = api_key or llm_config.get('api_key')
        self.model = model or llm_config.get('model', 'gpt-4o-mini')
        self.base_url = base_url or llm_config.get('base_url', 'https://api.openai.com/v1')
        
        if not self.api_key:
            raise ValueError("API key 未配置")
    
    def _encode_image(self, image_path: str) -> str:
        """将图像编码为 base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def describe_image(self, image_path: str, prompt: str = None) -> str:
        """调用 LLM 描述图像"""
        image_path = Path(image_path)
        
        if not image_path.exists():
            logger.error(f"图像文件不存在: {image_path}")
            return None
        
        if self.provider == 'openai' or self.provider == 'siliconflow':
            return self._describe_with_openai(image_path, prompt)
        elif self.provider == 'anthropic':
            return self._describe_with_anthropic(image_path, prompt)
        else:
            logger.error(f"不支持的 LLM 提供商: {self.provider}")
            return None
    
    def _describe_with_openai(self, image_path: Path, prompt: str = None) -> str:
        """使用 OpenAI API 描述图像"""
        if prompt is None:
            prompt = "请详细描述这张图片的内容，包括所有可见的文本、图形、颜色和布局。"
        
        base64_image = self._encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            description = result['choices'][0]['message']['content']
            logger.info(f"图像描述成功: {image_path.name}")
            return description
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API 调用失败: {e}")
            return None
    
    def _describe_with_anthropic(self, image_path: Path, prompt: str = None) -> str:
        """使用 Anthropic API 描述图像"""
        if prompt is None:
            prompt = "请详细描述这张图片的内容。"
        
        base64_image = self._encode_image(image_path)
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64_image
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            description = result['content'][0]['text']
            logger.info(f"图像描述成功: {image_path.name}")
            return description
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API 调用失败: {e}")
            return None


# 全局 LLM 客户端实例
llm_client = None


def get_llm_client() -> LLMClient:
    """获取 LLM 客户端实例"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client
