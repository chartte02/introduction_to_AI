"""SJTU LLM API 客户端模块。

对接上海交通大学本地大模型 API（OpenAI 兼容格式）。
支持多种模型：DeepSeek、MiniMax、GLM、Qwen 等。
"""

import os
import time
from typing import Optional, List, Dict

from openai import OpenAI


# 默认配置
DEFAULT_BASE_URL = "https://models.sjtu.edu.cn/api/v1"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TIMEOUT = 120


class LLMClient:
    """SJTU LLM API 客户端。

    封装了与 SJTU 本地大模型 API 的交互，支持：
      - 文本生成（同步）
      - 多轮对话
      - 自动重试（指数退避）

    Args:
        api_key: API 密钥。未提供时从环境变量 SJTU_API_KEY 读取。
        base_url: API 基础地址。未提供时使用默认值。
        model: 默认模型名。
        max_tokens: 最大生成 token 数。
        temperature: 生成温度（0.0~1.0）。
        timeout: 请求超时时间（秒）。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.environ.get("SJTU_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "未提供 API key。请通过参数 api_key 传入，"
                "或设置环境变量 SJTU_API_KEY。"
            )

        self.base_url = base_url or os.environ.get(
            "SJTU_BASE_URL", DEFAULT_BASE_URL
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=timeout,
            max_retries=3,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """发送对话请求，返回模型生成的文本。

        Args:
            messages: 消息列表，格式为 [{"role": "...", "content": "..."}, ...]。
            model: 模型名，覆盖默认模型。
            max_tokens: 最大生成 token 数，覆盖默认值。
            temperature: 生成温度，覆盖默认值。

        Returns:
            模型生成的文本内容。

        Raises:
            RuntimeError: 多次重试后仍然失败。
        """
        model_name = model or self.model
        kwargs = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens

        if temperature is not None:
            kwargs["temperature"] = temperature
        else:
            kwargs["temperature"] = self.temperature

        # API 要求 V3.2 必须包含 user 角色消息
        has_user = any(m.get("role") == "user" for m in messages)
        if not has_user:
            raise ValueError("消息列表必须包含至少一条 role='user' 的消息")

        last_error = None
        for attempt in range(1, 6):  # 最多重试 5 次
            try:
                resp = self.client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                return content.strip() if content else ""

            except Exception as e:
                last_error = e
                status_code = getattr(e, "status_code", None) or getattr(
                    getattr(e, "response", None), "status_code", None
                )

                # 4xx（非 429）重试无意义，直接抛出
                if status_code and 400 <= status_code < 500 and status_code != 429:
                    raise RuntimeError(
                        f"API 请求失败 (HTTP {status_code}): {e}"
                    ) from e

                if attempt < 5:
                    wait = 2 ** attempt  # 指数退避：2, 4, 8, 16 秒
                    print(f"  [LLM] 请求失败 (尝试 {attempt}/5)，{wait} 秒后重试: {e}")
                    time.sleep(wait)

        raise RuntimeError(
            f"LLM API 请求多次重试后仍然失败: {last_error}"
        )

    def generate_explanation(
        self,
        text: str,
        label: int,
        label_name: str,
        confidence: float,
        model: Optional[str] = None,
    ) -> str:
        """基于推文文本和分类结果生成判断依据。

        这是对 chat() 的高层封装，自动组装 prompt 并解析结果。

        Args:
            text: 原始推文文本。
            label: 分类标签（0 或 1）。
            label_name: 标签名称（"谣言" 或 "非谣言"）。
            confidence: 模型置信度。
            model: 使用的 LLM 模型名。

        Returns:
            判断依据文本。
        """
        from model.prompts import build_explanation_prompt

        messages = build_explanation_prompt(
            text=text, label=label, label_name=label_name, confidence=confidence
        )
        return self.chat(messages, model=model)
