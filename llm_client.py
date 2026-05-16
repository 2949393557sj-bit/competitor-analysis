"""LLM 调用客户端

基于 OpenAI 兼容 SDK 封装 DeepSeek-V4-Flash 调用。
支持两种模式：
  - thinking=True:  开启深度推理（Orchestrator、Analyst）
  - thinking=False: 快速模式（Research Agent、Writer）
"""

import json
from openai import AsyncOpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

# 初始化异步客户端（全局复用连接池）
client = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    thinking: bool = False,
    temperature: float = 0.1,
) -> str:
    """调用 LLM 并返回纯文本响应。

    Args:
        system_prompt: 系统提示词（设定角色和输出格式）
        user_prompt:   用户提示词（具体任务内容）
        thinking:      是否开启深度推理模式
                       True  → Orchestrator/Analyst 等需要复杂推理的场景
                       False → Research/Writer 等简单任务
        temperature:   生成温度，越低越确定性

    Returns:
        LLM 响应的文本内容
    """
    # 构建请求参数
    kwargs = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    # DeepSeek thinking 模式开关
    # thinking=True  → {"thinking": {"type": "enabled"}}  深度推理
    # thinking=False → {"thinking": {"type": "disabled"}} 快速响应
    if thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


async def call_llm_json(
    system_prompt: str,
    user_prompt: str,
    thinking: bool = False,
    temperature: float = 0.1,
) -> dict:
    """调用 LLM 并解析 JSON 响应。

    用于 Orchestrator 等需要结构化输出的场景。
    会强制 LLM 返回 JSON 格式，并自动解析为 dict。

    Args:
        system_prompt: 系统提示词（必须要求返回 JSON）
        user_prompt:   用户提示词
        thinking:      是否开启深度推理
        temperature:   生成温度

    Returns:
        解析后的 JSON 字典

    Raises:
        json.JSONDecodeError: LLM 返回的内容不是有效 JSON
    """
    kwargs = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},  # 强制 JSON 输出
        "temperature": temperature,
    }

    if thinking:
        kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
    else:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    response = await client.chat.completions.create(**kwargs)
    raw = response.choices[0].message.content
    return json.loads(raw)
