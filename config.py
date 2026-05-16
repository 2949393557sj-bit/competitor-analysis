# ============================================================
# 竞品分析系统 - 全局配置文件
# 统一管理 API 密钥、模型配置、输出路径等全局参数
# ============================================================

import os

# ---- Tavily 搜索 API ----
# Tavily 是一个专为 AI 应用设计的搜索引擎，用于获取网页搜索结果
# 需要在环境变量中设置 TAVILY_API_KEY，或在此处直接填写
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ---- LLM 大语言模型配置（DeepSeek-V4-Flash）----
# 所有 Agent 共用同一个模型，通过 thinking 开关区分推理深度
# - Orchestrator / Analyst：thinking=True  （需要深度推理）
# - Research / Writer：     thinking=False （简单任务，省 token）
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")

# ---- 输出配置 ----
# 分析报告的输出目录，最终生成的 Markdown 报告会保存到此目录
REPORT_OUTPUT_DIR = "output"
