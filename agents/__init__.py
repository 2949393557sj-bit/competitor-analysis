# ============================================================
# agents 包初始化
# 统一导出所有 Agent 类，方便外部直接 import
# ============================================================

from .orchestrator import OrchestratorAgent
from .researcher import ResearchAgentA, ResearchAgentB, ResearchAgentC, run_research_agents
from .analyst import AnalysisAgent
from .writer import WriterAgent

__all__ = [
    "OrchestratorAgent",       # 编排 Agent：意图识别 + 框架选择
    "ResearchAgentA",          # 调研 Agent A：功能特性调研
    "ResearchAgentB",          # 调研 Agent B：商业数据调研
    "ResearchAgentC",          # 调研 Agent C：用户反馈调研
    "run_research_agents",     # 并行启动三个调研 Agent 的便捷函数
    "AnalysisAgent",           # 分析 Agent：结构化处理调研数据
    "WriterAgent",             # 写作 Agent：生成 Markdown 报告
]
