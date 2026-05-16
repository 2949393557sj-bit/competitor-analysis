# ============================================================
# 竞品分析系统 - 数据模型定义
# 使用 Python dataclass 定义系统中所有核心数据结构
# 这些模型贯穿整个分析流水线：输入 → 调研 → 分析 → 报告
# ============================================================

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---- 分析模式枚举 ----
# 系统支持四种分析场景，由用户输入自动识别
class AnalysisMode(Enum):
    COMPARE = "指定竞品对比"          # 已知竞品，做 A vs B 的详细对比
    FIND_COMPETITORS = "找竞品"       # 不知道竞品是谁，帮用户发现替代产品
    PAIN_POINTS = "发现痛点"          # 分析产品和竞品的用户不满与市场空白
    FEATURE_DESIGN = "功能设计辅助"   # 描述想做的功能，找已有方案、分析优劣、发现差异化机会


# ---- 用户输入模型 ----
# 解析后的结构化用户请求，包含：目标产品、分析模式、竞品列表、原始上下文
@dataclass
class UserInput:
    product: str                              # 要分析的目标产品名称
    mode: AnalysisMode                        # 分析模式（对比/找竞品/痛点/功能设计）
    competitors: list[str] = field(default_factory=list)  # 已知竞品列表（对比模式下使用）
    extra_context: str = ""                   # 用户原始输入文本，保留完整上下文


# ---- Agent A 调研报告：功能特性 ----
# 负责收集产品的功能特性、能力对比等信息
@dataclass
class FeatureReport:
    agent_name: str = "Agent_A_Features"           # 所属 Agent 标识
    search_queries: list[str] = field(default_factory=list)  # 实际执行的搜索查询列表
    raw_results: list[dict] = field(default_factory=list)    # 搜索引擎返回的原始结果
    summary: str = ""                                      # LLM 摘要（待接入）


# ---- Agent B 调研报告：商业数据 ----
# 负责收集融资、营收、用户增长、市场份额等商业指标
@dataclass
class BusinessReport:
    agent_name: str = "Agent_B_Business"
    search_queries: list[str] = field(default_factory=list)
    raw_results: list[dict] = field(default_factory=list)
    summary: str = ""


# ---- Agent C 调研报告：用户反馈 ----
# 负责收集应用商店评论、社交媒体讨论等用户声音
@dataclass
class UserFeedbackReport:
    agent_name: str = "Agent_C_UserFeedback"
    search_queries: list[str] = field(default_factory=list)
    app_store_reviews: list[dict] = field(default_factory=list)    # 应用商店评论数据
    social_media_mentions: list[dict] = field(default_factory=list)  # 社交媒体提及数据
    summary: str = ""


# ---- 三路调研汇总 ----
# 将三个调研 Agent 的结果打包在一起，供分析 Agent 消费
@dataclass
class ResearchResults:
    features: Optional[FeatureReport] = None         # 功能特性调研结果
    business: Optional[BusinessReport] = None        # 商业数据调研结果
    user_feedback: Optional[UserFeedbackReport] = None  # 用户反馈调研结果


# ---- 分析结果 ----
# 分析 Agent 对调研数据进行结构化处理后的产出
@dataclass
class AnalysisResult:
    key_findings: list[str] = field(default_factory=list)       # 核心发现
    comparison_table: dict = field(default_factory=dict)         # 对比表格（功能/商业/口碑）
    strengths_weaknesses: dict = field(default_factory=dict)     # 各产品优劣势
    pain_points: list[str] = field(default_factory=list)         # 用户痛点列表
    recommendations: list[str] = field(default_factory=list)     # 建议列表
    raw_analysis: str = ""                                       # 原始分析文本


# ---- 最终报告 ----
# Writer Agent 生成的最终 Markdown 报告
@dataclass
class FinalReport:
    title: str = ""                            # 报告标题
    markdown_content: str = ""                 # 完整的 Markdown 内容
    output_path: str = "report.md"             # 输出文件路径
