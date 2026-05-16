"""调研 Agents（Research Agents）

三个并行执行的调研 Agent，各自负责不同维度的数据采集：
  - Agent A（功能特性）：搜索产品的功能对比、能力差异
  - Agent B（商业数据）：搜索融资、营收、用户增长等商业指标
  - Agent C（用户反馈）：抓取应用商店评论和社交媒体上的用户声音

三个 Agent 互不依赖，通过 asyncio.gather 并行执行。
使用 DeepSeek-V4-Flash non-thinking 模式（任务简单，不需要深度推理）。
"""

import asyncio
from abc import ABC, abstractmethod

from models import FeatureReport, BusinessReport, UserFeedbackReport, ResearchResults
from tools.tavily_search import tavily_search
from tools.app_store import fetch_app_store_reviews, fetch_social_mentions
from llm_client import call_llm


# ---- LLM 摘要提示词 ----
# 用于将原始搜索结果综合成结构化摘要（non-thinking 模式）
SYNTHESIZE_PROMPT = """你是一个竞品分析调研助手。请将以下搜索结果整理成一段结构化的调研摘要。

要求：
1. 提取关键事实和数据点
2. 去除重复和无关信息
3. 用中文输出，简洁明了
4. 如果搜索结果是占位数据（包含 [PLACEHOLDER]），直接说明"暂无真实数据"

搜索结果：
{results}
"""


class BaseResearchAgent(ABC):
    """调研 Agent 基类，定义通用接口和工具方法。"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def research(self, framework: dict, task: dict = None) -> dict:
        """根据框架和任务执行调研（子类必须实现）。

        Args:
            framework: 分析框架配置
            task: Orchestrator 分配的结构化任务（可选，有则优先使用）
        """
        ...

    def _build_queries(self, focus: str, base_terms: list[str]) -> list[str]:
        """根据搜索方向和基础关键词生成搜索查询列表（fallback）。"""
        queries = []
        for term in base_terms:
            queries.append(f"{focus} {term}")
        return queries

    async def _synthesize(self, raw_results: list[dict]) -> str:
        """用 LLM 将原始搜索结果综合成摘要（non-thinking 模式）。"""
        if not raw_results:
            return ""

        # 拼接搜索结果为文本
        results_text = ""
        for r in raw_results[:10]:  # 最多取前 10 条避免过长
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            results_text += f"- {title}: {content}\n"

        try:
            summary = await call_llm(
                system_prompt="你是一个专业的竞品分析助手，擅长从搜索结果中提取关键信息。",
                user_prompt=SYNTHESIZE_PROMPT.replace("{results}", results_text),
                thinking=False,  # Research Agent 用 non-thinking 模式
            )
            return summary
        except Exception as e:
            return f"[LLM 摘要生成失败: {e}]"


class ResearchAgentA(BaseResearchAgent):
    """Agent A：功能特性调研

    搜索并收集产品的功能对比信息，包括：
      - 功能列表和能力差异
      - 各产品的独特功能
      - 功能对比评测文章
    """

    def __init__(self):
        super().__init__("Agent_A_Features")

    async def research(self, framework: dict, task: dict = None) -> FeatureReport:
        report = FeatureReport(agent_name=self.name)
        product = framework["product"]
        competitors = framework.get("competitors", [])

        # 优先使用 Orchestrator 分配的任务
        if task:
            report.search_queries = task.get("search_queries", [])
        else:
            # fallback：自行生成搜索查询
            focus = framework["research_focus"]["features"]
            terms = ["features", "capabilities", "functionality comparison"]
            if competitors:
                terms.append(" vs ".join([product] + competitors))
            report.search_queries = self._build_queries(focus, terms)

        # 并行执行所有搜索查询
        search_tasks = [tavily_search(query=q) for q in report.search_queries]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                report.raw_results.append({"error": str(result)})
            else:
                report.raw_results.extend(result)

        # 用 LLM 综合搜索结果为摘要（non-thinking 模式）
        report.summary = await self._synthesize(report.raw_results)
        return report


class ResearchAgentB(BaseResearchAgent):
    """Agent B：商业数据调研

    搜索并收集产品的商业表现数据，包括：
      - 融资金额和投资方
      - 营收规模和增长趋势
      - 用户数量和市场份额
      - 估值和上市情况
    """

    def __init__(self):
        super().__init__("Agent_B_Business")

    async def research(self, framework: dict, task: dict = None) -> BusinessReport:
        report = BusinessReport(agent_name=self.name)
        product = framework["product"]
        competitors = framework.get("competitors", [])

        # 优先使用 Orchestrator 分配的任务
        if task:
            report.search_queries = task.get("search_queries", [])
        else:
            # fallback：自行生成搜索查询
            focus = framework["research_focus"]["business"]
            terms = ["funding", "revenue", "user growth", "market share", "valuation"]
            if competitors:
                for comp in competitors:
                    terms.append(f"{comp} funding users")
            report.search_queries = self._build_queries(focus, terms)

        # 并行执行搜索
        search_tasks = [tavily_search(query=q) for q in report.search_queries]
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                report.raw_results.append({"error": str(result)})
            else:
                report.raw_results.extend(result)

        # 用 LLM 综合搜索结果为摘要
        report.summary = await self._synthesize(report.raw_results)
        return report


class ResearchAgentC(BaseResearchAgent):
    """Agent C：用户反馈调研

    从两个渠道收集用户声音：
      1. 应用商店评论（Google Play / App Store）
      2. 社交媒体讨论（Reddit、Twitter 等）

    同时还会进行通用网页搜索，获取用户评价文章。
    """

    def __init__(self):
        super().__init__("Agent_C_UserFeedback")

    async def research(self, framework: dict, task: dict = None) -> UserFeedbackReport:
        report = UserFeedbackReport(agent_name=self.name)
        product = framework["product"]
        competitors = framework.get("competitors", [])
        all_names = [product] + competitors

        # ---- 并行抓取应用商店评论 + 社交媒体提及 ----
        tasks = []
        for name in all_names:
            tasks.append(fetch_app_store_reviews(app_name=name))
            tasks.append(fetch_social_mentions(product_name=name))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 分拣结果：偶数索引 = 应用商店评论，奇数索引 = 社交媒体提及
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            if i % 2 == 0:
                report.app_store_reviews.extend(result)
            else:
                report.social_media_mentions.extend(result)

        # ---- 通用网页搜索：用户评价和优缺点 ----
        if task:
            sentiment_queries = task.get("search_queries", ["user reviews", "pros cons"])
        else:
            focus = framework["research_focus"]["user_feedback"]
            sentiment_queries = self._build_queries(focus, ["user reviews", "pros cons"])

        search_tasks = [tavily_search(query=q) for q in sentiment_queries]
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        for result in search_results:
            if isinstance(result, Exception):
                report.search_queries.append(str(result))
            else:
                report.search_queries.extend(result)

        # 用 LLM 综合用户反馈为摘要
        all_feedback = report.app_store_reviews + report.social_media_mentions
        report.summary = await self._synthesize(all_feedback)
        return report


async def run_research_agents(framework: dict, tasks: list[dict] = None) -> ResearchResults:
    """并行启动三个调研 Agent 并汇总结果。

    Args:
        framework: 分析框架配置
        tasks:     Orchestrator 分配的结构化任务列表（可选）

    Returns:
        包含三路调研结果的 ResearchResults 对象
    """
    agent_a = ResearchAgentA()
    agent_b = ResearchAgentB()
    agent_c = ResearchAgentC()

    # 将任务列表映射到对应的 Agent
    task_map = {}
    if tasks:
        for t in tasks:
            task_map[t.get("agent_name", "")] = t

    # 三个 Agent 并行执行
    results = await asyncio.gather(
        agent_a.research(framework, task_map.get("Agent_A_Features")),
        agent_b.research(framework, task_map.get("Agent_B_Business")),
        agent_c.research(framework, task_map.get("Agent_C_UserFeedback")),
        return_exceptions=True,
    )

    # 汇总结果
    research = ResearchResults()
    for result in results:
        if isinstance(result, Exception):
            continue
        if isinstance(result, FeatureReport):
            research.features = result
        elif isinstance(result, BusinessReport):
            research.business = result
        elif isinstance(result, UserFeedbackReport):
            research.user_feedback = result

    return research
