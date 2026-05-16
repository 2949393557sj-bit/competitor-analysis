"""编排 Agent（Orchestrator）

职责（三步合一，一次 LLM 调用完成）：
  1. 意图识别：判断用户想做什么（四选一）
  2. 框架选择：根据意图确定分析策略
  3. 任务分配：为三个 Research Agent 生成具体的结构化任务

找竞品模式特殊流程（两轮调用）：
  第一轮 LLM → 识别意图，提取产品名
  Tavily 搜索 → 获取竞品候选列表
  第二轮 LLM → 整理竞品名单，生成结构化任务

使用 DeepSeek-V4-Flash thinking 模式（需要强推理能力）。
"""

import json
from models import UserInput, AnalysisMode
from llm_client import call_llm_json
from tools.tavily_search import tavily_search


# ============================================================
# 提示词模板
# ============================================================

# ---- 第一步：意图识别 + 任务生成（非找竞品模式）----
INTENT_AND_TASK_PROMPT = """你是一个竞品分析系统的编排 Agent。请分析用户输入，完成以下三步：

## 第一步：意图识别
判断用户属于以下哪种模式：
- COMPARE：用户明确提到了要对比的产品（如 "A vs B"、"A和B对比"）
- FIND_COMPETITORS：用户想寻找竞品/替代品（如 "找竞品"、"有什么替代"）
- PAIN_POINTS：用户想发现痛点/问题（如 "痛点"、"用户抱怨"、"有什么不足"）
- FEATURE_DESIGN：用户想设计/实现某个功能，需要参考已有方案（如 "怎么做X"、"如何实现"、"功能设计"）

**重要**：如果你不认识用户提到的某个产品、工具或术语（比如你不确定它是什么），请在 `needs_context` 中列出需要搜索确认的关键词。

## 第二步：分析框架
根据意图确定分析策略和关注维度。

## 第三步：任务分配
为三个 Research Agent 生成具体的搜索任务：
- Agent_A_Features：功能特性调研（功能对比、能力差异、技术实现）
- Agent_B_Business：商业数据调研（融资、营收、用户增长、市场份额）
- Agent_C_UserFeedback：用户反馈调研（应用商店评论、社交媒体讨论）

每个任务包含：objective（目标）、search_queries（建议搜索词）、focus_areas（关注维度）

## 输出 JSON 格式
{
  "mode": "COMPARE|FIND_COMPETITORS|PAIN_POINTS|FEATURE_DESIGN",
  "product": "目标产品名称",
  "competitors": ["竞品1", "竞品2"],
  "reasoning": "简要说明你的判断依据",
  "needs_context": ["你不认识的术语或产品名，如 OpenClaw、龙虾等"],
  "framework": {
    "analysis_type": "对应分析类型标识",
    "research_focus": {
      "features": "功能维度搜索方向",
      "business": "商业维度搜索方向",
      "user_feedback": "用户反馈维度搜索方向"
    }
  },
  "tasks": [
    {
      "agent_name": "Agent_A_Features",
      "objective": "任务目标",
      "search_queries": ["搜索词1", "搜索词2", "搜索词3"],
      "focus_areas": ["关注点1", "关注点2"]
    },
    {
      "agent_name": "Agent_B_Business",
      "objective": "任务目标",
      "search_queries": ["搜索词1", "搜索词2"],
      "focus_areas": ["关注点1", "关注点2"]
    },
    {
      "agent_name": "Agent_C_UserFeedback",
      "objective": "任务目标",
      "search_queries": ["搜索词1", "搜索词2"],
      "focus_areas": ["关注点1", "关注点2"]
    }
  ]
}
"""

# ---- 上下文补充提示词 ----
# 当 LLM 不认识用户提到的产品/术语时，用搜索结果补充上下文
CONTEXT_ENRICHMENT_PROMPT = """你之前不认识以下术语/产品，我们已经通过搜索找到了相关信息。
请基于这些信息重新理解用户意图，并更新你的分析。

## 需要补充的术语
{terms}

## 搜索结果
{search_results}

## 用户原始输入
{raw_input}

请重新分析用户意图，输出与之前相同格式的 JSON（包含 mode、product、competitors、reasoning、framework、tasks）。
特别注意：现在你已经了解了这些术语，请基于搜索结果生成更精准的搜索任务。
"""

# ---- 找竞品模式：第一步 —— 识别意图 + 提取产品名 ----
COMPETITOR_INTENT_PROMPT = """你是一个竞品分析系统的编排 Agent。用户想找某个产品的竞品/替代品。

请分析用户输入，提取：
1. 目标产品名称
2. 用户的具体需求场景（用于后续搜索竞品）

输出 JSON 格式：
{
  "product": "目标产品名称",
  "use_case": "用户需求场景描述",
  "reasoning": "判断依据"
}
"""

# ---- 找竞品模式：第二步 —— 整理竞品 + 生成任务 ----
COMPETITOR_TASK_PROMPT = """你是一个竞品分析系统的编排 Agent。用户想找竞品，我们已经通过搜索找到了一些候选结果。

## 用户信息
- 目标产品：{product}
- 用户场景：{use_case}
- 用户原始输入：{raw_input}

## 搜索结果
{search_results}

## 你的任务
1. 从搜索结果中筛选出最相关的 3-5 个竞品/替代品
2. 为三个 Research Agent 生成具体的调研任务（重点围绕已发现的竞品）

## 输出 JSON 格式
{
  "competitors": ["竞品1", "竞品2", "竞品3"],
  "reasoning": "筛选依据",
  "framework": {
    "analysis_type": "competitive_landscape",
    "research_focus": {
      "features": "功能维度搜索方向",
      "business": "商业维度搜索方向",
      "user_feedback": "用户反馈维度搜索方向"
    }
  },
  "tasks": [
    {
      "agent_name": "Agent_A_Features",
      "objective": "任务目标",
      "search_queries": ["搜索词1", "搜索词2", "搜索词3"],
      "focus_areas": ["关注点1", "关注点2"]
    },
    {
      "agent_name": "Agent_B_Business",
      "objective": "任务目标",
      "search_queries": ["搜索词1", "搜索词2"],
      "focus_areas": ["关注点1", "关注点2"]
    },
    {
      "agent_name": "Agent_C_UserFeedback",
      "objective": "任务目标",
      "search_queries": ["搜索词1", "搜索词2"],
      "focus_areas": ["关注点1", "关注点2"]
    }
  ]
}
"""


# ============================================================
# Orchestrator Agent
# ============================================================


class OrchestratorAgent:
    """顶层编排 Agent：意图识别 → 框架选择 → 任务分配。

    所有 LLM 调用使用 thinking 模式（需要强推理能力）。
    """

    def __init__(self):
        self.name = "Orchestrator"

    async def arun(self, raw_input: str) -> dict:
        """主入口：执行完整的编排流程。

        流程：
          1. 第一轮 LLM 调用：意图识别
          2. 如果 LLM 不认识某些术语 → 自动搜索补充上下文 → 第二轮 LLM 重新理解
          3. 找竞品模式 → 额外搜索竞品候选
          4. 输出结构化结果

        Args:
            raw_input: 用户原始查询文本

        Returns:
            {
                "user_input": UserInput,
                "framework": dict,
                "tasks": list[dict]  # 三个 Agent 的结构化任务
            }
        """
        # ---- 第一轮 LLM 调用：意图识别 ----
        first_pass = await call_llm_json(
            system_prompt=INTENT_AND_TASK_PROMPT,
            user_prompt=raw_input,
            thinking=True,
        )

        # ---- 上下文补充：LLM 不认识的术语，自动搜索 ----
        needs_context = first_pass.get("needs_context", [])
        if needs_context:
            print(f"[Orchestrator] 发现未知术语: {needs_context}，自动搜索补充上下文...")
            first_pass = await self._enrich_context(raw_input, first_pass, needs_context)

        mode = first_pass.get("mode", "FIND_COMPETITORS")

        if mode == "FIND_COMPETITORS":
            # 找竞品模式：两轮调用流程
            return await self._run_competitor_flow(raw_input, first_pass)
        else:
            # 非找竞品模式：一步到位
            return self._build_result(raw_input, first_pass)

    async def _enrich_context(
        self, raw_input: str, first_pass: dict, terms: list[str]
    ) -> dict:
        """当 LLM 不认识某些术语时，自动搜索补充上下文。

        流程：
          1. 为每个未知术语构造搜索查询
          2. 并行调用 Tavily 搜索
          3. 将搜索结果注入提示词，让 LLM 重新理解意图

        Args:
            raw_input:    用户原始输入
            first_pass:   第一轮 LLM 输出
            terms:        LLM 不认识的术语列表

        Returns:
            补充上下文后的 LLM 输出（同 first_pass 格式）
        """
        import asyncio

        # 为每个术语构造搜索查询
        search_queries = [f"{term} 是什么 产品 功能" for term in terms]
        search_tasks = [tavily_search(query=q, max_results=5) for q in search_queries]
        results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 汇总搜索结果
        results_text = ""
        for i, results in enumerate(results_raw):
            results_text += f"\n### {terms[i]} 的搜索结果:\n"
            if isinstance(results, Exception):
                results_text += f"搜索失败: {results}\n"
            else:
                for r in results:
                    results_text += f"- {r.get('title', '')}: {r.get('content', '')[:300]}\n"

        # 第二轮 LLM：基于搜索结果重新理解意图
        enriched = await call_llm_json(
            system_prompt=CONTEXT_ENRICHMENT_PROMPT.format(
                terms=", ".join(terms),
                search_results=results_text,
                raw_input=raw_input,
            ),
            user_prompt="请基于搜索结果重新分析用户意图并输出 JSON。",
            thinking=True,
        )

        return enriched

    async def _run_competitor_flow(self, raw_input: str, first_pass: dict) -> dict:
        """找竞品模式的两轮调用流程。

        流程：
          1. 从第一轮结果提取产品名
          2. 调用 Tavily 搜索竞品候选
          3. 第二轮 LLM 整理竞品名单并生成任务
        """
        product = first_pass.get("product", raw_input.split()[0])
        use_case = first_pass.get("use_case", raw_input)

        # ---- Tavily 搜索竞品候选 ----
        search_queries = [
            f"{product} alternatives competitors",
            f"{product} vs 替代品 竞品",
            f"best {product} alternatives 2026",
        ]
        search_tasks = [tavily_search(query=q, max_results=5) for q in search_queries]
        import asyncio
        search_results_raw = await asyncio.gather(*search_tasks, return_exceptions=True)

        # 汇总搜索结果为文本
        results_text = ""
        for i, results in enumerate(search_results_raw):
            if isinstance(results, Exception):
                results_text += f"\n查询 {search_queries[i]}: 错误 - {results}\n"
            else:
                for r in results:
                    results_text += f"- [{r.get('title', '')}]({r.get('url', '')})\n  {r.get('content', '')[:200]}\n"

        # ---- 第二轮 LLM 调用：整理竞品 + 生成任务 ----
        # 用 .replace() 替代 .format()，避免搜索结果中的 {} 导致 KeyError
        system_prompt = COMPETITOR_TASK_PROMPT
        system_prompt = system_prompt.replace("{product}", product)
        system_prompt = system_prompt.replace("{use_case}", use_case)
        system_prompt = system_prompt.replace("{raw_input}", raw_input)
        system_prompt = system_prompt.replace("{search_results}", results_text)

        second_pass = await call_llm_json(
            system_prompt=system_prompt,
            user_prompt="请根据搜索结果筛选竞品并生成调研任务。",
            thinking=True,
        )

        # 合并两轮结果
        second_pass["mode"] = "FIND_COMPETITORS"
        second_pass["product"] = product
        return self._build_result(raw_input, second_pass)

    def _build_result(self, raw_input: str, llm_output: dict) -> dict:
        """将 LLM 输出组装为标准结果格式。

        Args:
            raw_input:  用户原始输入
            llm_output: LLM 返回的 JSON 字典

        Returns:
            标准化的编排结果
        """
        # 解析分析模式
        mode_map = {
            "COMPARE": AnalysisMode.COMPARE,
            "FIND_COMPETITORS": AnalysisMode.FIND_COMPETITORS,
            "PAIN_POINTS": AnalysisMode.PAIN_POINTS,
            "FEATURE_DESIGN": AnalysisMode.FEATURE_DESIGN,
        }
        mode = mode_map.get(llm_output.get("mode", ""), AnalysisMode.FIND_COMPETITORS)

        # 构建 UserInput
        user_input = UserInput(
            product=llm_output.get("product", raw_input.split()[0]),
            mode=mode,
            competitors=llm_output.get("competitors", []),
            extra_context=raw_input,
        )

        # 提取框架配置（LLM 可能返回字符串，需转为 dict）
        framework = llm_output.get("framework", {})
        if isinstance(framework, str):
            framework = {"analysis_type": framework}
        framework["mode"] = user_input.mode.value
        framework["product"] = user_input.product
        framework["competitors"] = user_input.competitors

        # 提取任务列表（过滤掉非 dict 的异常项）
        tasks = [t for t in llm_output.get("tasks", []) if isinstance(t, dict)]

        # 如果 LLM 没有生成任务，用框架信息生成默认任务
        if not tasks:
            tasks = self._generate_default_tasks(framework)

        return {
            "user_input": user_input,
            "framework": framework,
            "tasks": tasks,
        }

    def _generate_default_tasks(self, framework: dict) -> list[dict]:
        """当 LLM 没有生成任务时，用框架信息生成默认任务。

        根据 framework 中的 research_focus 为每个 Agent 生成基础搜索任务。
        """
        focus = framework.get("research_focus", {})
        product = framework.get("product", "")
        competitors = framework.get("competitors", [])
        names = [product] + competitors if competitors else [product]

        return [
            {
                "agent_name": "Agent_A_Features",
                "objective": f"调研 {', '.join(names)} 的功能特性和技术实现",
                "search_queries": [
                    f"{product} features capabilities",
                    f"{' vs '.join(names)} feature comparison" if len(names) > 1 else f"{product} review",
                ],
                "focus_areas": ["功能对比", "技术实现", "产品差异"],
            },
            {
                "agent_name": "Agent_B_Business",
                "objective": f"调研 {', '.join(names)} 的商业数据",
                "search_queries": [
                    f"{product} funding revenue users",
                    f"{product} market share valuation",
                ],
                "focus_areas": ["融资", "营收", "用户增长", "市场份额"],
            },
            {
                "agent_name": "Agent_C_UserFeedback",
                "objective": f"调研 {', '.join(names)} 的用户反馈",
                "search_queries": [
                    f"{product} user reviews pros cons",
                    f"{product} 用户评价 优缺点",
                ],
                "focus_areas": ["用户评价", "常见抱怨", "满意度"],
            },
        ]

    # ---- 保留旧接口兼容 ----

    def parse_input(self, raw_input: str) -> UserInput:
        """同步版本的意图识别（简单关键词匹配，作为 fallback）。"""
        mode = self._detect_mode(raw_input)
        competitors = self._extract_competitors(raw_input)
        product = self._extract_product(raw_input)
        return UserInput(
            product=product,
            mode=mode,
            competitors=competitors,
            extra_context=raw_input,
        )

    def select_framework(self, user_input: UserInput) -> dict:
        """同步版本的框架选择（基于规则，作为 fallback）。"""
        base_framework = {
            "mode": user_input.mode.value,
            "product": user_input.product,
            "competitors": user_input.competitors,
        }

        if user_input.mode == AnalysisMode.COMPARE:
            base_framework["research_focus"] = {
                "features": f"Compare features of {user_input.product} vs {', '.join(user_input.competitors)}",
                "business": f"Compare business metrics of {user_input.product} vs {', '.join(user_input.competitors)}",
                "user_feedback": f"Compare user reviews of {user_input.product} vs {', '.join(user_input.competitors)}",
            }
            base_framework["analysis_type"] = "head_to_head_comparison"

        elif user_input.mode == AnalysisMode.FIND_COMPETITORS:
            base_framework["research_focus"] = {
                "features": f"What features does {user_input.product} offer? Who are the closest alternatives?",
                "business": f"Market landscape and competitors of {user_input.product}",
                "user_feedback": f"What alternatives do users recommend instead of {user_input.product}?",
            }
            base_framework["analysis_type"] = "competitive_landscape"

        elif user_input.mode == AnalysisMode.PAIN_POINTS:
            base_framework["research_focus"] = {
                "features": f"What are the feature gaps and limitations of {user_input.product} and competitors?",
                "business": f"Market trends and unserved segments around {user_input.product}",
                "user_feedback": f"What are the most common complaints about {user_input.product}?",
            }
            base_framework["analysis_type"] = "pain_point_analysis"

        elif user_input.mode == AnalysisMode.FEATURE_DESIGN:
            base_framework["research_focus"] = {
                "features": f"What products solve similar use case: {user_input.product}? How do they implement this feature?",
                "business": f"Market demand and user adoption for features like: {user_input.product}",
                "user_feedback": f"What do users complain about regarding {user_input.product} features? What gaps exist?",
            }
            base_framework["analysis_type"] = "feature_design_assistance"

        return base_framework

    def _detect_mode(self, raw_input: str) -> AnalysisMode:
        """从原始文本中检测分析模式（关键词匹配 fallback）。"""
        lower = raw_input.lower()
        if any(kw in lower for kw in ["对比", "compare", "vs", "versus", "和...比"]):
            return AnalysisMode.COMPARE
        elif any(kw in lower for kw in ["竞品", "competitor", "alternative", "替代", "找"]):
            return AnalysisMode.FIND_COMPETITORS
        elif any(kw in lower for kw in ["痛点", "pain", "complaint", "问题", "不足"]):
            return AnalysisMode.PAIN_POINTS
        elif any(kw in lower for kw in ["怎么做", "如何做", "功能设计", "设计方案", "实现方案", "辅助设计", "功能取舍"]):
            return AnalysisMode.FEATURE_DESIGN
        return AnalysisMode.FIND_COMPETITORS

    def _extract_competitors(self, raw_input: str) -> list[str]:
        """从输入中提取竞品名称列表（占位 fallback）。"""
        return []

    def _extract_product(self, raw_input: str) -> str:
        """从输入中提取目标产品名称（占位 fallback）。"""
        return raw_input.split()[0] if raw_input.split() else "Unknown Product"
