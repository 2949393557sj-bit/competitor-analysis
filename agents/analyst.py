"""分析 Agent（Analyst）

职责：
  - 接收三个调研 Agent 的原始数据
  - 根据分析框架类型选择对应的分析策略
  - 用 LLM（thinking 模式）进行深度分析
  - 产出结构化的分析结果（关键发现、对比表、优劣势、痛点、建议）

使用 DeepSeek-V4-Flash thinking 模式（核心分析，质量最重要）。
"""

from models import ResearchResults, AnalysisResult
from llm_client import call_llm_json


# ---- 分析提示词模板 ----
# 每种分析类型对应一个专门的提示词，指导 LLM 从不同角度分析

COMPARISON_ANALYSIS_PROMPT = """你是一个竞品分析专家。请对以下调研数据进行深度对比分析。

## 分析框架
{framework}

## 功能特性调研
{features}

## 商业数据调研
{business}

## 用户反馈调研
{user_feedback}

请从以下维度分析并输出 JSON：
1. key_findings：列出 3-5 个最关键的发现
2. comparison_table：各维度的对比要点（features/business/user_feedback）
3. strengths_weaknesses：每个产品的优劣势 {"产品名": {"strengths": [...], "weaknesses": [...]}}
4. recommendations：3-5 条可操作的建议

输出 JSON 格式：
{
  "key_findings": ["发现1", "发现2"],
  "comparison_table": {"features": "...", "business": "...", "user_feedback": "..."},
  "strengths_weaknesses": {"产品A": {"strengths": [], "weaknesses": []}},
  "recommendations": ["建议1", "建议2"]
}
"""

LANDSCAPE_ANALYSIS_PROMPT = """你是一个竞品分析专家。请分析以下调研数据，帮助用户发现竞品。

## 分析框架
{framework}

## 功能特性调研
{features}

## 商业数据调研
{business}

## 用户反馈调研
{user_feedback}

请从以下维度分析并输出 JSON：
1. key_findings：发现的主要竞品和替代方案
2. strengths_weaknesses：各竞品的定位和差异 {"竞品名": {"strengths": [], "weaknesses": []}}
3. recommendations：推荐关注哪些竞品及原因

输出 JSON 格式：
{
  "key_findings": ["发现1", "发现2"],
  "strengths_weaknesses": {"竞品A": {"strengths": [], "weaknesses": []}},
  "recommendations": ["建议1", "建议2"]
}
"""

PAIN_POINT_ANALYSIS_PROMPT = """你是一个用户痛点分析专家。请从以下调研数据中提取和归纳用户痛点。

## 分析框架
{framework}

## 功能特性调研
{features}

## 商业数据调研
{business}

## 用户反馈调研
{user_feedback}

请从以下维度分析并输出 JSON：
1. pain_points：归纳出 3-5 个核心用户痛点
2. key_findings：支撑痛点的详细发现
3. recommendations：针对每个痛点的机会和建议

输出 JSON 格式：
{
  "pain_points": ["痛点1", "痛点2"],
  "key_findings": ["发现1", "发现2"],
  "recommendations": ["建议1", "建议2"]
}
"""

FEATURE_DESIGN_PROMPT = """你是一个功能设计顾问。请分析以下调研数据，帮助用户做功能设计决策。

## 用户想做的功能/场景
{framework}

## 已有方案调研（别人怎么做的）
{features}

## 市场需求和采纳度
{business}

## 用户对现有方案的反馈
{user_feedback}

请从以下维度分析并输出 JSON：
1. key_findings：已有产品的实现方案和设计思路
2. pain_points：用户对现有方案的不满和痛点
3. strengths_weaknesses：各方案的优劣对比 {"方案/产品": {"strengths": [], "weaknesses": []}}
4. recommendations：差异化机会和功能取舍建议

输出 JSON 格式：
{
  "key_findings": ["方案1的做法", "方案2的做法"],
  "pain_points": ["用户不满1", "用户不满2"],
  "strengths_weaknesses": {"方案A": {"strengths": [], "weaknesses": []}},
  "recommendations": ["建议1", "建议2"]
}
"""


class AnalysisAgent:
    """分析 Agent：用 LLM（thinking 模式）对调研数据进行深度分析。"""

    def __init__(self):
        self.name = "Analyst"

    async def analyze(self, research: ResearchResults, framework: dict) -> AnalysisResult:
        """根据分析框架类型，调用 LLM 进行深度分析。

        Args:
            research:  三个调研 Agent 的汇总结果
            framework: 编排 Agent 生成的分析框架

        Returns:
            结构化的 AnalysisResult
        """
        analysis_type = framework.get("analysis_type", "head_to_head_comparison")

        # 提取各维度调研摘要
        features = research.features.summary if research.features else "暂无数据"
        business = research.business.summary if research.business else "暂无数据"
        user_feedback = research.user_feedback.summary if research.user_feedback else "暂无数据"

        # 根据分析类型选择对应的提示词
        prompt_map = {
            "head_to_head_comparison": COMPARISON_ANALYSIS_PROMPT,
            "competitive_landscape": LANDSCAPE_ANALYSIS_PROMPT,
            "pain_point_analysis": PAIN_POINT_ANALYSIS_PROMPT,
            "feature_design_assistance": FEATURE_DESIGN_PROMPT,
        }
        prompt_template = prompt_map.get(analysis_type, COMPARISON_ANALYSIS_PROMPT)

        # 填充提示词（用 replace 而非 format，避免 framework 字符串中的 {} 冲突）
        prompt = prompt_template
        prompt = prompt.replace("{framework}", str(framework))
        prompt = prompt.replace("{features}", features)
        prompt = prompt.replace("{business}", business)
        prompt = prompt.replace("{user_feedback}", user_feedback)

        # 调用 LLM（thinking 模式，核心分析需要深度推理）
        try:
            llm_result = await call_llm_json(
                system_prompt="你是一个专业的竞品分析专家。请基于调研数据进行深度分析，输出结构化 JSON。",
                user_prompt=prompt,
                thinking=True,
            )
        except Exception as e:
            # LLM 调用失败时返回基础结果
            return self._fallback_result(research, str(e))

        # 将 LLM 输出转换为 AnalysisResult
        return AnalysisResult(
            key_findings=llm_result.get("key_findings", []),
            comparison_table=llm_result.get("comparison_table", {}),
            strengths_weaknesses=llm_result.get("strengths_weaknesses", {}),
            pain_points=llm_result.get("pain_points", []),
            recommendations=llm_result.get("recommendations", []),
            raw_analysis=str(llm_result),
        )

    def _fallback_result(self, research: ResearchResults, error: str) -> AnalysisResult:
        """LLM 调用失败时的降级处理：直接使用调研摘要。"""
        result = AnalysisResult()
        result.raw_analysis = f"[LLM 分析失败: {error}]"

        if research.features:
            result.key_findings.append(research.features.summary)
        if research.business:
            result.key_findings.append(research.business.summary)
        if research.user_feedback:
            result.pain_points.append(research.user_feedback.summary)

        return result
