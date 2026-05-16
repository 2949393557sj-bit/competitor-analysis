"""写作 Agent（Writer）

职责：
  - 接收分析 Agent 的结构化分析结果
  - 根据分析类型选择对应的报告模板
  - 用 LLM（non-thinking 模式）格式化各章节内容
  - 组装成完整的 Markdown 报告并输出

使用 DeepSeek-V4-Flash non-thinking 模式（格式化输出，不需要深度推理）。
"""

from datetime import datetime

from models import AnalysisResult, UserInput, FinalReport
from llm_client import call_llm


# ---- 章节生成提示词 ----
SECTION_PROMPT = """你是一个竞品分析报告写作助手。请将以下分析数据整理成一段流畅的 Markdown 文段。

## 报告类型
{report_type}

## 章节标题
{section_title}

## 分析数据
{data}

要求：
1. 用中文输出
2. 语言专业但易读
3. 使用 Markdown 格式（标题、列表、加粗等）
4. 如果数据为空或占位，输出"暂无数据"
"""


class WriterAgent:
    """写作 Agent：将结构化分析结果转化为可读的 Markdown 报告。"""

    def __init__(self):
        self.name = "Writer"

    async def write(
        self,
        user_input: UserInput,
        analysis: AnalysisResult,
        framework: dict,
    ) -> FinalReport:
        """生成最终的 Markdown 报告。

        Args:
            user_input: 用户原始请求
            analysis:   分析 Agent 产出的结构化分析结果
            framework:  使用的分析框架配置

        Returns:
            包含完整 Markdown 内容的 FinalReport 对象
        """
        analysis_type = framework.get("analysis_type", "head_to_head_comparison")
        mode = framework.get("mode", "")

        # 根据分析类型选择报告模板
        # LLM 可能返回非标准 analysis_type，用 mode 做兜底
        if analysis_type == "competitive_landscape" or mode == "找竞品":
            md = await self._build_landscape_report(user_input, analysis, framework)
        elif analysis_type == "pain_point_analysis" or mode == "发现痛点":
            md = await self._build_pain_point_report(user_input, analysis, framework)
        elif "feature_design" in analysis_type or mode == "功能设计辅助":
            md = await self._build_feature_design_report(user_input, analysis, framework)
        else:
            md = await self._build_comparison_report(user_input, analysis, framework)

        title = self._generate_title(user_input, analysis_type, framework)

        return FinalReport(
            title=title,
            markdown_content=md,
            output_path="report.md",
        )

    def _generate_title(self, user_input: UserInput, analysis_type: str, framework: dict = None) -> str:
        """生成报告标题。"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        framework = framework or {}
        mode = framework.get("mode", "")
        mode_titles = {
            "head_to_head_comparison": "竞品对比分析",
            "competitive_landscape": "竞品发现报告",
            "pain_point_analysis": "痛点分析报告",
        }
        mode_title = mode_titles.get(analysis_type)
        if not mode_title:
            if mode == "找竞品":
                mode_title = "竞品发现报告"
            elif mode == "发现痛点":
                mode_title = "痛点分析报告"
            elif mode == "功能设计辅助":
                mode_title = "功能设计辅助报告"
            elif "feature_design" in analysis_type:
                mode_title = "功能设计辅助报告"
            else:
                mode_title = "竞品分析"
        return f"{user_input.product} - {mode_title} ({date_str})"

    async def _build_comparison_report(
        self, user_input: UserInput, analysis: AnalysisResult, framework: dict
    ) -> str:
        """构建竞品对比分析报告。"""
        competitors = ", ".join(user_input.competitors) if user_input.competitors else "待定"

        # 用 LLM 格式化各章节
        features_section = await self._format_section(
            "竞品对比分析", "功能对比", analysis.comparison_table.get("features", "")
        )
        business_section = await self._format_section(
            "竞品对比分析", "商业表现对比", analysis.comparison_table.get("business", "")
        )
        feedback_section = await self._format_section(
            "竞品对比分析", "用户口碑对比", analysis.comparison_table.get("user_feedback", "")
        )

        sections = [
            f"# {user_input.product} 竞品对比分析报告",
            f"",
            f"**生成日期:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**分析对象:** {user_input.product}",
            f"**对比竞品:** {competitors}",
            f"",
            f"## 1. 功能对比",
            f"",
            features_section,
            f"",
            f"## 2. 商业表现对比",
            f"",
            business_section,
            f"",
            f"## 3. 用户口碑对比",
            f"",
            feedback_section,
            f"",
            f"## 4. 优劣势总结",
            f"",
            self._format_strengths_weaknesses(analysis.strengths_weaknesses),
            f"",
            f"## 5. 建议",
            f"",
            self._format_list(analysis.recommendations),
        ]
        return "\n".join(sections)

    async def _build_landscape_report(
        self, user_input: UserInput, analysis: AnalysisResult, framework: dict
    ) -> str:
        """构建竞品发现报告。"""
        sections = [
            f"# {user_input.product} 竞品发现报告",
            f"",
            f"**生成日期:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**分析对象:** {user_input.product}",
            f"",
            f"## 1. 主要发现",
            f"",
            self._format_list(analysis.key_findings),
            f"",
            f"## 2. 竞品对比矩阵",
            f"",
            self._format_strengths_weaknesses(analysis.strengths_weaknesses),
            f"",
            f"## 3. 建议",
            f"",
            self._format_list(analysis.recommendations),
        ]
        return "\n".join(sections)

    async def _build_pain_point_report(
        self, user_input: UserInput, analysis: AnalysisResult, framework: dict
    ) -> str:
        """构建痛点分析报告。"""
        sections = [
            f"# {user_input.product} 痛点分析报告",
            f"",
            f"**生成日期:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**分析对象:** {user_input.product}",
            f"",
            f"## 1. 核心痛点",
            f"",
            self._format_list(analysis.pain_points),
            f"",
            f"## 2. 详细分析",
            f"",
            self._format_list(analysis.key_findings),
            f"",
            f"## 3. 机会与建议",
            f"",
            self._format_list(analysis.recommendations),
        ]
        return "\n".join(sections)

    async def _build_feature_design_report(
        self, user_input: UserInput, analysis: AnalysisResult, framework: dict
    ) -> str:
        """构建功能设计辅助报告。"""
        sections = [
            f"# {user_input.product} - 功能设计辅助报告",
            f"",
            f"**生成日期:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**功能/场景:** {user_input.product}",
            f"",
            f"## 1. 已有方案分析",
            f"",
            f"_市面上类似产品是如何实现的？_",
            f"",
            self._format_list(analysis.key_findings),
            f"",
            f"## 2. 用户痛点",
            f"",
            f"_用户对现有方案有哪些不满？_",
            f"",
            self._format_list(analysis.pain_points),
            f"",
            f"## 3. 差异化机会",
            f"",
            self._format_strengths_weaknesses(analysis.strengths_weaknesses),
            f"",
            f"## 4. 功能取舍建议",
            f"",
            self._format_list(analysis.recommendations),
        ]
        return "\n".join(sections)

    async def _format_section(self, report_type: str, section_title: str, data: str) -> str:
        """用 LLM 格式化单个章节内容（non-thinking 模式）。"""
        if not data:
            return "_暂无数据_"

        try:
            result = await call_llm(
                system_prompt="你是一个专业的竞品分析报告写作助手。",
                user_prompt=SECTION_PROMPT
                    .replace("{report_type}", report_type)
                    .replace("{section_title}", section_title)
                    .replace("{data}", data),
                thinking=False,
            )
            return result
        except Exception:
            return data  # LLM 失败时直接返回原始数据

    @staticmethod
    def _format_list(items: list[str]) -> str:
        """将字符串列表格式化为 Markdown 无序列表。"""
        if not items:
            return "_暂无数据_"
        return "\n".join(f"- {item}" for item in items)

    @staticmethod
    def _format_strengths_weaknesses(sw: dict) -> str:
        """将优劣势字典格式化为 Markdown。"""
        if not sw:
            return "_暂无数据_"
        lines = []
        for product, data in sw.items():
            lines.append(f"### {product}")
            lines.append(f"- **优势:** {', '.join(data.get('strengths', []))}")
            lines.append(f"- **劣势:** {', '.join(data.get('weaknesses', []))}")
            lines.append("")
        return "\n".join(lines)
