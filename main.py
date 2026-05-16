"""竞品分析系统 - 程序入口

多 Agent 架构的竞品分析系统，流水线：
  1. Orchestrator（编排 Agent）→ 意图识别 + 框架选择 + 任务分配（thinking 模式）
  2. Research Agents（调研 Agent x3）→ 并行搜索 + 整理结果（non-thinking 模式）
  3. Analyst（分析 Agent）→ 深度分析调研数据（thinking 模式）
  4. Writer（写作 Agent）→ 生成 Markdown 报告（non-thinking 模式）

使用示例：
    python main.py "Slack vs Teams"
    python main.py "找Notion的竞品"
    python main.py "分析Zoom的用户痛点"
    python main.py "怎么做协作白板功能"
"""

import asyncio
import sys
import os

from agents.orchestrator import OrchestratorAgent
from agents.researcher import run_research_agents
from agents.analyst import AnalysisAgent
from agents.writer import WriterAgent


async def run_analysis(user_input_text: str) -> str:
    """执行完整的分析流水线。

    Args:
        user_input_text: 用户的原始查询文本

    Returns:
        生成的 Markdown 报告文件路径
    """
    # ---- 第 1 步：编排 Agent（thinking 模式）----
    # 一次 LLM 调用完成：意图识别 → 框架选择 → 任务分配
    # 找竞品模式会自动触发两轮调用（LLM → Tavily → LLM）
    orchestrator = OrchestratorAgent()
    result = await orchestrator.arun(user_input_text)
    user_input = result["user_input"]
    framework = result["framework"]
    tasks = result["tasks"]

    print(f"[Orchestrator] 分析模式: {user_input.mode.value}")
    print(f"[Orchestrator] 目标产品: {user_input.product}")
    print(f"[Orchestrator] 竞品列表: {user_input.competitors}")
    print(f"[Orchestrator] 分析框架: {framework.get('analysis_type', 'N/A')}")
    print(f"[Orchestrator] 分配任务: {len(tasks)} 个")
    print()

    # ---- 第 2 步：三个调研 Agent 并行执行（non-thinking 模式）----
    # 接收 Orchestrator 分配的结构化任务，各自搜索并整理结果
    print("[Research] 启动并行调研 Agent...")
    research_results = await run_research_agents(framework, tasks)

    print(f"[Research] Agent A (功能特性): {len(research_results.features.raw_results)} 条结果")
    print(f"[Research] Agent B (商业数据): {len(research_results.business.raw_results)} 条结果")
    print(f"[Research] Agent C (用户反馈): {len(research_results.user_feedback.app_store_reviews)} 条评论, {len(research_results.user_feedback.social_media_mentions)} 条社交提及")
    print()

    # ---- 第 3 步：分析 Agent（thinking 模式）----
    # 对调研数据进行深度分析，提取关键发现、痛点、建议
    analyst = AnalysisAgent()
    analysis = await analyst.analyze(research_results, framework)
    print(f"[Analysis] 关键发现: {len(analysis.key_findings)} 条")
    print(f"[Analysis] 用户痛点: {len(analysis.pain_points)} 条")
    print()

    # ---- 第 4 步：写作 Agent（non-thinking 模式）----
    # 将结构化分析结果组装成 Markdown 报告
    writer = WriterAgent()
    report = await writer.write(user_input, analysis, framework)

    # 确保输出目录存在，写入报告文件
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "report.md")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report.markdown_content)

    print(f"[Writer] 报告已生成: {output_path}")
    print(f"[Writer] 报告标题: {report.title}")
    return output_path


def main():
    """命令行入口函数。校验参数并启动分析流水线。"""
    if len(sys.argv) < 2:
        print("用法: python main.py <查询内容>")
        print()
        print("示例:")
        print('  python main.py "Slack vs Teams"')
        print('  python main.py "找Notion的竞品"')
        print('  python main.py "分析Zoom的用户痛点"')
        print('  python main.py "怎么做协作白板功能"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"=== 竞品分析系统 ===")
    print(f"查询: {query}")
    print("=" * 40)
    print()

    output = asyncio.run(run_analysis(query))
    print()
    print(f"完成！报告已保存至: {output}")


if __name__ == "__main__":
    main()
