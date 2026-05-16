# 竞品分析系统 — 设计文档

> 本文档深度描述项目架构、每个模块的设计决策、数据流向和技术细节。
> 每次代码修改后同步更新本文档。

---

## 目录

1. [系统概述](#1-系统概述)
2. [架构总览](#2-架构总览)
3. [LLM 分层策略](#3-llm-分层策略)
4. [数据模型](#4-数据模型)
5. [流水线详解](#5-流水线详解)
   - 5.1 [Orchestrator — 编排 Agent](#51-orchestrator--编排-agent)
   - 5.2 [Research Agents — 调研 Agent x3](#52-research-agents--调研-agent-x3)
   - 5.3 [Analyst — 分析 Agent](#53-analyst--分析-agent)
   - 5.4 [Writer — 写作 Agent](#54-writer--写作-agent)
6. [工具层](#6-工具层)
7. [配置与环境变量](#7-配置与环境变量)
8. [四种分析模式](#8-四种分析模式)
9. [文件结构](#9-文件结构)

---

## 1. 系统概述

这是一个基于 **多 Agent 架构** 的竞品分析系统。用户输入一句话（如 "Slack vs Teams"），系统自动完成：

```
用户输入 → 意图识别 → 数据采集 → 深度分析 → 报告生成
```

核心设计原则：
- **Agent 职责单一**：每个 Agent 只做一件事，通过流水线串联
- **并行执行**：三个调研 Agent 互不依赖，用 `asyncio.gather` 并行
- **LLM 分层调用**：按任务复杂度决定是否开启 thinking 模式，平衡质量与成本
- **结构化输出**：Agent 之间通过 JSON 传递结构化数据，而非自由文本

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────────┐
│                        用户输入                              │
│                   "Slack vs Teams"                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Orchestrator Agent (thinking=True)              │
│                                                             │
│  三步合一（一次 LLM 调用）：                                  │
│    1. 意图识别：判断四选一模式                                 │
│    2. 框架选择：确定分析策略                                   │
│    3. 任务分配：为每个 Research Agent 生成搜索任务              │
│                                                             │
│  特殊流程（找竞品模式）：                                      │
│    LLM → Tavily 搜索 → LLM（两轮调用）                       │
└──────┬──────────────┬──────────────┬────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────────────┐
│  Agent A   │ │  Agent B   │ │     Agent C        │
│ 功能特性   │ │ 商业数据   │ │    用户反馈        │
│(thinking=  │ │(thinking=  │ │  (thinking=        │
│  False)    │ │  False)    │ │    False)          │
│            │ │            │ │                    │
│ Tavily 搜  │ │ Tavily 搜  │ │ 应用商店评论       │
│ → LLM 摘要 │ │ → LLM 摘要 │ │ 社交媒体 + Tavily  │
└──────┬─────┘ └──────┬─────┘ └────────┬───────────┘
       │              │                │
       └──────────────┼────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│               Analyst Agent (thinking=True)                  │
│                                                             │
│  根据分析类型选择提示词，用 LLM 做深度分析                     │
│  输出：关键发现、对比表、优劣势、痛点、建议                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│               Writer Agent (thinking=False)                  │
│                                                             │
│  根据分析类型选择报告模板                                      │
│  用 LLM 格式化各章节内容，组装完整 Markdown 报告               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                    output/report.md
```

---

## 3. LLM 分层策略

所有 Agent 共用同一个模型（DeepSeek-V4-Flash），通过 `thinking` 开关区分推理深度：

| Agent | thinking | 原因 |
|-------|:--------:|------|
| **Orchestrator** | `True` | 意图识别 + 框架选择 + 任务分配需要强推理，多花 token 值得 |
| **Research x3** | `False` | 任务简单——就是整理搜索结果，不需要深度推理 |
| **Analyst** | `True` | 核心分析环节，质量最重要 |
| **Writer** | `False` | 格式化输出，不需要推理 |

实现方式（`llm_client.py`）：

```python
# thinking=True  → extra_body={"thinking": {"type": "enabled"}}
# thinking=False → extra_body={"thinking": {"type": "disabled"}}
```

调用接口：
- `call_llm()` — 返回纯文本（Writer、Research 用）
- `call_llm_json()` — 强制 JSON 输出 + 自动解析（Orchestrator、Analyst 用）

---

## 4. 数据模型

定义在 `models.py`，使用 Python `dataclass`。

### 4.1 输入模型

```
AnalysisMode (枚举)
├── COMPARE           "指定竞品对比"
├── FIND_COMPETITORS  "找竞品"
├── PAIN_POINTS       "发现痛点"
└── FEATURE_DESIGN    "功能设计辅助"

UserInput
├── product: str              目标产品名
├── mode: AnalysisMode        分析模式
├── competitors: list[str]    竞品列表
└── extra_context: str        原始输入文本
```

### 4.2 调研模型（三个 Agent 各自产出）

```
FeatureReport (Agent A)
├── agent_name, search_queries, raw_results
└── summary (LLM 摘要)

BusinessReport (Agent B)
├── agent_name, search_queries, raw_results
└── summary

UserFeedbackReport (Agent C)
├── agent_name, search_queries
├── app_store_reviews       应用商店评论
├── social_media_mentions   社交媒体提及
└── summary
```

### 4.3 分析模型

```
ResearchResults — 三路调研汇总
├── features: FeatureReport
├── business: BusinessReport
└── user_feedback: UserFeedbackReport

AnalysisResult — 分析 Agent 产出
├── key_findings: list[str]         核心发现
├── comparison_table: dict          对比表格
├── strengths_weaknesses: dict      各产品优劣势
├── pain_points: list[str]          痛点列表
├── recommendations: list[str]      建议列表
└── raw_analysis: str               原始分析文本
```

### 4.4 输出模型

```
FinalReport
├── title: str              报告标题
├── markdown_content: str   完整 Markdown 内容
└── output_path: str        输出文件路径
```

### 4.5 数据流向

```
UserInput ──→ Orchestrator ──→ framework: dict + tasks: list[dict]
                                      │
                                      ▼
                              Research Agents (x3)
                                      │
                                      ▼
                              ResearchResults
                                      │
                                      ▼
                               AnalysisResult
                                      │
                                      ▼
                                FinalReport
```

---

## 5. 流水线详解

### 5.1 Orchestrator — 编排 Agent

**文件**：`agents/orchestrator.py`
**LLM 模式**：thinking=True

#### 三步合一设计

传统做法是三步串行调用 LLM（意图识别 → 框架选择 → 任务分配），本系统将三步合并为**一次 LLM 调用**，通过精心设计的提示词让 LLM 在一次推理中完成全部工作。

**优势**：
- 减少 LLM 调用次数（1 次 vs 3 次），降低延迟和成本
- LLM 在一次推理中能看到完整上下文，决策更连贯

**提示词结构**（`INTENT_AND_TASK_PROMPT`）：
```
第一步：意图识别 → 四选一模式
第二步：分析框架 → 确定 analysis_type 和 research_focus
第三步：任务分配 → 为三个 Agent 生成 search_queries、objective、focus_areas
输出：完整 JSON
```

#### 找竞品模式的两轮调用

当意图识别为 `FIND_COMPETITORS` 时，系统需要先知道竞品是谁才能分配任务，因此走特殊流程：

```
第一轮 LLM（thinking）
  ↓ 识别意图，提取产品名
Tavily 搜索（3 个查询并行）
  ↓ 获取竞品候选列表
第二轮 LLM（thinking）
  ↓ 从搜索结果中筛选竞品，生成调研任务
```

**为什么找竞品需要两轮？**
- 第一轮只拿到产品名，不知道竞品是谁
- 需要 Tavily 搜索补充外部信息
- 第二轮 LLM 基于搜索结果做筛选决策

**其他模式为什么不需要？**
- COMPARE：用户已在输入中给出竞品名
- PAIN_POINTS / FEATURE_DESIGN：分析对象就是用户指定的产品，不需要先找竞品

#### 旧接口兼容

保留了 `parse_input()` 和 `select_framework()` 两个同步方法，作为 LLM 不可用时的 fallback（基于关键词匹配）。

---

### 5.2 Research Agents — 调研 Agent x3

**文件**：`agents/researcher.py`
**LLM 模式**：thinking=False

三个 Agent 互不依赖，并行执行：

| Agent | 职责 | 数据来源 |
|-------|------|----------|
| Agent A | 功能特性 | Tavily 搜索 |
| Agent B | 商业数据 | Tavily 搜索 |
| Agent C | 用户反馈 | 应用商店 + 社交媒体 + Tavily |

#### 任务接收机制

每个 Agent 的 `research()` 方法接受两个参数：
- `framework`：分析框架（包含 product、competitors、research_focus）
- `task`：Orchestrator 分配的结构化任务（包含 search_queries、objective、focus_areas）

**优先级**：有 task 则用 task 的 search_queries，否则自行生成。

#### Agent C 的特殊设计

Agent C 除了 Tavily 搜索，还会：
1. 抓取应用商店评论（`fetch_app_store_reviews`）
2. 搜索社交媒体提及（`fetch_social_mentions`）
3. 结果分拣：偶数索引 = 评论，奇数索引 = 社交媒体

#### LLM 摘要

每个 Agent 在收集完原始数据后，用 LLM（non-thinking）将搜索结果综合成一段结构化摘要。这个摘要是 Analyst Agent 的输入。

---

### 5.3 Analyst — 分析 Agent

**文件**：`agents/analyst.py`
**LLM 模式**：thinking=True

#### 分析类型路由

根据 `framework["analysis_type"]` 选择不同的提示词：

| analysis_type | 提示词 | 分析重点 |
|---------------|--------|----------|
| `head_to_head_comparison` | `COMPARISON_ANALYSIS_PROMPT` | 逐项对比、优劣势 |
| `competitive_landscape` | `LANDSCAPE_ANALYSIS_PROMPT` | 竞品发现、市场定位 |
| `pain_point_analysis` | `PAIN_POINT_ANALYSIS_PROMPT` | 痛点提取、机会分析 |
| `feature_design_assistance` | `FEATURE_DESIGN_PROMPT` | 已有方案、差异化机会 |

#### 提示词设计

每个提示词都：
1. 设定角色（竞品分析专家 / 痛点分析专家 / 功能设计顾问）
2. 注入三路调研摘要（features、business、user_feedback）
3. 指定 JSON 输出格式（key_findings、comparison_table、pain_points 等）

#### 降级处理

LLM 调用失败时，`_fallback_result()` 直接将调研摘要作为 key_findings 返回，保证系统不会崩溃。

---

### 5.4 Writer — 写作 Agent

**文件**：`agents/writer.py`
**LLM 模式**：thinking=False

#### 报告模板

四种分析类型对应四种报告结构：

**竞品对比报告**：
```
1. 功能对比
2. 商业表现对比
3. 用户口碑对比
4. 优劣势总结
5. 建议
```

**竞品发现报告**：
```
1. 主要发现
2. 竞品对比矩阵
3. 建议
```

**痛点分析报告**：
```
1. 核心痛点
2. 详细分析
3. 机会与建议
```

**功能设计辅助报告**：
```
1. 已有方案分析（别人怎么做的）
2. 用户痛点（哪里做得不好）
3. 差异化机会（从哪切入）
4. 功能取舍建议
```

#### LLM 章节格式化

对于竞品对比报告的功能/商业/口碑三个章节，Writer 会调用 LLM（non-thinking）将 Analyst 的结构化数据转化为流畅的 Markdown 文段。LLM 失败时降级为原始数据。

其他章节（优劣势、建议等）直接使用 `_format_list()` 和 `_format_strengths_weaknesses()` 静态格式化。

---

## 6. 工具层

**目录**：`tools/`

### 6.1 Tavily 搜索（`tools/tavily_search.py`）

Tavily 是专为 AI 应用设计的搜索引擎 API，返回结构化结果（title、url、content、score）。

**参数**：
- `query`：搜索查询
- `search_depth`：basic（快）/ advanced（全面）
- `max_results`：结果数量上限
- `include_domains` / `exclude_domains`：域名过滤

**当前状态**：占位实现，返回模拟数据。接入真实 API 后取消注释即可。

### 6.2 应用商店评论（`tools/app_store.py`）

两个函数：
- `fetch_app_store_reviews()`：从 Google Play / App Store 抓取评论
- `fetch_social_mentions()`：通过 Tavily 的 `site:` 查询搜索社交媒体讨论

**当前状态**：占位实现。真实抓取逻辑以注释形式保留。

---

## 7. 配置与环境变量

**文件**：`config.py`

所有配置通过环境变量注入，不硬编码：

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `TAVILY_API_KEY` | Tavily 搜索 API 密钥 | 空 |
| `LLM_API_KEY` | DeepSeek API 密钥 | 空 |
| `LLM_MODEL` | 模型名称 | `deepseek-v4-flash` |
| `LLM_BASE_URL` | API 基础地址 | `https://api.deepseek.com` |

使用方式：
```bash
cp .env.example .env
# 编辑 .env 填入真实密钥
```

---

## 8. 四种分析模式

### COMPARE — 指定竞品对比

**触发词**：对比、compare、vs、versus
**场景**："Slack vs Teams"、"Notion和飞书对比"
**流程**：一步到位（LLM → 任务分配 → 调研 → 分析 → 报告）

### FIND_COMPETITORS — 找竞品

**触发词**：竞品、competitor、alternative、替代、找
**场景**："找Notion的竞品"、"有什么替代Slack的"
**流程**：两轮调用（LLM → Tavily 搜竞品 → LLM 整理 → 调研 → 分析 → 报告）

### PAIN_POINTS — 发现痛点

**触发词**：痛点、pain、complaint、问题、不足
**场景**："分析Zoom的用户痛点"、"Slack有什么不足"
**流程**：一步到位

### FEATURE_DESIGN — 功能设计辅助

**触发词**：怎么做、如何做、功能设计、设计方案、实现方案
**场景**："怎么做协作白板功能"、"如何实现消息已读回执"
**流程**：一步到位
**特殊价值**：不只找竞品，更关注"别人怎么做的"和"用户哪里不满意"，辅助功能取舍决策

---

## 9. 文件结构

```
competitor-analysis/
├── config.py                  # 全局配置（API 密钥、模型、输出路径）
├── models.py                  # 数据模型（dataclass 定义）
├── llm_client.py              # LLM 调用客户端（OpenAI 兼容 SDK）
├── main.py                    # 程序入口（流水线编排）
├── requirements.txt           # Python 依赖
├── .env.example               # 环境变量模板
├── .gitignore                 # Git 忽略规则
├── DESIGN.md                  # 本文档
│
├── agents/                    # Agent 层
│   ├── __init__.py            # 包导出
│   ├── orchestrator.py        # 编排 Agent（thinking）
│   ├── researcher.py          # 调研 Agent x3（non-thinking）
│   ├── analyst.py             # 分析 Agent（thinking）
│   └── writer.py              # 写作 Agent（non-thinking）
│
├── tools/                     # 工具层
│   ├── __init__.py            # 包导出
│   ├── tavily_search.py       # Tavily 搜索封装
│   └── app_store.py           # 应用商店评论抓取
│
└── output/                    # 输出目录（gitignore）
    └── report.md              # 生成的分析报告
```
