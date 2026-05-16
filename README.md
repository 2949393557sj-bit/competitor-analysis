# 竞品分析系统

多 Agent 架构的自动化竞品分析工具。输入一句话，自动完成意图识别 → 数据采集 → 深度分析 → 报告生成。

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API 密钥

```bash
# 复制环境变量模板
cp .env.example .env
```

然后编辑 `.env`，填入你的真实密钥：

```
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxx
LLM_API_KEY=sk-xxxxxxxxxxxxxxx
```

> **安全提示**：`.env` 已被 `.gitignore` 排除，不会被提交到 GitHub。

### 3. 运行

```bash
python main.py "Slack vs Teams"
python main.py "找Notion的竞品"
python main.py "分析Zoom的用户痛点"
python main.py "怎么做协作白板功能"
```

报告输出到 `output/report.md`。

## API 密钥申请

| 密钥 | 用途 | 申请地址 |
|------|------|----------|
| `TAVILY_API_KEY` | 网页搜索 | https://tavily.com/ |
| `LLM_API_KEY` | 大模型推理 | https://platform.deepseek.com/ |

`LLM_MODEL` 和 `LLM_BASE_URL` 已有默认值，一般不需要改。
