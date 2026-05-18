"""免费搜索工具（无需 API Key）

用于 Orchestrator 的上下文补充（搜索未知术语），
不消耗 Tavily 配额。

使用 DuckDuckGo 搜索（免费、稳定、无需 API Key）。
"""

from ddgs import DDGS


def google_search(query: str, num_results: int = 5) -> list[dict]:
    """使用 DuckDuckGo 搜索获取结果。

    Args:
        query: 搜索查询
        num_results: 返回结果数量

    Returns:
        搜索结果列表，每个包含 title、url、content
    """
    results = []
    try:
        for r in DDGS().text(query, max_results=num_results):
            results.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "content": r.get("body", ""),
            })
    except Exception as e:
        results.append({"title": "搜索失败", "url": "", "content": str(e)})
    return results
