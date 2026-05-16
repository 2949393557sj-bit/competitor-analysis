"""Tavily 搜索工具封装

Tavily 是一个专为 AI 应用设计的搜索引擎 API，
返回结构化的搜索结果（标题、URL、内容摘要、相关性评分）。

官方文档：https://docs.tavily.com/

当前为占位实现，接入真实 API 后取消注释即可使用。
"""

from typing import Optional


async def tavily_search(
    query: str,
    search_depth: str = "basic",
    max_results: int = 5,
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> list[dict]:
    """使用 Tavily API 进行网页搜索。

    Args:
        query:           搜索查询字符串
        search_depth:    搜索深度，"basic"（快速）或 "advanced"（更全面）
        max_results:     返回结果数量上限
        include_domains: 限定搜索范围到这些域名（如 ["reddit.com"]）
        exclude_domains: 排除这些域名的搜索结果

    Returns:
        搜索结果列表，每个结果包含：
          - title:   页面标题
          - url:     页面链接
          - content: 内容摘要
          - score:   相关性评分（0-1）
    """
    # ---- 真实 Tavily API 调用（取消注释以启用）----
    # from tavily import TavilyClient
    # client = TavilyClient(api_key=TAVILY_API_KEY)
    # response = client.search(
    #     query=query,
    #     search_depth=search_depth,
    #     max_results=max_results,
    #     include_domains=include_domains,
    #     exclude_domains=exclude_domains,
    # )
    # return response.get("results", [])

    # ---- 占位实现：返回模拟数据 ----
    return [
        {
            "title": f"[PLACEHOLDER] Search result for: {query}",
            "url": "https://example.com",
            "content": f"Placeholder content for query: {query}",
            "score": 0.0,
        }
    ]
