"""应用商店评论抓取工具

支持从 Google Play 和 App Store 获取用户评论，
以及通过 Tavily 搜索获取社交媒体上的产品讨论。

当前为占位实现，真实抓取逻辑以注释形式保留在函数体内。
"""

from typing import Optional


async def fetch_app_store_reviews(
    app_name: str,
    platform: str = "google_play",
    count: int = 50,
    sort_by: str = "newest",
    country: str = "us",
    lang: str = "en",
) -> list[dict]:
    """从应用商店抓取用户评论。

    Args:
        app_name: 应用名称或包名标识符（如 "com.slack"）
        platform: 平台类型，"google_play" 或 "app_store"
        count:    获取评论数量
        sort_by:  排序方式，"newest"（最新）/ "rating"（评分）/ "relevance"（相关性）
        country:  国家代码（如 "us"、"cn"）
        lang:     语言代码（如 "en"、"zh"）

    Returns:
        评论列表，每条评论包含：
          - text:    评论正文
          - rating:  评分（1-5）
          - date:    评论日期
          - author:  评论者昵称
          - title:   评论标题
    """
    # ---- Google Play 真实抓取逻辑 ----
    # from google_play_scraper import Sort, reviews as gp_reviews
    # sort_map = {"newest": Sort.NEWEST, "rating": Sort.RATING}
    # result, _ = gp_reviews(
    #     app_name, lang=lang, country=country,
    #     sort=sort_map.get(sort_by, Sort.NEWEST),
    #     count=count,
    # )
    # return [
    #     {
    #         "text": r["content"],
    #         "rating": r["score"],
    #         "date": str(r["at"]),
    #         "author": r["userName"],
    #         "title": r.get("title", ""),
    #     }
    #     for r in result
    # ]

    # ---- App Store 真实抓取逻辑 ----
    # from app_store_scraper import AppStore
    # store = AppStore(country=country, app_name=app_name)
    # store.review(how_many=count)
    # return [
    #     {
    #         "text": r["review"],
    #         "rating": r["rating"],
    #         "date": str(r["date"]),
    #         "author": r["userName"],
    #         "title": r.get("title", ""),
    #     }
    #     for r in store.reviews
    # ]

    # ---- 占位实现：返回模拟评论数据 ----
    return [
        {
            "text": f"[PLACEHOLDER] Review for {app_name} on {platform}",
            "rating": 4,
            "date": "2026-01-01",
            "author": "placeholder_user",
            "title": "Placeholder Review",
        }
    ]


async def fetch_social_mentions(
    product_name: str,
    platforms: Optional[list[str]] = None,
    count: int = 20,
) -> list[dict]:
    """通过 Tavily 搜索获取社交媒体上的产品讨论和提及。

    通过构造 site: 限定查询，从 Reddit、Twitter 等平台
    搜索用户对产品的评价和讨论。

    Args:
        product_name: 要搜索的产品名称
        platforms:    目标平台列表，默认 ["reddit", "twitter"]
        count:        获取提及数量上限

    Returns:
        提及列表，每条包含：
          - text:      提及内容
          - source:    来源平台（reddit/twitter 等）
          - url:       原文链接
          - date:      发布日期
          - sentiment: 情感倾向（positive/negative/neutral）
    """
    # ---- 真实实现思路 ----
    # 使用 Tavily 的 site: 查询语法限定搜索范围：
    #   "site:reddit.com {product_name} review"
    #   "site:twitter.com {product_name}"

    from tools.tavily_search import tavily_search

    platforms = platforms or ["reddit", "twitter"]
    results = []

    # 为每个平台构造限定域名的搜索查询
    for platform in platforms:
        query = f"{product_name} review opinion site:{platform}.com"
        search_results = await tavily_search(query=query, max_results=count // len(platforms))
        for r in search_results:
            results.append(
                {
                    "text": r.get("content", ""),
                    "source": platform,
                    "url": r.get("url", ""),
                    "date": "",
                    "sentiment": "neutral",  # 情感分析待接入 LLM
                }
            )
    return results
