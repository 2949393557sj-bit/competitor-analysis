# ============================================================
# tools 包初始化
# 统一导出所有数据采集工具，供调研 Agent 调用
# ============================================================

from .tavily_search import tavily_search
from .app_store import fetch_app_store_reviews, fetch_social_mentions
from .google_search import google_search

__all__ = [
    "tavily_search",              # Tavily 网页搜索（消耗配额）
    "fetch_app_store_reviews",    # 应用商店评论抓取
    "fetch_social_mentions",      # 社交媒体提及抓取
    "google_search",              # Google 搜索（免费，用于上下文补充）
]
