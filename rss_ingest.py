from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser


DEFAULT_RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "https://www.investing.com/rss/news.rss",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
]


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _parse_dt(entry: dict[str, Any]) -> datetime:
    for k in ["published", "updated", "created"]:
        val = entry.get(k)
        if val:
            try:
                dt = parsedate_to_datetime(str(val))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)


def _id_of(title: str, link: str) -> str:
    return hashlib.sha256(f"{title}|{link}".encode("utf-8")).hexdigest()[:16]


@dataclass
class RssItem:
    id: str
    source: str
    title: str
    summary: str
    link: str
    published_at: str
    matched_keywords: list[str]


def fetch_rss_items(
    keywords: list[str],
    feeds: list[str] | None = None,
    lookback_hours: int = 72,
    max_items: int = 80,
) -> list[RssItem]:
    feeds = feeds or DEFAULT_RSS_FEEDS
    kws = [k.strip().lower() for k in keywords if k and k.strip()]
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - int(lookback_hours) * 3600

    out: list[RssItem] = []
    seen: set[str] = set()
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
        except Exception:
            continue

        source = _normalize_text(parsed.feed.get("title", "")) or feed_url
        for entry in parsed.entries:
            title = _normalize_text(entry.get("title", ""))
            summary = _normalize_text(entry.get("summary", "") or entry.get("description", ""))
            link = _normalize_text(entry.get("link", ""))
            if not title:
                continue
            dt = _parse_dt(entry)
            if dt.timestamp() < cutoff:
                continue
            blob = f"{title} {summary}".lower()
            matched = [k for k in kws if k in blob]
            if kws and not matched:
                continue
            uid = _id_of(title, link)
            if uid in seen:
                continue
            seen.add(uid)
            out.append(
                RssItem(
                    id=uid,
                    source=source,
                    title=title,
                    summary=summary[:1200],
                    link=link,
                    published_at=dt.strftime("%Y-%m-%d %H:%M:%S"),
                    matched_keywords=matched,
                )
            )

    out.sort(key=lambda x: x.published_at, reverse=True)
    return out[: max(1, int(max_items))]

