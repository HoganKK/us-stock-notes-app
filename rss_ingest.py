from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser


DEFAULT_RSS_FEEDS = [
    # English
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.reuters.com/reuters/worldNews",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/19854910/device/rss/rss.html",
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    # Chinese (supplement)
    "https://rsshub.app/reuters/world-news?lang=zh",
    "https://rsshub.app/reuters/business-news?lang=zh",
    "https://www.zaobao.com/realtime/china/rss.xml",
    "https://www.zaobao.com/realtime/world/rss.xml",
]


# Input keyword expansion dictionary (zh <-> en + variants)
KEYWORD_SYNONYMS = {
    "美伊戰爭": ["iran war", "iran", "u.s.", "us", "middle east conflict", "hormuz", "iran-israel"],
    "以伊戰爭": ["iran war", "iran", "israel", "middle east conflict", "hormuz"],
    "美以伊": ["iran war", "iran", "israel", "u.s.", "middle east conflict", "hormuz"],
    "中東衝突": ["middle east conflict", "iran", "israel", "hormuz", "oil"],
    "霍爾木茲": ["hormuz", "strait of hormuz", "oil shipping", "tanker"],
    "石油": ["oil", "crude", "brent", "wti", "opec"],
    "天然氣": ["natural gas", "lng", "gas futures"],
    "有色金屬": ["copper", "aluminum", "nickel", "zinc", "base metals"],
    "化肥": ["fertilizer", "potash", "urea", "ammonia"],
    "腦機接口": ["brain-computer interface", "bci", "neuralink", "brain chip"],
    "spacex": ["spacex", "starlink", "falcon", "launch services", "space economy"],
    "gtc": ["gtc", "nvidia gtc", "nvidia conference", "gpu conference"],
    "量子": ["quantum", "quantum computing", "qubit", "quantum hardware"],
}


TITLE_TRANSLATE_MAP = {
    "iran war": "美伊/以伊戰爭",
    "middle east conflict": "中東衝突",
    "strait of hormuz": "霍爾木茲海峽",
    "oil": "石油",
    "natural gas": "天然氣",
    "fertilizer": "化肥",
    "brain-computer interface": "腦機接口",
    "quantum": "量子",
    "spacex": "SpaceX",
    "starlink": "星鏈",
    "nvidia": "輝達",
    "data center": "資料中心",
}


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


def expand_keywords(keywords: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    base = [k.strip().lower() for k in keywords if k and k.strip()]
    expanded = set(base)
    mapping: dict[str, list[str]] = {}
    for raw in keywords:
        key = str(raw or "").strip()
        if not key:
            continue
        k = key.lower()
        extra = []
        if key in KEYWORD_SYNONYMS:
            extra.extend(KEYWORD_SYNONYMS[key])
        if k in KEYWORD_SYNONYMS:
            extra.extend(KEYWORD_SYNONYMS[k])
        # fuzzy trigger: if contains known key, append synonyms
        for sk, vals in KEYWORD_SYNONYMS.items():
            if sk.lower() in k or k in sk.lower():
                extra.extend(vals)
        extra_norm = sorted({x.strip().lower() for x in extra if str(x).strip()})
        mapping[key] = extra_norm
        for x in extra_norm:
            expanded.add(x)
    return sorted(expanded), mapping


def _quick_zh_title(title: str) -> str:
    t = str(title or "")
    low = t.lower()
    hits = []
    for k, v in TITLE_TRANSLATE_MAP.items():
        if k in low:
            hits.append(v)
    if not hits:
        return ""
    prefix = "、".join(dict.fromkeys(hits))
    return f"{prefix} | {t}"


@dataclass
class RssItem:
    id: str
    source: str
    title: str
    title_zh_tw: str
    summary: str
    link: str
    published_at: str
    matched_keywords: list[str]
    matched_expanded: list[str]


def fetch_rss_items(
    keywords: list[str],
    feeds: list[str] | None = None,
    lookback_hours: int = 72,
    max_items: int = 80,
) -> list[RssItem]:
    feeds = feeds or DEFAULT_RSS_FEEDS
    kws = [k.strip().lower() for k in keywords if k and k.strip()]
    expanded, _map = expand_keywords(keywords)
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
            zh_title = _quick_zh_title(title)
            blob = f"{title} {summary} {zh_title}".lower()
            matched = [k for k in kws if k in blob]
            matched_expanded = [k for k in expanded if k in blob]
            if kws and not matched and not matched_expanded:
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
                    title_zh_tw=zh_title,
                    summary=summary[:1200],
                    link=link,
                    published_at=dt.strftime("%Y-%m-%d %H:%M:%S"),
                    matched_keywords=matched,
                    matched_expanded=matched_expanded[:10],
                )
            )

    out.sort(key=lambda x: x.published_at, reverse=True)
    return out[: max(1, int(max_items))]

