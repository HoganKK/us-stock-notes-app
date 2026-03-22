from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class EventThemeResult:
    themes: list[str]
    hits: list[dict[str, Any]]


def _request_anthropic_messages(
    api_key: str,
    base_url: str,
    model: str,
    user_text: str,
    timeout_sec: int = 70,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/messages"):
        url = f"{url}/messages"
    payload = {
        "model": model,
        "max_tokens": 1800,
        "messages": [{"role": "user", "content": user_text}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "Authorization": f"Bearer {api_key}",
        "anthropic-version": "2023-06-01",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    r.raise_for_status()
    obj = r.json()
    if "choices" in obj:
        c0 = (obj.get("choices") or [{}])[0]
        msg = c0.get("message", {})
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            out = [str(x.get("text", "")).strip() for x in content if str(x.get("text", "")).strip()]
            return "\n".join(out)
    out = []
    for c in obj.get("content", []) or []:
        t = str(c.get("text", "")).strip()
        if t:
            out.append(t)
    return "\n".join(out)


def _robust_json_loads(txt: str) -> dict[str, Any]:
    s = (txt or "").strip()
    if not s:
        raise ValueError("Empty AI response")
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            return json.loads(m.group(0))
        raise


def _pick_candidates(event_text: str, universe_rows: list[dict[str, Any]], limit: int = 280) -> list[dict[str, Any]]:
    # Fast heuristic pre-filter so token usage is controlled.
    q_terms = [x for x in re.split(r"[^\w\u4e00-\u9fff]+", event_text.lower()) if len(x) >= 2]
    q_set = set(q_terms)
    scored: list[tuple[int, dict[str, Any]]] = []
    for r in universe_rows:
        blob = " ".join(
            [
                str(r.get("company_name", "")),
                str(r.get("sector", "")),
                str(r.get("subsector", "")),
                str(r.get("tags", "")),
                str(r.get("summary", ""))[:300],
            ]
        ).lower()
        score = 0
        for t in q_set:
            if t in blob:
                score += 1
        if score > 0:
            scored.append((score, r))

    if not scored:
        # fallback: pick first N diversified by subsector occurrence
        return universe_rows[:limit]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _s, r in scored[:limit]]


def classify_event_impact(
    api_key: str,
    base_url: str,
    model: str,
    event_title: str,
    event_detail: str,
    universe_rows: list[dict[str, Any]],
    extra_context: str = "",
) -> EventThemeResult:
    event_text = f"{event_title}\n{event_detail}\n{extra_context}".strip()
    candidates = _pick_candidates(event_text, universe_rows, limit=280)
    payload = {
        "task": "從事件中找出投資主題與受影響股票，輸出繁體中文 JSON。",
        "event_title": event_title,
        "event_detail": event_detail,
        "extra_context": extra_context,
        "candidate_stocks": [
            {
                "ticker": str(r.get("ticker", "")),
                "company_name": str(r.get("company_name", "")),
                "sector": str(r.get("sector", "")),
                "subsector": str(r.get("subsector", "")),
                "tags": str(r.get("tags", "")),
                "summary": str(r.get("summary", ""))[:300],
            }
            for r in candidates
        ],
        "output_format": {
            "themes": ["主題A", "主題B"],
            "hits": [
                {
                    "ticker": "LITE",
                    "theme": "光通訊",
                    "impact": "利多|利空|中性",
                    "confidence": 0.75,
                    "reason": "一句話原因",
                }
            ],
        },
        "rules": [
            "theme 用事件驅動角度，不要只重複產業名稱。",
            "impact 必須是 利多/利空/中性。",
            "confidence 介於 0 到 1。",
            "同一 ticker 可對應多個 theme。",
            "只從 candidate_stocks 中選 ticker。",
            "最多輸出 120 條 hits，優先高關聯。",
        ],
    }
    user_text = (
        "你是股票主題分析助理。只輸出 JSON，不要任何額外文字。\n"
        + json.dumps(payload, ensure_ascii=False)
    )
    txt = _request_anthropic_messages(api_key=api_key, base_url=base_url, model=model, user_text=user_text)
    obj = _robust_json_loads(txt)
    themes = [str(x).strip() for x in (obj.get("themes", []) or []) if str(x).strip()]
    allowed = {"利多", "利空", "中性"}
    hits: list[dict[str, Any]] = []
    candidate_tickers = {str(x.get("ticker", "")).upper() for x in candidates}
    for h in (obj.get("hits", []) or []):
        ticker = str(h.get("ticker", "")).strip().upper()
        if not ticker or ticker not in candidate_tickers:
            continue
        theme = str(h.get("theme", "")).strip()
        impact = str(h.get("impact", "中性")).strip()
        if impact not in allowed:
            impact = "中性"
        try:
            conf = float(h.get("confidence", 0.5) or 0.5)
        except Exception:
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        reason = str(h.get("reason", "")).strip()[:260]
        if not theme:
            continue
        hits.append(
            {
                "ticker": ticker,
                "theme": theme,
                "impact": impact,
                "confidence": conf,
                "reason": reason,
            }
        )
    if not themes:
        themes = sorted({x["theme"] for x in hits})
    return EventThemeResult(themes=themes, hits=hits)

