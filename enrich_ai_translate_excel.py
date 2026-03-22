#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


def _extract_output_text(resp: Any) -> str:
    text = getattr(resp, "output_text", "") or ""
    if text:
        return text.strip()
    try:
        parts = []
        for item in getattr(resp, "output", []) or []:
            for c in getattr(item, "content", []) or []:
                t = getattr(c, "text", "")
                if t:
                    parts.append(t)
        return "\n".join(parts).strip()
    except Exception:
        return ""


def _request_anthropic_messages(
    api_key: str,
    base_url: str,
    model: str,
    user_text: str,
    timeout_sec: int = 45,
) -> str:
    url = base_url.rstrip("/")
    if not url.endswith("/messages"):
        url = f"{url}/messages"

    payload = {
        "model": model,
        "max_tokens": 600,
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

    # Compatibility: sometimes providers return OpenAI-like schema.
    if "choices" in obj:
        try:
            c0 = (obj.get("choices") or [])[0]
            msg = c0.get("message", {}) if isinstance(c0, dict) else {}
            content = msg.get("content", "")
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                parts = []
                for it in content:
                    t = str((it or {}).get("text", "")).strip()
                    if t:
                        parts.append(t)
                return "\n".join(parts).strip()
        except Exception:
            pass

    out = []
    for c in obj.get("content", []) or []:
        t = str(c.get("text", "")).strip()
        if t:
            out.append(t)
    return "\n".join(out).strip()


class AIClient:
    def __init__(self, api_type: str, model: str, api_key: str, base_url: str | None = None):
        self.api_type = api_type
        self.model = model
        self.api_key = api_key
        self.base_url = (base_url or "").strip()
        self.client = None
        if api_type == "openai-responses":
            if OpenAI is None:
                raise RuntimeError("openai package not installed.")
            if self.base_url:
                self.client = OpenAI(api_key=api_key, base_url=self.base_url)
            else:
                self.client = OpenAI(api_key=api_key)

    def ask_json(self, system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.api_type == "anthropic-messages":
            txt = _request_anthropic_messages(
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
                user_text=system_prompt + "\n" + json.dumps(payload, ensure_ascii=False),
            )
        else:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            txt = _extract_output_text(resp)
        return _robust_json_loads(txt)


def _robust_json_loads(txt: str) -> dict[str, Any]:
    s = (txt or "").strip()
    if not s:
        raise ValueError("Empty model response.")

    # Strip fenced blocks
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # Direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Extract first JSON object from noisy text
    m = re.search(r"\{[\s\S]*\}", s)
    if m:
        snippet = m.group(0)
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj

    raise ValueError(f"Non-JSON model response: {s[:220]}")


def _fallback_summary(name: str, sub: str) -> str:
    if not name:
        name = "該公司"
    if sub:
        return f"{name}主要業務可歸類於{sub}。"
    return f"{name}為美股上市公司，主要從事其核心產品與服務相關業務。"


def _needs_translation(text: str) -> bool:
    if not text:
        return False
    latin = len(re.findall(r"[A-Za-z]", text))
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    return latin > 6 and latin > cjk


def enrich_row_with_ai(ai: AIClient, row: pd.Series) -> tuple[str, str, str]:
    payload = {
        "ticker": str(row.get("ticker", "")),
        "exchange": str(row.get("exchange", "")),
        "company_name": str(row.get("security_name", "")),
        "sec_title": str(row.get("sec_title", "")),
        "sic_description": str(row.get("sic_description", "")),
        "major_sector": str(row.get("major_sector", "")),
        "subsector": str(row.get("subsector", "")),
        "task": "輸出JSON：ai_subsector(繁體中文)、ai_small_tags(繁體中文陣列，可多個)、company_summary_zh_tw(繁體中文1-2句)。",
        "output_format": '{"ai_subsector":"...", "ai_small_tags":["..."], "company_summary_zh_tw":"..."}',
    }
    obj = ai.ask_json(
        "你是嚴謹的股票分類助理，只輸出JSON，不要輸出任何其他文字。",
        payload,
    )
    ai_sub = str(obj.get("ai_subsector", "")).strip()
    tags = obj.get("ai_small_tags", [])
    if isinstance(tags, list):
        tags = [str(x).strip() for x in tags if str(x).strip()]
    else:
        tags = [x.strip() for x in str(tags).split(",") if x.strip()]
    ai_tags = "|".join(dict.fromkeys(tags))
    summ = str(obj.get("company_summary_zh_tw", "")).strip()
    return ai_sub, ai_tags, summ


def translate_term(ai: AIClient, term: str, cache: dict[str, str]) -> str:
    term = (term or "").strip()
    if not term:
        return ""
    if term in cache:
        return cache[term]
    if re.search(r"[\u4e00-\u9fff]", term):
        cache[term] = term
        return term
    payload = {
        "text": term,
        "task": "翻譯成繁體中文，盡量使用金融/產業常用名詞。只輸出翻譯結果字串。",
    }
    obj = ai.ask_json(
        "你是財經翻譯助理。輸出JSON，格式：{\"zh_tw\":\"...\"}",
        payload,
    )
    zh = str(obj.get("zh_tw", "")).strip() or term
    cache[term] = zh
    return zh


def run(
    input_path: Path,
    output_path: Path,
    api_type: str,
    model: str,
    api_key: str,
    base_url: str,
    ai_only_missing: bool,
    ai_max_rows: int,
    ai_sleep_sec: float,
    translate_terms: bool,
    translate_summaries: bool,
) -> None:
    df = pd.read_excel(input_path, sheet_name="all_stocks").fillna("")
    ai = AIClient(api_type=api_type, model=model, api_key=api_key, base_url=base_url)

    # Stage A: AI fill missing columns only (or all rows if requested)
    idxs = df.index.tolist()
    if ai_only_missing:
        cond = (
            (df.get("ai_subsector", "").astype(str).str.strip() == "")
            | (df.get("ai_small_tags", "").astype(str).str.strip() == "")
            | (df.get("company_summary_zh_tw", "").astype(str).str.strip() == "")
        )
        idxs = df.index[cond].tolist()
    if ai_max_rows > 0:
        idxs = idxs[:ai_max_rows]

    print(f"[INFO] AI enrich rows={len(idxs)}")
    for n, i in enumerate(idxs, start=1):
        row = df.loc[i]
        try:
            ai_sub, ai_tags, summary = enrich_row_with_ai(ai, row)
            if ai_sub:
                df.at[i, "ai_subsector"] = ai_sub
                df.at[i, "final_subsector"] = ai_sub
            if ai_tags:
                df.at[i, "ai_small_tags"] = ai_tags
            if summary:
                df.at[i, "company_summary_zh_tw"] = summary
            elif not str(df.at[i, "company_summary_zh_tw"]).strip():
                df.at[i, "company_summary_zh_tw"] = _fallback_summary(
                    str(df.at[i, "security_name"]), str(df.at[i, "subsector"])
                )
        except Exception as e:
            if not str(df.at[i, "company_summary_zh_tw"]).strip():
                df.at[i, "company_summary_zh_tw"] = _fallback_summary(
                    str(df.at[i, "security_name"]), str(df.at[i, "subsector"])
                )
            print(f"[WARN] AI enrich fail ticker={row.get('ticker','')} err={e}")
        time.sleep(max(0.0, ai_sleep_sec) + random.uniform(0.05, 0.2))
        if n % 100 == 0:
            print(f"[INFO] AI enrich processed {n}/{len(idxs)}")

    # Stage B: Translate taxonomy terms to zh-tw
    term_cache: dict[str, str] = {}
    if translate_terms:
        for col in ["exchange", "major_sector", "subsector", "ai_subsector", "final_subsector"]:
            if col in df.columns:
                vals = sorted(set(df[col].astype(str).str.strip().tolist()))
                vals = [v for v in vals if v]
                print(f"[INFO] translate terms col={col} unique={len(vals)}")
                for idx_t, v in enumerate(vals, start=1):
                    try:
                        translate_term(ai, v, term_cache)
                    except Exception as e:
                        term_cache[v] = v
                        print(f"[WARN] translate term fail col={col} term={v} err={e}")
                    if idx_t % 25 == 0:
                        print(f"[INFO] translate terms progress col={col} {idx_t}/{len(vals)}")
                    time.sleep(max(0.0, ai_sleep_sec) + random.uniform(0.03, 0.1))
    else:
        print("[INFO] skip term translation stage.")

    def map_term(x: Any) -> str:
        s = str(x).strip()
        if not s:
            return ""
        return term_cache.get(s, s)

    df["exchange_zh_tw"] = df["exchange"].map(map_term) if "exchange" in df.columns else ""
    df["major_sector_zh_tw"] = df["major_sector"].map(map_term) if "major_sector" in df.columns else ""
    df["subsector_zh_tw"] = df["subsector"].map(map_term) if "subsector" in df.columns else ""
    df["ai_subsector_zh_tw"] = df["ai_subsector"].map(map_term) if "ai_subsector" in df.columns else ""
    df["final_subsector_zh_tw"] = df["final_subsector"].map(map_term) if "final_subsector" in df.columns else ""

    # Translate ai_small_tags
    def map_tags(v: Any) -> str:
        s = str(v).strip()
        if not s:
            return ""
        tags = [x.strip() for x in s.split("|") if x.strip()]
        out = [term_cache.get(t, t) for t in tags]
        return "|".join(dict.fromkeys(out))

    if "ai_small_tags" in df.columns:
        if translate_terms:
            # pre-translate tags that were not in term cache
            all_tags = set()
            for s in df["ai_small_tags"].astype(str).tolist():
                for t in s.split("|"):
                    t = t.strip()
                    if t:
                        all_tags.add(t)
            tags_sorted = sorted(all_tags)
            print(f"[INFO] translate tag terms unique={len(tags_sorted)}")
            for idx_t, t in enumerate(tags_sorted, start=1):
                if t in term_cache:
                    continue
                try:
                    translate_term(ai, t, term_cache)
                except Exception:
                    term_cache[t] = t
                if idx_t % 25 == 0:
                    print(f"[INFO] translate tag terms progress {idx_t}/{len(tags_sorted)}")
                time.sleep(max(0.0, ai_sleep_sec) + random.uniform(0.03, 0.1))
            df["ai_small_tags_zh_tw"] = df["ai_small_tags"].map(map_tags)
        else:
            df["ai_small_tags_zh_tw"] = df["ai_small_tags"]
    else:
        df["ai_small_tags_zh_tw"] = ""

    # Summary text fallback translate if still English-heavy
    if "company_summary_zh_tw" in df.columns and translate_summaries:
        idx_sum = [i for i, txt in enumerate(df["company_summary_zh_tw"].astype(str).tolist()) if _needs_translation(txt)]
        print(f"[INFO] translate summary rows={len(idx_sum)}")
        for n, i in enumerate(idx_sum, start=1):
            txt = str(df.at[i, "company_summary_zh_tw"])
            try:
                obj = ai.ask_json(
                    "你是翻譯助理。輸出JSON：{\"zh_tw\":\"...\"}",
                    {"text": txt, "task": "翻譯為繁體中文，保留公司名和專有名詞。"},
                )
                zh = str(obj.get("zh_tw", "")).strip()
                if zh:
                    df.at[i, "company_summary_zh_tw"] = zh
            except Exception:
                pass
            if n % 25 == 0:
                print(f"[INFO] translate summary progress {n}/{len(idx_sum)}")
            time.sleep(max(0.0, ai_sleep_sec) + random.uniform(0.03, 0.1))
    elif not translate_summaries:
        print("[INFO] skip summary translation stage.")

    if "generated_at" not in df.columns:
        df["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_stocks", index=False)
        summary = (
            df.groupby(["major_sector_zh_tw", "final_subsector_zh_tw"], dropna=False)["ticker"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        summary.to_excel(writer, sheet_name="summary", index=False)
        for major, g in df.groupby("major_sector_zh_tw", dropna=False):
            sheet = str(major).strip() or "其他"
            sheet = sheet[:31]
            g.to_excel(writer, sheet_name=sheet, index=False)
    print(f"[DONE] output={output_path} rows={len(df)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI-only enrichment + Traditional Chinese translation for existing Excel.")
    p.add_argument(
        "--in",
        dest="input_path",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors.xlsx",
        help="Input Excel path (must contain all_stocks sheet).",
    )
    p.add_argument(
        "--out",
        dest="output_path",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors_ai_zh_tw.xlsx",
        help="Output Excel path.",
    )
    p.add_argument(
        "--ai-api-type",
        default="anthropic-messages",
        choices=["openai-responses", "anthropic-messages"],
        help="AI API style.",
    )
    p.add_argument("--ai-model", default="kimi-for-coding", help="Model id.")
    p.add_argument("--ai-api-key", default=os.getenv("KIMI_API_KEY", "") or os.getenv("OPENAI_API_KEY", ""))
    p.add_argument("--ai-base-url", default=os.getenv("KIMI_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", ""))
    p.add_argument("--ai-only-missing", action="store_true", help="Only enrich rows with missing AI fields.")
    p.add_argument("--ai-max-rows", type=int, default=1800, help="AI enrich max rows (0 means no limit).")
    p.add_argument("--ai-sleep-sec", type=float, default=0.9, help="Sleep between AI calls.")
    p.add_argument("--skip-translate-terms", action="store_true", help="Skip translating category/tag terms.")
    p.add_argument("--skip-translate-summaries", action="store_true", help="Skip translating summaries.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    key = str(args.ai_api_key).strip()
    if not key:
        raise RuntimeError("Missing AI API key. Set --ai-api-key or KIMI_API_KEY / OPENAI_API_KEY.")
    run(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        api_type=str(args.ai_api_type),
        model=str(args.ai_model),
        api_key=key,
        base_url=str(args.ai_base_url),
        ai_only_missing=bool(args.ai_only_missing),
        ai_max_rows=int(args.ai_max_rows),
        ai_sleep_sec=max(0.0, float(args.ai_sleep_sec)),
        translate_terms=not bool(args.skip_translate_terms),
        translate_summaries=not bool(args.skip_translate_summaries),
    )


if __name__ == "__main__":
    main()
