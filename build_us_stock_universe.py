#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_FMT = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

ADR_PAT = re.compile(
    r"\bADR\b|\bADS\b|AMERICAN DEPOSITARY|DEPOSITARY SHARES|SPONSORED ADS|REPRESENTING",
    flags=re.IGNORECASE,
)


@dataclass
class FetchConfig:
    user_agent: str
    timeout_sec: int = 25
    request_sleep_sec: float = 0.35
    max_retries: int = 4
    backoff_base_sec: float = 1.2


def _request_with_retry(session: requests.Session, url: str, cfg: FetchConfig) -> requests.Response:
    last_err: Exception | None = None
    for attempt in range(1, cfg.max_retries + 1):
        try:
            r = session.get(url, timeout=cfg.timeout_sec)
            if r.status_code in {429, 500, 502, 503, 504}:
                raise requests.HTTPError(f"retryable_status={r.status_code}")
            r.raise_for_status()
            if cfg.request_sleep_sec > 0:
                time.sleep(cfg.request_sleep_sec)
            return r
        except Exception as e:
            last_err = e
            sleep_sec = cfg.backoff_base_sec * (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
            time.sleep(min(12.0, sleep_sec))
    if last_err:
        raise last_err
    raise RuntimeError("HTTP request failed without explicit exception.")


def _http_get_text(session: requests.Session, url: str, cfg: FetchConfig) -> str:
    return _request_with_retry(session, url, cfg).text


def _http_get_json(session: requests.Session, url: str, cfg: FetchConfig) -> Any:
    return _request_with_retry(session, url, cfg).json()


def _read_nasdaq_pipe_table(raw_text: str) -> pd.DataFrame:
    lines = raw_text.splitlines()
    clean_lines = []
    for line in lines:
        if line.startswith("File Creation Time"):
            break
        if line.strip():
            clean_lines.append(line)
    return pd.read_csv(StringIO("\n".join(clean_lines)), sep="|", dtype=str).fillna("")


def load_us_listed_symbols(session: requests.Session, cfg: FetchConfig) -> pd.DataFrame:
    nasdaq_df = _read_nasdaq_pipe_table(_http_get_text(session, NASDAQ_LISTED_URL, cfg))
    other_df = _read_nasdaq_pipe_table(_http_get_text(session, OTHER_LISTED_URL, cfg))

    nasdaq_df = nasdaq_df.rename(columns={"Symbol": "ticker", "Security Name": "security_name"})
    nasdaq_df["exchange"] = "NASDAQ"
    nasdaq_df["is_test"] = nasdaq_df.get("Test Issue", "")
    nasdaq_df["is_etf"] = nasdaq_df.get("ETF", "")

    other_df = other_df.rename(columns={"ACT Symbol": "ticker", "Security Name": "security_name"})
    ex_map = {"N": "NYSE", "P": "NYSE ARCA", "A": "NYSE AMEX", "Z": "BATS", "V": "IEX"}
    other_df["exchange"] = other_df.get("Exchange", "").map(ex_map).fillna(other_df.get("Exchange", ""))
    other_df["is_test"] = other_df.get("Test Issue", "")
    other_df["is_etf"] = other_df.get("ETF", "")

    merged = pd.concat(
        [
            nasdaq_df[["ticker", "security_name", "exchange", "is_test", "is_etf"]],
            other_df[["ticker", "security_name", "exchange", "is_test", "is_etf"]],
        ],
        ignore_index=True,
    ).fillna("")

    merged["ticker"] = merged["ticker"].str.strip().str.upper()
    merged["security_name"] = merged["security_name"].str.strip()
    merged = merged[merged["ticker"] != ""].copy()
    merged = merged[merged["is_test"].str.upper() != "Y"].copy()
    merged = merged[merged["is_etf"].str.upper() != "Y"].copy()
    merged["is_adr_by_name"] = merged["security_name"].str.contains(ADR_PAT, na=False)
    merged = merged[~merged["is_adr_by_name"]].copy()
    merged = merged.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return merged


def load_sec_ticker_map(session: requests.Session, cfg: FetchConfig) -> pd.DataFrame:
    data = _http_get_json(session, SEC_TICKERS_URL, cfg)
    rows = []
    for _, item in data.items():
        ticker = str(item.get("ticker", "")).strip().upper()
        cik = int(item.get("cik_str", 0) or 0)
        title = str(item.get("title", "")).strip()
        if ticker and cik > 0:
            rows.append({"ticker": ticker, "cik": cik, "sec_title": title})
    return pd.DataFrame(rows)


def fetch_sec_company_profile(session: requests.Session, cfg: FetchConfig, cik: int) -> dict[str, Any]:
    url = SEC_SUBMISSIONS_URL_FMT.format(cik=int(cik))
    obj = _http_get_json(session, url, cfg)
    return {
        "sic": obj.get("sic", ""),
        "sic_description": obj.get("sicDescription", "") or "",
        "entity_type": obj.get("entityType", "") or "",
        "fiscal_year_end": obj.get("fiscalYearEnd", "") or "",
    }


def classify_subsector(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("security_name", "")),
            str(row.get("sic_description", "")),
            str(row.get("sec_title", "")),
        ]
    ).lower()

    rules = [
        (r"optical|photon|fiber|transceiver|laser|opto", "Optical Communications"),
        (r"openclaw|cloud ai|ai cloud|ai infra", "AI Infrastructure"),
        (r"ai application|generative ai|copilot|llm", "AI Applications"),
        (r"memory|dram|nand|storage", "Memory / Storage"),
        (r"oil|gas|exploration|midstream|refining", "Oil & Gas"),
        (r"coal|mining coal", "Coal"),
        (r"gold|precious metal", "Gold"),
        (r"copper|mining copper", "Copper"),
        (r"metal|mining", "Base Metals"),
        (r"shipping|freight|port|logistics", "Shipping & Logistics"),
        (r"aerospace|airline|aviation|space", "Aerospace & Aviation"),
        (r"automobile|ev|electric vehicle|auto parts", "Automotive / EV"),
        (r"beverage|drink|tea|coffee", "Beverage"),
        (r"retail|e-commerce|consumer", "Consumer / Retail"),
        (r"semiconductor|chip|wafer|fab|foundry", "Semiconductors"),
        (r"software|saas|application|cloud platform", "Application Software"),
        (r"cybersecurity|security software|endpoint", "Cybersecurity"),
        (r"data center|server|compute|gpu|accelerator", "Data Center Infrastructure"),
        (r"biotech|therapeutic|pharma|drug", "Biotech / Pharma"),
        (r"medical device|diagnostic|healthcare equipment", "Medical Devices"),
        (r"bank|financial|credit|lending|insurance", "Financial Services"),
        (r"utility|electric power|renewable|solar|wind", "Utilities / Clean Energy"),
        (r"industrial|machinery|aerospace|defense", "Industrials / Aerospace"),
        (r"telecom|communications equipment|network", "Telecom & Networking"),
    ]

    for pat, label in rules:
        if re.search(pat, text):
            return label

    sic_desc = str(row.get("sic_description", "")).strip()
    if sic_desc:
        return sic_desc
    return "Unclassified"


def classify_major_sector(row: pd.Series) -> str:
    text = " ".join(
        [
            str(row.get("security_name", "")),
            str(row.get("sic_description", "")),
            str(row.get("sec_title", "")),
            str(row.get("subsector", "")),
        ]
    ).lower()

    rules = [
        (r"semiconductor|software|cyber|data center|ai|optical|telecom", "科技"),
        (r"biotech|pharma|medical|health", "醫療健康"),
        (r"oil|gas|coal|utility|renewable|solar|wind", "能源與公用事業"),
        (r"bank|financial|insurance|credit|lending", "金融"),
        (r"retail|consumer|beverage|food|restaurant", "消費"),
        (r"industrial|machinery|aerospace|defense|shipping|logistics|automotive", "工業與運輸"),
        (r"metal|mining|gold|copper", "原材料"),
        (r"real estate|reit|property", "房地產"),
    ]
    for pat, label in rules:
        if re.search(pat, text):
            return label
    return "其他"


def fallback_zh_summary(row: pd.Series) -> str:
    name = str(row.get("security_name", "")).strip() or str(row.get("sec_title", "")).strip() or "該公司"
    sic_desc = str(row.get("sic_description", "")).strip()
    sub = str(row.get("subsector", "")).strip()
    if sic_desc:
        return f"{name}主要從事{sic_desc}相關業務。"
    if sub:
        return f"{name}主要業務可歸類於{sub}。"
    return f"{name}主要從事美國上市公司常見的相關業務。"


def build_universe(cfg: FetchConfig, max_sec_profiles: int | None = None) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": cfg.user_agent, "Accept-Encoding": "gzip, deflate"})

    listed = load_us_listed_symbols(session, cfg)
    sec_map = load_sec_ticker_map(session, cfg)
    df = listed.merge(sec_map, on="ticker", how="left")

    profiles: dict[int, dict[str, Any]] = {}
    cik_series = df["cik"].dropna().astype("Int64")
    unique_ciks = [int(x) for x in cik_series.unique() if pd.notna(x)]
    if max_sec_profiles is not None:
        unique_ciks = unique_ciks[: max(0, int(max_sec_profiles))]

    for idx, cik in enumerate(unique_ciks, start=1):
        try:
            profiles[cik] = fetch_sec_company_profile(session, cfg, cik)
        except Exception:
            profiles[cik] = {"sic": "", "sic_description": "", "entity_type": "", "fiscal_year_end": ""}
        if idx % 200 == 0:
            print(f"[INFO] SEC profiles fetched: {idx}/{len(unique_ciks)}")

    def _get_profile_value(cik_value: Any, key: str) -> Any:
        try:
            cik_int = int(cik_value)
        except Exception:
            return ""
        return profiles.get(cik_int, {}).get(key, "")

    for key in ["sic", "sic_description", "entity_type", "fiscal_year_end"]:
        df[key] = df["cik"].apply(lambda v: _get_profile_value(v, key))

    df["subsector"] = df.apply(classify_subsector, axis=1)
    df["major_sector"] = df.apply(classify_major_sector, axis=1)
    df["ai_subsector"] = ""
    df["ai_small_tags"] = ""
    df["company_summary_zh_tw"] = df.apply(fallback_zh_summary, axis=1)
    df["final_subsector"] = df["subsector"]
    df["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = df.sort_values(["subsector", "ticker"]).reset_index(drop=True)
    return df


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


def _ai_parse_json_fields(txt: str) -> tuple[str, str, str]:
    obj = json.loads(txt)
    ai_sub = str(obj.get("ai_subsector", "")).strip()
    tags_raw = obj.get("ai_small_tags", [])
    if isinstance(tags_raw, list):
        tags = [str(x).strip() for x in tags_raw if str(x).strip()]
    else:
        tags = [t.strip() for t in str(tags_raw).split(",") if t.strip()]
    ai_tags = "|".join(dict.fromkeys(tags))
    summary = str(obj.get("company_summary_zh_tw", "")).strip()
    return ai_sub, ai_tags, summary


def _build_ai_prompt_payload(row: pd.Series) -> dict[str, Any]:
    prompt = {
        "ticker": str(row.get("ticker", "")),
        "exchange": str(row.get("exchange", "")),
        "company_name": str(row.get("security_name", "")),
        "sec_title": str(row.get("sec_title", "")),
        "sic_description": str(row.get("sic_description", "")),
        "rule_subsector": str(row.get("subsector", "")),
        "major_sector": str(row.get("major_sector", "")),
        "task": (
            "請輸出JSON，欄位為 ai_subsector、ai_small_tags、company_summary_zh_tw。"
            "ai_subsector請用精細主題分類（例如光通訊、AI應用、存儲、石油與天然氣等）。"
            "ai_small_tags請輸出陣列，可多標籤（例如[\"茶飲\",\"光通訊\",\"以巴戰爭\"]）。"
            "company_summary_zh_tw請用繁體中文，1-2句，簡介公司主要業務。"
        ),
        "output_format": '{"ai_subsector":"...", "ai_small_tags":["..."], "company_summary_zh_tw":"..."}',
    }
    return prompt


def _ai_classify_one_openai(client: Any, model: str, row: pd.Series) -> tuple[str, str, str]:
    prompt = _build_ai_prompt_payload(row)

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a precise financial classifier. Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    )
    txt = _extract_output_text(resp)
    return _ai_parse_json_fields(txt)


def _ai_classify_one_anthropic(
    api_key: str,
    base_url: str,
    model: str,
    row: pd.Series,
    timeout_sec: int = 45,
) -> tuple[str, str, str]:
    prompt = _build_ai_prompt_payload(row)
    url = base_url.rstrip("/")
    if not url.endswith("/messages"):
        url = f"{url}/messages"

    payload = {
        "model": model,
        "max_tokens": 400,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are a precise financial classifier. Return strict JSON only.\n"
                    + json.dumps(prompt, ensure_ascii=False)
                ),
            }
        ],
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
    text_blocks = []
    for c in obj.get("content", []) or []:
        t = str(c.get("text", "")).strip()
        if t:
            text_blocks.append(t)
    txt = "\n".join(text_blocks).strip()
    return _ai_parse_json_fields(txt)


def apply_ai_stage(
    df: pd.DataFrame,
    enable_ai: bool,
    ai_model: str,
    ai_only_unclassified: bool,
    ai_max_rows: int | None,
    ai_sleep_sec: float,
    ai_api_key: str | None = None,
    ai_base_url: str | None = None,
    ai_api_type: str = "openai-responses",
) -> pd.DataFrame:
    if not enable_ai:
        return df
    if OpenAI is None:
        print("[WARN] openai package unavailable. Skip AI stage.")
        return df

    api_key = (ai_api_key or os.getenv("OPENAI_API_KEY", "") or os.getenv("KIMI_API_KEY", "")).strip()
    if not api_key:
        print("[WARN] No AI key found (OPENAI_API_KEY/KIMI_API_KEY). Skip AI stage.")
        return df

    base_url = (ai_base_url or os.getenv("OPENAI_BASE_URL", "") or os.getenv("KIMI_BASE_URL", "")).strip()
    client = None
    if ai_api_type == "openai-responses":
        if base_url:
            client = OpenAI(api_key=api_key, base_url=base_url)
        else:
            client = OpenAI(api_key=api_key)
    if ai_only_unclassified:
        candidate_idx = df.index[df["subsector"].astype(str).str.lower().eq("unclassified")].tolist()
    else:
        candidate_idx = df.index.tolist()

    if ai_max_rows is not None:
        candidate_idx = candidate_idx[: max(0, int(ai_max_rows))]

    print(f"[INFO] AI stage rows={len(candidate_idx)} model={ai_model}")
    for i, idx in enumerate(candidate_idx, start=1):
        row = df.loc[idx]
        try:
            if ai_api_type == "anthropic-messages":
                if not base_url:
                    raise RuntimeError("anthropic-messages requires --ai-base-url")
                ai_sub, ai_tags, summary = _ai_classify_one_anthropic(
                    api_key=api_key,
                    base_url=base_url,
                    model=ai_model,
                    row=row,
                )
            else:
                ai_sub, ai_tags, summary = _ai_classify_one_openai(client, ai_model, row)
            if ai_sub:
                df.at[idx, "ai_subsector"] = ai_sub
                df.at[idx, "final_subsector"] = ai_sub
            if ai_tags:
                df.at[idx, "ai_small_tags"] = ai_tags
            if summary:
                df.at[idx, "company_summary_zh_tw"] = summary
        except Exception as e:
            print(f"[WARN] AI failed ticker={row.get('ticker','')} err={e}")

        if ai_sleep_sec > 0:
            time.sleep(ai_sleep_sec + random.uniform(0.05, 0.2))
        if i % 100 == 0:
            print(f"[INFO] AI processed: {i}/{len(candidate_idx)}")

    return df


def write_excel(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        all_cols = [
            "ticker",
            "exchange",
            "security_name",
            "company_summary_zh_tw",
            "major_sector",
            "sec_title",
            "cik",
            "sic",
            "sic_description",
            "entity_type",
            "subsector",
            "ai_subsector",
            "ai_small_tags",
            "final_subsector",
            "generated_at",
        ]
        cols = [c for c in all_cols if c in df.columns]
        df[cols].to_excel(writer, sheet_name="all_stocks", index=False)

        summary = (
            df.groupby(["major_sector", "final_subsector"], dropna=False)["ticker"]
            .count()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        summary.to_excel(writer, sheet_name="summary", index=False)

        for major, g in df.groupby("major_sector", dropna=False):
            sheet = str(major)[:31] if str(major).strip() else "其他"
            g[cols].to_excel(writer, sheet_name=sheet, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build US stock universe Excel (exclude ADR by name rules).")
    p.add_argument(
        "--out",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors.xlsx",
        help="Output Excel path.",
    )
    p.add_argument(
        "--user-agent",
        default="stock-universe-builder/1.0 your_email@example.com",
        help="User-Agent for SEC requests (replace with your email).",
    )
    p.add_argument(
        "--max-sec-profiles",
        type=int,
        default=None,
        help="Debug mode: limit SEC profile requests.",
    )
    p.add_argument(
        "--request-sleep-sec",
        type=float,
        default=0.35,
        help="Sleep after each HTTP request. Increase for safer crawling.",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retries on 429/5xx.",
    )
    p.add_argument("--ai-enable", action="store_true", help="Enable AI second-stage classification.")
    p.add_argument("--ai-model", default="gpt-4.1-mini", help="OpenAI model for AI classification.")
    p.add_argument(
        "--ai-only-unclassified",
        action="store_true",
        help="Run AI only for rows where rule subsector is Unclassified.",
    )
    p.add_argument(
        "--ai-max-rows",
        type=int,
        default=800,
        help="Max rows to send to AI (cost control).",
    )
    p.add_argument(
        "--ai-sleep-sec",
        type=float,
        default=0.7,
        help="Sleep between AI calls to reduce rate risk.",
    )
    p.add_argument(
        "--ai-api-key",
        default=None,
        help="Optional AI API key. If omitted, read OPENAI_API_KEY or KIMI_API_KEY.",
    )
    p.add_argument(
        "--ai-base-url",
        default=None,
        help="Optional OpenAI-compatible base URL (for non-OpenAI providers).",
    )
    p.add_argument(
        "--ai-api-type",
        default="openai-responses",
        choices=["openai-responses", "anthropic-messages"],
        help="AI API style. Use anthropic-messages for providers exposing Anthropic Messages API.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FetchConfig(
        user_agent=args.user_agent,
        request_sleep_sec=max(0.0, float(args.request_sleep_sec)),
        max_retries=max(1, int(args.max_retries)),
    )
    df = build_universe(cfg=cfg, max_sec_profiles=args.max_sec_profiles)
    df = apply_ai_stage(
        df=df,
        enable_ai=bool(args.ai_enable),
        ai_model=str(args.ai_model),
        ai_only_unclassified=bool(args.ai_only_unclassified),
        ai_max_rows=args.ai_max_rows,
        ai_sleep_sec=max(0.0, float(args.ai_sleep_sec)),
        ai_api_key=args.ai_api_key,
        ai_base_url=args.ai_base_url,
        ai_api_type=str(args.ai_api_type),
    )

    out_path = Path(args.out)
    write_excel(df, out_path)
    print(f"[DONE] rows={len(df)} excel={out_path}")


if __name__ == "__main__":
    main()
