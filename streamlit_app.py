from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from ai_event_theme import classify_event_impact
from data_loader import DEFAULT_INPUT_PATH, DataBundle, load_bundle_from_path, load_bundle_from_upload
from github_sync import get_file_content, upsert_file_content
from notes_store import (
    DB_PATH,
    add_macro_note,
    add_stock_note,
    delete_keyword_synonym,
    delete_event_theme_hits_by_theme,
    clear_all_event_theme_hits,
    export_json_text,
    get_event_min_confidence,
    get_meta,
    import_json_text,
    init_db,
    list_event_theme_hits,
    list_keyword_synonyms,
    list_macro_notes,
    list_stock_notes,
    list_theme_rules,
    list_themes_ranked,
    macro_note_exists,
    replace_event_theme_hits,
    set_event_min_confidence,
    set_meta,
    update_macro_note,
    upsert_keyword_synonym,
    upsert_theme_rule,
)
from rss_ingest import DEFAULT_RSS_FEEDS, expand_keywords, fetch_rss_items

st.set_page_config(page_title="美股清單與筆記系統", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _load_default(path_str: str) -> DataBundle:
    return load_bundle_from_path(Path(path_str))


def _is_editor() -> bool:
    return bool(st.session_state.get("editor_authed", False))


def _safe_df_rows(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    dfx = pd.DataFrame(rows)
    for c in columns:
        if c not in dfx.columns:
            dfx[c] = ""
    return dfx[columns]


def _split_tags(s: str) -> list[str]:
    if not str(s or "").strip():
        return []
    return [x.strip() for x in re.split(r"[,\|;/，；、]+", str(s)) if x.strip()]


def _norm_title(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff ]+", " ", str(s or "").lower())
    return re.sub(r"\s+", " ", s).strip()


def _title_sim(a: str, b: str) -> float:
    sa, sb = set(_norm_title(a).split()), set(_norm_title(b).split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _get_llm_config() -> tuple[str, str, str]:
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    kimi_key = st.secrets.get("KIMI_API_KEY", "")
    api_key = openai_key or kimi_key
    base_url = (
        st.secrets.get("OPENAI_BASE_URL", "")
        or st.secrets.get("KIMI_BASE_URL", "")
        or "https://api.kimi.com/coding/v1"
    )
    model = (
        st.secrets.get("OPENAI_MODEL", "")
        or st.secrets.get("KIMI_MODEL", "")
        or "kimi-for-coding"
    )
    return api_key, base_url, model


def _theme_rule_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for r in list_theme_rules():
        if r.get("enabled", True):
            out[str(r["raw_theme"]).strip().lower()] = str(r["canonical_theme"]).strip()
    return out


PRECIOUS_ANCHORS = [
    "gold",
    "silver",
    "precious metal",
    "bullion",
    "gold miner",
    "silver miner",
    "gold mining",
    "silver mining",
    "金礦",
    "銀礦",
    "貴金屬",
]

SAAS_ANCHORS = [
    "saas",
    "subscription software",
    "project management software",
    "雲端軟件",
    "軟件",
]

OIL_GAS_ANCHORS = ["oil", "gas", "lng", "upstream", "midstream", "downstream", "refining", "pipeline", "opec", "石油", "天然氣"]
SEMIS_ANCHORS = ["semiconductor", "chip", "foundry", "wafer", "fab", "gpu", "asic", "半導體", "晶片"]
DEFENSE_ANCHORS = ["defense", "military", "aerospace", "missile", "drone", "naval", "army", "國防", "軍工", "航太"]
CYBER_ANCHORS = ["cyber", "security", "endpoint", "siem", "firewall", "zero trust", "網絡安全"]
BIOTECH_ANCHORS = ["biotech", "pharma", "drug", "clinical", "fda", "療法", "藥", "生技", "醫藥"]
BANK_ANCHORS = ["bank", "insurance", "broker", "asset management", "payments", "fintech", "銀行", "保險", "券商", "支付"]
POWER_ANCHORS = ["utility", "power", "grid", "nuclear", "renewable", "電力", "核能", "電網"]
LOGISTICS_ANCHORS = ["shipping", "logistics", "freight", "container", "rail", "air cargo", "航運", "物流"]


THEME_GUARDRAILS = [
    {"theme_keys": ["gold", "precious", "bullion", "金", "貴金屬"], "must_match": PRECIOUS_ANCHORS, "must_not": SAAS_ANCHORS},
    {"theme_keys": ["oil", "gas", "energy", "石油", "天然氣", "能源"], "must_match": OIL_GAS_ANCHORS, "must_not": SAAS_ANCHORS},
    {"theme_keys": ["semiconductor", "chip", "半導體", "晶片"], "must_match": SEMIS_ANCHORS, "must_not": []},
    {"theme_keys": ["defense", "military", "國防", "軍工", "航太"], "must_match": DEFENSE_ANCHORS, "must_not": []},
    {"theme_keys": ["cyber", "security", "網絡安全"], "must_match": CYBER_ANCHORS, "must_not": []},
    {"theme_keys": ["biotech", "pharma", "醫藥", "生技"], "must_match": BIOTECH_ANCHORS, "must_not": []},
    {"theme_keys": ["bank", "fintech", "金融", "保險"], "must_match": BANK_ANCHORS, "must_not": []},
    {"theme_keys": ["power", "utility", "電力", "核能"], "must_match": POWER_ANCHORS, "must_not": []},
    {"theme_keys": ["shipping", "logistics", "航運", "物流"], "must_match": LOGISTICS_ANCHORS, "must_not": []},
    {"theme_keys": ["saas", "cloud software", "雲端軟件", "軟件"], "must_match": SAAS_ANCHORS, "must_not": PRECIOUS_ANCHORS + OIL_GAS_ANCHORS},
]


def _stock_blob(row: dict) -> str:
    return " ".join(
        [
            str(row.get("company_name", "")),
            str(row.get("sector", "")),
            str(row.get("subsector", "")),
            str(row.get("tags", "")),
            str(row.get("summary", "")),
        ]
    ).lower()


def _theme_is_precious(theme: str) -> bool:
    t = str(theme or "").lower()
    keys = ["gold", "precious", "bullion", "金", "貴金屬", "銀"]
    return any(k in t for k in keys)


def _is_precious_stock(row: dict) -> bool:
    b = _stock_blob(row)
    if any(k in b for k in PRECIOUS_ANCHORS):
        return True
    # strict word-boundary fallback for English
    return bool(re.search(r"\b(gold|silver|bullion|mining)\b", b))


def _is_saas_stock(row: dict) -> bool:
    b = _stock_blob(row)
    return any(k in b for k in SAAS_ANCHORS)


def _keep_english_like_terms(items: list[str]) -> list[str]:
    out = []
    for x in items:
        s = str(x or "").strip()
        if not s:
            continue
        # Keep mostly English/number/symbol query terms, drop pure CJK long phrases.
        if re.fullmatch(r"[A-Za-z0-9\-\+\.\s]{2,60}", s):
            out.append(re.sub(r"\s+", " ", s).strip().lower())
    return list(dict.fromkeys(out))


def _theme_guardrail_pass(theme: str, row: dict) -> bool:
    t = str(theme or "").lower()
    b = _stock_blob(row)
    for g in THEME_GUARDRAILS:
        if any(k in t for k in g["theme_keys"]):
            if not any(x in b for x in g["must_match"]):
                return False
            if any(x in b for x in g["must_not"]):
                return False
            return True
    return True


def _run_ai_for_event(event_row: dict, df: pd.DataFrame) -> tuple[int, int]:
    api_key, base_url, model = _get_llm_config()
    if not api_key:
        raise RuntimeError("缺少 API key（OPENAI_API_KEY / KIMI_API_KEY）")

    rows = df[["ticker", "company_name", "sector", "subsector", "tags", "summary"]].to_dict("records")
    res = classify_event_impact(
        api_key=api_key,
        base_url=base_url,
        model=model,
        event_title=str(event_row.get("event_title", "")),
        event_detail=str(event_row.get("event_detail", "")),
        universe_rows=rows,
    )

    min_conf = get_event_min_confidence()
    rule_map = _theme_rule_map()
    refined = []
    row_map = {str(r.get("ticker", "")).upper(): r for r in rows}
    for h in res.hits:
        conf = float(h.get("confidence", 0.0) or 0.0)
        if conf < min_conf:
            continue
        raw_theme = str(h.get("theme", "")).strip()
        h["theme"] = rule_map.get(raw_theme.lower(), raw_theme)
        tk = str(h.get("ticker", "")).upper()
        r0 = row_map.get(tk, {})
        if not _theme_guardrail_pass(h["theme"], r0):
            continue
        # Guardrail: precious-metals theme should not map to obvious SaaS names
        if _theme_is_precious(h["theme"]):
            if _is_saas_stock(r0) and not _is_precious_stock(r0):
                continue
        refined.append(h)

    # Augment missing obvious precious-metals names (e.g., NEM) when theme exists
    has_precious_theme = any(_theme_is_precious(x.get("theme", "")) for x in refined)
    if has_precious_theme:
        exist_tickers = {str(x.get("ticker", "")).upper() for x in refined}
        base_impact = "中性"
        for x in refined:
            if _theme_is_precious(x.get("theme", "")):
                base_impact = str(x.get("impact", "中性"))
                break
        added_cnt = 0
        for r in rows:
            tk = str(r.get("ticker", "")).upper()
            if tk in exist_tickers:
                continue
            if _is_precious_stock(r):
                refined.append(
                    {
                        "ticker": tk,
                        "theme": "貴金屬市場波動",
                        "impact": base_impact,
                        "confidence": max(0.56, min_conf),
                        "reason": "規則補全：公司屬性與貴金屬/礦業高度相關",
                    }
                )
                added_cnt += 1
                if added_cnt >= 30:
                    break

    replace_event_theme_hits(int(event_row["id"]), refined)
    return len(set([x["theme"] for x in refined])), len(refined)


def _suggest_keyword_expansions_ai(term: str, current_expanded: list[str]) -> tuple[list[str], str]:
    api_key, base_url, model = _get_llm_config()
    if not api_key:
        return [], "缺少 API key"
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = (
        "你是金融新聞檢索助手。請根據輸入主題詞，輸出 6-12 個最有檢索價值的擴展關鍵字（中英混合）。"
        "只輸出 JSON 陣列，不要其他文字。"
        f"\n主題詞: {term}"
        f"\n已存在詞（避免重複）: {', '.join(current_expanded[:40])}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=220,
        )
        text = (resp.choices[0].message.content or "").strip()
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            text = m.group(0)
        arr = json.loads(text)
        if not isinstance(arr, list):
            return [], "模型未回傳 JSON 陣列"
        out = []
        seen = set([x.lower() for x in current_expanded])
        for x in arr:
            sx = str(x).strip()
            if not sx or sx.lower() in seen:
                continue
            seen.add(sx.lower())
            out.append(sx)
        return out[:20], ""
    except Exception as e:
        return [], f"{type(e).__name__}: {str(e)[:220]}"


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()
    kw = str(f.get("keyword", "")).strip().lower()
    if kw:
        out = out[out["search_blob"].str.contains(kw, na=False)]
    if f.get("sectors"):
        out = out[out["sector"].isin(f["sectors"])]
    if f.get("subsectors"):
        out = out[out["subsector"].isin(f["subsectors"])]
    if f.get("exchanges"):
        out = out[out["exchange"].isin(f["exchanges"])]
    tags = f.get("ai_tags", []) or []
    if tags:
        mode = str(f.get("tag_mode", "any"))
        if mode == "all":
            out = out[out["tags"].astype(str).apply(lambda x: all(t in _split_tags(x) for t in tags))]
        else:
            out = out[out["tags"].astype(str).apply(lambda x: any(t in _split_tags(x) for t in tags))]
    return out


def _import_rss_rows(
    rdf2: pd.DataFrame,
    picks: list[str],
    auto_ai: bool,
    df: pd.DataFrame,
    show_progress: bool = False,
) -> tuple[int, int, int, int]:
    existing = list_macro_notes()
    imported = merged = skipped = mapped = 0
    selected_rows = [r for _, r in rdf2.iterrows() if r["title"] in picks]
    total = len(selected_rows)
    bar = st.progress(0) if show_progress else None
    status = st.empty() if show_progress else None
    for idx, r in enumerate(selected_rows, start=1):
        if show_progress and status is not None:
            status.info(f"處理中 {idx}/{total}：{str(r.get('title',''))[:80]}")
        if r["title"] not in picks:
            continue
        title = (str(r.get("title_zh_tw", "")).strip() + " | " + str(r.get("title", "")).strip()).strip(" |")
        link = str(r.get("link", "")).strip()
        summary = str(r.get("summary", "")).strip()

        if macro_note_exists(title, link):
            skipped += 1
            continue

        target = None
        for e in existing[:220]:
            if _title_sim(title, str(e.get("event_title", ""))) >= 0.74:
                target = e
                break

        if target is not None:
            src = str(target.get("source_url", ""))
            src2 = src if (not link or link in src.split("|")) else (src + "|" + link if src else link)
            det = str(target.get("event_detail", ""))
            det2 = det if (not summary or summary in det) else (det + "\n\n" + summary)[:8000]
            update_macro_note(
                int(target["id"]),
                str(target["note_date"]),
                str(target["event_title"]),
                det2,
                target["affected_sectors_list"],
                target["affected_subsectors_list"],
                target["affected_tickers_list"],
                str(target["impact"]),
                src2,
            )
            merged += 1
            continue

        nid = add_macro_note(
            note_date=str(r.get("published_at", ""))[:10],
            event_title=title[:220],
            event_detail=summary[:4000],
            affected_sectors=[],
            affected_subsectors=[],
            affected_tickers=[],
            impact="中性",
            source_url=link,
        )
        imported += 1
        if auto_ai:
            _, hc = _run_ai_for_event({"id": nid, "event_title": title, "event_detail": summary}, df)
            mapped += hc
        if show_progress and bar is not None and total > 0:
            bar.progress(min(100, int(idx * 100 / total)))
    if show_progress and status is not None:
        status.success(f"完成：新增 {imported} / 合併 {merged} / 略過 {skipped} / AI 命中 {mapped}")
    return imported, merged, skipped, mapped


def _maybe_daily_auto_rss(raw_keywords: list[str], feeds: list[str], lookback: int, max_items: int, auto_ai: bool, custom_synonyms: dict[str, list[str]], df: pd.DataFrame) -> tuple[bool, str]:
    enabled = str(get_meta("rss_daily_auto_enabled", "0")) == "1"
    if not enabled:
        return False, "disabled"
    today = datetime.now().strftime("%Y-%m-%d")
    if get_meta("rss_daily_last_run_date", "") == today:
        return False, "already-ran"
    if not raw_keywords or not feeds:
        return False, "no-config"

    items = fetch_rss_items(raw_keywords, feeds, int(lookback), int(max_items), custom_synonyms=custom_synonyms)
    if not items:
        set_meta("rss_daily_last_run_date", today)
        set_meta("rss_daily_last_summary", "no-items")
        return False, "no-items"
    rdf = pd.DataFrame([x.__dict__ for x in items])
    picks = rdf["title"].tolist()[: min(40, len(rdf))]
    imported, merged, skipped, mapped = _import_rss_rows(rdf, picks, auto_ai, df)
    summary = f"imported={imported},merged={merged},skipped={skipped},mapped={mapped}"
    set_meta("rss_daily_last_run_date", today)
    set_meta("rss_daily_last_summary", summary)
    return True, summary


def _rebuild_all_theme_hits(df: pd.DataFrame) -> tuple[int, int]:
    events = list_macro_notes()
    if not events:
        return 0, 0
    bar = st.progress(0)
    status = st.empty()
    rebuilt_themes = 0
    rebuilt_hits = 0
    total = len(events)
    for i, ev in enumerate(events, start=1):
        status.info(f"重算事件 {i}/{total}: {str(ev.get('event_title',''))[:90]}")
        tc, hc = _run_ai_for_event(ev, df)
        rebuilt_themes += tc
        rebuilt_hits += hc
        bar.progress(min(100, int(i * 100 / total)))
    status.success(f"重算完成：事件 {total} 筆，主題數累計 {rebuilt_themes}，命中累計 {rebuilt_hits}")
    return rebuilt_themes, rebuilt_hits


def main() -> None:
    init_db()
    if "bundle" not in st.session_state:
        st.session_state["bundle"] = _load_default(str(DEFAULT_INPUT_PATH))
    if "filters" not in st.session_state:
        st.session_state["filters"] = {
            "keyword": "",
            "sectors": [],
            "subsectors": [],
            "exchanges": [],
            "ai_tags": [],
            "tag_mode": "any",
        }

    b: DataBundle = st.session_state["bundle"]
    set_meta("data_source_name", b.source_name)
    df = b.schema_df.copy()
    filtered_df = _apply_filters(df, st.session_state["filters"])

    st.title("📈 美股清單與筆記系統")
    st.caption(f"資料來源：{b.source_name} | Hash：`{b.source_hash}` | 載入時間：{b.loaded_at}")

    with st.sidebar:
        st.subheader("資料與權限")
        uploaded = st.file_uploader("上傳新 Excel", type=["xlsx"])
        if st.button("載入上傳檔案", disabled=uploaded is None):
            st.session_state["bundle"] = load_bundle_from_upload(uploaded)
            st.rerun()

        pwd = st.secrets.get("EDITOR_PASSWORD", "")
        ip = st.text_input("編輯密碼", type="password")
        if st.button("登入編輯模式"):
            st.session_state["editor_authed"] = bool(pwd and ip == pwd)
            st.rerun()
        if _is_editor() and st.button("登出編輯模式"):
            st.session_state["editor_authed"] = False
            st.rerun()
        st.caption("你目前可編輯" if _is_editor() else "你目前為訪客只讀")

        st.markdown("---")
        st.subheader("篩選")
        all_sectors = sorted([x for x in df["sector"].fillna("").astype(str).unique().tolist() if x.strip()])
        all_subsectors = sorted([x for x in df["subsector"].fillna("").astype(str).unique().tolist() if x.strip()])
        all_exchanges = sorted([x for x in df["exchange"].fillna("").astype(str).unique().tolist() if x.strip()])
        tag_pool = set()
        for tx in df["tags"].fillna("").astype(str).tolist():
            for tg in _split_tags(tx):
                tag_pool.add(tg)
        all_ai_tags = sorted(tag_pool)

        f0 = st.session_state["filters"]
        with st.form("sidebar_filter_form"):
            fk = st.text_input("關鍵字搜尋（代號/公司/簡介/子板塊/標籤）", value=f0.get("keyword", ""))
            fs = st.multiselect("大分類", all_sectors, default=[x for x in f0.get("sectors", []) if x in all_sectors])
            fss = st.multiselect("最終子分類", all_subsectors, default=[x for x in f0.get("subsectors", []) if x in all_subsectors])
            fe = st.multiselect("交易所", all_exchanges, default=[x for x in f0.get("exchanges", []) if x in all_exchanges])
            ft = st.multiselect("AI 小分類標籤", all_ai_tags, default=[x for x in f0.get("ai_tags", []) if x in all_ai_tags])
            fm = st.radio("標籤匹配模式", ["任一命中", "全部命中"], index=0 if f0.get("tag_mode", "any") == "any" else 1, horizontal=True)
            c1, c2 = st.columns(2)
            ap = c1.form_submit_button("套用篩選", use_container_width=True)
            cl = c2.form_submit_button("清空篩選", use_container_width=True)
            if cl:
                st.session_state["filters"] = {"keyword": "", "sectors": [], "subsectors": [], "exchanges": [], "ai_tags": [], "tag_mode": "any"}
                st.rerun()
            if ap:
                st.session_state["filters"] = {
                    "keyword": fk,
                    "sectors": fs,
                    "subsectors": fss,
                    "exchanges": fe,
                    "ai_tags": ft,
                    "tag_mode": "all" if fm == "全部命中" else "any",
                }
                st.rerun()
        st.caption(f"篩選後筆數：{len(filtered_df):,}")

    tabs = st.tabs(["股票清單", "時事事件筆記", "時事主題", "字典/規則", "設定/同步"])

    with tabs[0]:
        k = st.text_input("頁內搜尋（代號/公司/簡介/子板塊/標籤）", "")
        show = filtered_df.copy()
        if k.strip():
            show = show[show["search_blob"].str.contains(k.strip().lower(), na=False)]
        st.dataframe(
            show[["ticker", "company_name", "exchange", "sector", "subsector", "tags", "summary"]].rename(
                columns={"ticker": "代號", "company_name": "公司", "exchange": "交易所", "sector": "大分類", "subsector": "最終子分類", "tags": "AI標籤", "summary": "簡介"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        t = st.selectbox("個股筆記代號", sorted(show["ticker"].tolist())[:800] if len(show) else [])
        if t:
            notes = list_stock_notes(ticker=t)
            st.caption(f"{t} 個股筆記：{len(notes)} 筆")
            if _is_editor():
                with st.form("add_stock_note_form"):
                    nd = st.date_input("日期", value=date.today())
                    nt = st.text_input("標題")
                    ndt = st.text_area("內容")
                    ni = st.selectbox("方向", ["利多", "利空", "中性"])
                    if st.form_submit_button("新增個股筆記"):
                        add_stock_note(t, str(nd), nt, ndt, ni, 0.6, "", [])
                        st.rerun()
            if notes:
                st.dataframe(_safe_df_rows(notes, ["note_date", "impact", "event_title", "event_detail"]), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("RSS v2 新聞匯入")
        default_kw = "美伊戰爭, 以伊戰爭, 中東衝突, 荷姆茲海峽, 石油, 天然氣, 有色金屬, 化肥, spacex, 腦機接口, 量子, gtc"
        kw_text = st.text_area("關鍵字（逗號分隔）", value=default_kw, key="rss_kw")
        feeds_text = st.text_area("RSS feeds（每行一個）", value="\n".join(DEFAULT_RSS_FEEDS), height=120, key="rss_feeds")
        raw_keywords = [x.strip() for x in kw_text.split(",") if x.strip()]
        feeds = [x.strip() for x in feeds_text.splitlines() if x.strip()]
        lookback_for_refetch = int(st.session_state.get("rss_lookback", 48))
        max_items_for_refetch = int(st.session_state.get("rss_max_items", 30))
        custom = {r["keyword"]: r["expansions_list"] for r in list_keyword_synonyms() if r.get("enabled", True)}
        expanded, kmap = expand_keywords(raw_keywords, custom_synonyms=custom)
        st.caption(f"擴展關鍵字：{len(expanded)}（原始：{len(raw_keywords)}）")

        unmapped = [k for k in raw_keywords if not kmap.get(k)]
        with st.expander("查看/管理擴展關鍵字", expanded=False):
            if unmapped:
                st.warning(f"未映射關鍵字：{', '.join(unmapped)}")
                if st.button("AI 補全未映射關鍵字並寫入字典"):
                    added = 0
                    last_err = ""
                    for uk in unmapped[:10]:
                        sug, err = _suggest_keyword_expansions_ai(uk, expanded)
                        sug = _keep_english_like_terms(sug)
                        if sug:
                            upsert_keyword_synonym(uk, sug, True)
                            added += 1
                        elif err:
                            last_err = err
                    if added:
                        # Save + immediate refetch so user does not need to click fetch again.
                        custom2 = {r["keyword"]: r["expansions_list"] for r in list_keyword_synonyms() if r.get("enabled", True)}
                        items2 = fetch_rss_items(raw_keywords, feeds, lookback_for_refetch, max_items_for_refetch, custom_synonyms=custom2)
                        st.session_state["rss_items"] = [x.__dict__ for x in items2]
                        st.success(f"已補全 {added} 個關鍵字，並已自動重抓 RSS。")
                        st.rerun()
                    else:
                        st.error(f"目前無法補全：{last_err or '模型未回傳可用結果'}")
            compact = st.checkbox("精簡顯示（每詞只顯示前 3 個）", value=True)
            for kx, vals in kmap.items():
                if not vals:
                    continue
                if compact:
                    sv = vals[:3]
                    more = len(vals) - len(sv)
                    st.caption(f"{kx} -> {', '.join(sv)}" + (f" ... (+{more})" if more > 0 else ""))
                else:
                    st.caption(f"{kx} -> {', '.join(vals)}")

        c1, c2, c3 = st.columns(3)
        lookback = c1.number_input("回看小時", min_value=6, max_value=240, value=48, step=6, key="rss_lookback")
        max_items = c2.number_input("最多新聞數", min_value=10, max_value=300, value=30, step=10, key="rss_max_items")
        auto_ai = c3.checkbox("匯入後自動 AI 映射", value=True)
        auto_daily = st.checkbox("每日自動抓 RSS（每天首次開站自動執行一次）", value=(str(get_meta("rss_daily_auto_enabled", "0")) == "1"))
        set_meta("rss_daily_auto_enabled", "1" if auto_daily else "0")
        st.caption(f"每日自動：{'開啟' if auto_daily else '關閉'} | 上次執行：{get_meta('rss_daily_last_run_date', '-')}")

        ran, msg = _maybe_daily_auto_rss(raw_keywords, feeds, int(lookback), int(max_items), auto_ai, custom, df)
        if ran:
            st.success(f"今日自動抓取完成：{msg}")

        d1, d2 = st.columns(2)
        if d1.button("抓取 RSS"):
            items = fetch_rss_items(raw_keywords, feeds, int(lookback), int(max_items), custom_synonyms=custom)
            st.session_state["rss_items"] = [x.__dict__ for x in items]

        rss_items = st.session_state.get("rss_items", [])
        if rss_items:
            rdf = pd.DataFrame(rss_items)
            st.dataframe(_safe_df_rows(rdf.to_dict("records"), ["published_at", "source", "title", "title_zh_tw", "matched_keywords", "matched_expanded", "link"]), use_container_width=True, hide_index=True)
            q = st.text_input("在已抓取新聞中搜尋（中英）", "")
            rdf2 = rdf.copy()
            if q.strip():
                ql = q.strip().lower()
                rdf2 = rdf2[rdf2["title"].astype(str).str.lower().str.contains(ql, na=False) | rdf2["title_zh_tw"].astype(str).str.lower().str.contains(ql, na=False)]
            picks = st.multiselect("選擇要匯入的新聞", rdf2["title"].tolist(), default=rdf2["title"].tolist()[: min(20, len(rdf2))])
            if d2.button("匯入選中新聞"):
                imported, merged, skipped, mapped = _import_rss_rows(rdf2, picks, auto_ai, df, show_progress=True)
                st.success(f"新增 {imported} / 合併 {merged} / 略過 {skipped} / AI 命中 {mapped}")
                st.rerun()

        st.subheader("現有時事事件")
        events = list_macro_notes()
        st.dataframe(_safe_df_rows(events, ["id", "note_date", "impact", "event_title", "source_url"]), use_container_width=True, hide_index=True)
        if _is_editor() and events:
            eid = st.selectbox("選擇事件 ID 套用 AI", [x["id"] for x in events])
            if st.button("對選中事件跑 AI 映射"):
                row = next(x for x in events if x["id"] == eid)
                tc, hc = _run_ai_for_event(row, df)
                st.success(f"主題 {tc} 個，命中 {hc} 筆")
                st.rerun()

    with tabs[2]:
        st.subheader("時事主題（熱度排名）")
        tq = st.text_input("主題搜尋", "")
        rows = list_themes_ranked()
        if tq.strip():
            rows = [x for x in rows if tq.strip().lower() in str(x["theme"]).lower()]
        st.dataframe(
            _safe_df_rows(rows, ["theme", "heat_score", "stock_count", "hit_count", "recent_date", "bull_count", "bear_count", "neutral_count"]).rename(
                columns={
                    "theme": "主題",
                    "heat_score": "熱度分數",
                    "stock_count": "股票數",
                    "hit_count": "命中數",
                    "recent_date": "最近日期",
                    "bull_count": "利多",
                    "bear_count": "利空",
                    "neutral_count": "中性",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("計算說明", expanded=False):
            st.markdown(
                "- `股票數`：主題下去重後的 ticker 數量。\n"
                "- `命中數`：主題命中總筆數（同一 ticker 可因不同事件重複）。\n"
                "- 主題命中由事件 AI + 守門規則共同決定。\n"
                "- 建議先按 `最低 confidence` 再觀察結果。"
            )
        if rows:
            picked = st.selectbox("查看主題", [x["theme"] for x in rows])
            c1, c2, c3 = st.columns(3)
            if c1.button("清理選中主題命中", key="btn_clear_theme_hits", disabled=not _is_editor()):
                n = delete_event_theme_hits_by_theme(picked)
                st.success(f"已清理 {picked} 命中：{n} 筆")
                st.rerun()
            if c2.button("清理全部主題命中", key="btn_clear_all_hits", disabled=not _is_editor()):
                n = clear_all_event_theme_hits()
                st.success(f"已清理全部命中：{n} 筆")
                st.rerun()
            if c3.button("重跑全部事件 AI", key="btn_rebuild_all_hits", disabled=not _is_editor()):
                _rebuild_all_theme_hits(df)
                st.rerun()
            if not _is_editor():
                st.caption("訪客模式可查看；進入編輯模式後可使用清理/重跑按鈕。")
            hits = list_event_theme_hits(theme_keyword=picked)
            st.dataframe(_safe_df_rows(hits, ["ticker", "theme", "impact", "confidence", "reason", "note_date", "event_title"]), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("字典 / 規則")
        with st.form("kw_add_form"):
            k = st.text_input("關鍵字")
            ex = st.text_input("擴展詞（用 | 分隔）")
            en = st.checkbox("啟用", value=True)
            if st.form_submit_button("新增/更新關鍵字"):
                upsert_keyword_synonym(k, [x.strip() for x in ex.split("|") if x.strip()], en)
                st.rerun()
        kws = list_keyword_synonyms()
        if kws:
            st.dataframe(_safe_df_rows(kws, ["keyword", "expansions", "enabled", "updated_at"]), use_container_width=True, hide_index=True)
            dk = st.selectbox("刪除關鍵字", [""] + [x["keyword"] for x in kws])
            if st.button("刪除關鍵字") and dk:
                delete_keyword_synonym(dk)
                st.rerun()

        with st.form("theme_rule_form"):
            raw = st.text_input("原始主題")
            can = st.text_input("正規主題")
            en2 = st.checkbox("啟用規則", value=True)
            if st.form_submit_button("新增/更新主題規則"):
                upsert_theme_rule(raw, can, en2)
                st.rerun()
        rules = list_theme_rules()
        if rules:
            st.dataframe(_safe_df_rows(rules, ["raw_theme", "canonical_theme", "enabled", "updated_at"]), use_container_width=True, hide_index=True)

        cur = get_event_min_confidence()
        nv = st.slider("最低信心閾值", 0.0, 1.0, float(cur), 0.05)
        if st.button("保存閾值"):
            set_event_min_confidence(float(nv))
            st.success("已保存")

    with tabs[4]:
        st.subheader("設定 / 同步")
        st.caption(f"本地 DB：{DB_PATH}")
        payload = export_json_text()
        st.download_button("下載本地備份", payload.encode("utf-8"), "notes_export.json", "application/json")
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        path = st.secrets.get("GITHUB_NOTES_PATH", "data/notes_export.json")
        if token and repo:
            c1, c2 = st.columns(2)
            if c1.button("同步到 GitHub", disabled=not _is_editor()):
                upsert_file_content(token, repo, path, branch, payload, f"Update notes backup {datetime.now():%Y-%m-%d %H:%M:%S}")
                st.success("已同步")
            if c2.button("從 GitHub 還原", disabled=not _is_editor()):
                txt, _ = get_file_content(token, repo, path, branch)
                if txt.strip():
                    import_json_text(txt, replace=True)
                    st.success("已還原")
                    st.rerun()
        st.caption(f"上次同步：{get_meta('last_sync_to_github', '-')}")


if __name__ == "__main__":
    main()
