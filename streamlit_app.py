from __future__ import annotations

import json
import re
from datetime import date, datetime
from io import BytesIO
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

st.set_page_config(page_title="缇庤偂娓呭柈鑸囩瓎瑷樼郴绲?, page_icon="馃搱", layout="wide")


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
    return [x.strip() for x in re.split(r"[,\|;/锛岋紱銆乚+", str(s)) if x.strip()]


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


def _sync_notes_to_github_now() -> tuple[bool, str]:
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    path = st.secrets.get("GITHUB_NOTES_PATH", "data/notes_export.json")
    if not token or not repo:
        return False, "缂哄皯 GITHUB_TOKEN 鎴?GITHUB_REPO"
    payload = export_json_text()
    upsert_file_content(token, repo, path, branch, payload, f"Update notes backup {datetime.now():%Y-%m-%d %H:%M:%S}")
    set_meta("last_sync_to_github", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return True, "ok"


PRECIOUS_ANCHORS = [
    "gold",
    "silver",
    "precious metal",
    "bullion",
    "gold miner",
    "silver miner",
    "gold mining",
    "silver mining",
    "閲戠う",
    "閵€绀?,
    "璨撮噾灞?,
]

SAAS_ANCHORS = [
    "saas",
    "subscription software",
    "project management software",
    "闆茬杌熶欢",
    "杌熶欢",
]

OIL_GAS_ANCHORS = ["oil", "gas", "lng", "upstream", "midstream", "downstream", "refining", "pipeline", "opec", "鐭虫补", "澶╃劧姘?]
SEMIS_ANCHORS = ["semiconductor", "chip", "foundry", "wafer", "fab", "gpu", "asic", "鍗婂皫楂?, "鏅剁墖"]
DEFENSE_ANCHORS = ["defense", "military", "aerospace", "missile", "drone", "naval", "army", "鍦嬮槻", "杌嶅伐", "鑸お"]
CYBER_ANCHORS = ["cyber", "security", "endpoint", "siem", "firewall", "zero trust", "缍茬怠瀹夊叏"]
BIOTECH_ANCHORS = ["biotech", "pharma", "drug", "clinical", "fda", "鐧傛硶", "钘?, "鐢熸妧", "閱棩"]
BANK_ANCHORS = ["bank", "insurance", "broker", "asset management", "payments", "fintech", "閵€琛?, "淇濋毆", "鍒稿晢", "鏀粯"]
POWER_ANCHORS = ["utility", "power", "grid", "nuclear", "renewable", "闆诲姏", "鏍歌兘", "闆荤恫"]
LOGISTICS_ANCHORS = ["shipping", "logistics", "freight", "container", "rail", "air cargo", "鑸亱", "鐗╂祦"]


THEME_GUARDRAILS = [
    {"theme_keys": ["gold", "precious", "bullion", "閲?, "璨撮噾灞?], "must_match": PRECIOUS_ANCHORS, "must_not": SAAS_ANCHORS},
    {"theme_keys": ["oil", "gas", "energy", "鐭虫补", "澶╃劧姘?, "鑳芥簮"], "must_match": OIL_GAS_ANCHORS, "must_not": SAAS_ANCHORS},
    {"theme_keys": ["semiconductor", "chip", "鍗婂皫楂?, "鏅剁墖"], "must_match": SEMIS_ANCHORS, "must_not": []},
    {"theme_keys": ["defense", "military", "鍦嬮槻", "杌嶅伐", "鑸お"], "must_match": DEFENSE_ANCHORS, "must_not": []},
    {"theme_keys": ["cyber", "security", "缍茬怠瀹夊叏"], "must_match": CYBER_ANCHORS, "must_not": []},
    {"theme_keys": ["biotech", "pharma", "閱棩", "鐢熸妧"], "must_match": BIOTECH_ANCHORS, "must_not": []},
    {"theme_keys": ["bank", "fintech", "閲戣瀺", "淇濋毆"], "must_match": BANK_ANCHORS, "must_not": []},
    {"theme_keys": ["power", "utility", "闆诲姏", "鏍歌兘"], "must_match": POWER_ANCHORS, "must_not": []},
    {"theme_keys": ["shipping", "logistics", "鑸亱", "鐗╂祦"], "must_match": LOGISTICS_ANCHORS, "must_not": []},
    {"theme_keys": ["saas", "cloud software", "闆茬杌熶欢", "杌熶欢"], "must_match": SAAS_ANCHORS, "must_not": PRECIOUS_ANCHORS + OIL_GAS_ANCHORS},
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
    keys = ["gold", "precious", "bullion", "閲?, "璨撮噾灞?, "閵€"]
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


def _to_tv_symbol(exchange: str, ticker: str) -> str:
    ex = str(exchange or "").upper().strip()
    tk = str(ticker or "").upper().strip()
    if not tk:
        return ""
    if "NASDAQ" in ex:
        prefix = "NASDAQ"
    elif "AMEX" in ex:
        prefix = "AMEX"
    elif "NYSE" in ex:
        prefix = "NYSE"
    elif "ARCA" in ex:
        prefix = "AMEX"
    elif "BATS" in ex:
        prefix = "BATS"
    else:
        prefix = ex.replace(" ", "") if ex else "NASDAQ"
    return f"{prefix}:{tk}"


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
        raise RuntimeError("缂哄皯 API key锛圤PENAI_API_KEY / KIMI_API_KEY锛?)

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
        base_impact = "涓€?
        for x in refined:
            if _theme_is_precious(x.get("theme", "")):
                base_impact = str(x.get("impact", "涓€?))
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
                        "theme": "璨撮噾灞競鍫存尝鍕?,
                        "impact": base_impact,
                        "confidence": max(0.56, min_conf),
                        "reason": "瑕忓墖瑁滃叏锛氬叕鍙稿爆鎬ц垏璨撮噾灞?绀︽キ楂樺害鐩搁棞",
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
        return [], "缂哄皯 API key"
    client = OpenAI(api_key=api_key, base_url=base_url)

    prompt = (
        "浣犳槸閲戣瀺鏂拌仦妾㈢储鍔╂墜銆傝珛鏍规摎杓稿叆涓婚瑭烇紝杓稿嚭 6-12 鍊嬫渶鏈夋绱㈠児鍊肩殑鎿村睍闂滈嵉瀛楋紙涓嫳娣峰悎锛夈€?
        "鍙几鍑?JSON 闄ｅ垪锛屼笉瑕佸叾浠栨枃瀛椼€?
        f"\n涓婚瑭? {term}"
        f"\n宸插瓨鍦ㄨ锛堥伩鍏嶉噸瑜囷級: {', '.join(current_expanded[:40])}"
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
            return [], "妯″瀷鏈洖鍌?JSON 闄ｅ垪"
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
            status.info(f"铏曠悊涓?{idx}/{total}锛歿str(r.get('title',''))[:80]}")
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
            impact="涓€?,
            source_url=link,
        )
        imported += 1
        if auto_ai:
            _, hc = _run_ai_for_event({"id": nid, "event_title": title, "event_detail": summary}, df)
            mapped += hc
        if show_progress and bar is not None and total > 0:
            bar.progress(min(100, int(idx * 100 / total)))
    if show_progress and status is not None:
        status.success(f"瀹屾垚锛氭柊澧?{imported} / 鍚堜降 {merged} / 鐣ラ亷 {skipped} / AI 鍛戒腑 {mapped}")
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
        status.info(f"閲嶇畻浜嬩欢 {i}/{total}: {str(ev.get('event_title',''))[:90]}")
        tc, hc = _run_ai_for_event(ev, df)
        rebuilt_themes += tc
        rebuilt_hits += hc
        bar.progress(min(100, int(i * 100 / total)))
    status.success(f"閲嶇畻瀹屾垚锛氫簨浠?{total} 绛嗭紝涓婚鏁哥疮瑷?{rebuilt_themes}锛屽懡涓疮瑷?{rebuilt_hits}")
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

    st.title("馃搱 缇庤偂娓呭柈鑸囩瓎瑷樼郴绲?)
    st.caption(f"璩囨枡渚嗘簮锛歿b.source_name} | Hash锛歚{b.source_hash}` | 杓夊叆鏅傞枔锛歿b.loaded_at}")

    with st.sidebar:
        st.subheader("璩囨枡鑸囨瑠闄?)
        uploaded = st.file_uploader("涓婂偝鏂?Excel", type=["xlsx"])
        if st.button("杓夊叆涓婂偝妾旀", disabled=uploaded is None):
            st.session_state["bundle"] = load_bundle_from_upload(uploaded)
            st.rerun()

        pwd = st.secrets.get("EDITOR_PASSWORD", "")
        ip = st.text_input("绶ㄨ集瀵嗙⒓", type="password")
        if st.button("鐧诲叆绶ㄨ集妯″紡"):
            st.session_state["editor_authed"] = bool(pwd and ip == pwd)
            st.rerun()
        if _is_editor() and st.button("鐧诲嚭绶ㄨ集妯″紡"):
            st.session_state["editor_authed"] = False
            st.rerun()
        st.caption("浣犵洰鍓嶅彲绶ㄨ集" if _is_editor() else "浣犵洰鍓嶇偤瑷鍙畝")

        st.markdown("---")
        st.subheader("绡╅伕")
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
            fk = st.text_input("闂滈嵉瀛楁悳灏嬶紙浠ｈ櫉/鍏徃/绨′粙/瀛愭澘濉?妯欑堡锛?, value=f0.get("keyword", ""))
            fs = st.multiselect("澶у垎椤?, all_sectors, default=[x for x in f0.get("sectors", []) if x in all_sectors])
            fss = st.multiselect("鏈€绲傚瓙鍒嗛", all_subsectors, default=[x for x in f0.get("subsectors", []) if x in all_subsectors])
            fe = st.multiselect("浜ゆ槗鎵€", all_exchanges, default=[x for x in f0.get("exchanges", []) if x in all_exchanges])
            ft = st.multiselect("AI 灏忓垎椤炴绫?, all_ai_tags, default=[x for x in f0.get("ai_tags", []) if x in all_ai_tags])
            fm = st.radio("妯欑堡鍖归厤妯″紡", ["浠讳竴鍛戒腑", "鍏ㄩ儴鍛戒腑"], index=0 if f0.get("tag_mode", "any") == "any" else 1, horizontal=True)
            c1, c2 = st.columns(2)
            ap = c1.form_submit_button("濂楃敤绡╅伕", use_container_width=True)
            cl = c2.form_submit_button("娓呯┖绡╅伕", use_container_width=True)
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
                    "tag_mode": "all" if fm == "鍏ㄩ儴鍛戒腑" else "any",
                }
                st.rerun()
        st.caption(f"绡╅伕寰岀瓎鏁革細{len(filtered_df):,}")

    tabs = st.tabs(["鑲＄エ娓呭柈", "鏅備簨浜嬩欢绛嗚", "鏅備簨涓婚", "瀛楀吀/瑕忓墖", "瑷畾/鍚屾"])

    with tabs[0]:
        k = st.text_input("闋佸収鎼滃皨锛堜唬铏?鍏徃/绨′粙/瀛愭澘濉?妯欑堡锛?, "")
        show = filtered_df.copy()
        if k.strip():
            show = show[show["search_blob"].str.contains(k.strip().lower(), na=False)]
        st.dataframe(
            show[["ticker", "company_name", "exchange", "sector", "subsector", "tags", "summary"]].rename(
                columns={"ticker": "浠ｈ櫉", "company_name": "鍏徃", "exchange": "浜ゆ槗鎵€", "sector": "澶у垎椤?, "subsector": "鏈€绲傚瓙鍒嗛", "tags": "AI妯欑堡", "summary": "绨′粙"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        export_df = show[["ticker", "company_name", "exchange", "sector", "subsector", "tags", "summary"]].copy()
        export_df["tv_symbol"] = export_df.apply(lambda r: _to_tv_symbol(r.get("exchange", ""), r.get("ticker", "")), axis=1)
        export_df = export_df[["tv_symbol", "ticker", "exchange", "company_name", "sector", "subsector", "tags", "summary"]].rename(
            columns={
                "tv_symbol": "tradingview_symbol",
                "tags": "ai_tags",
                "summary": "summary_zh_tw",
            }
        )
        cexp1, cexp2 = st.columns(2)
        cexp1.download_button(
            "匯出篩選結果 CSV（TradingView）",
            export_df.to_csv(index=False, encoding="utf-8-sig"),
            file_name=f"tv_watchlist_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        xbuf = BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as xw:
            export_df.to_excel(xw, index=False, sheet_name="watchlist")
        cexp2.download_button(
            "匯出篩選結果 Excel",
            xbuf.getvalue(),
            file_name=f"tv_watchlist_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
        t = st.selectbox("鍊嬭偂绛嗚浠ｈ櫉", sorted(show["ticker"].tolist())[:800] if len(show) else [])
        if t:
            notes = list_stock_notes(ticker=t)
            st.caption(f"{t} 鍊嬭偂绛嗚锛歿len(notes)} 绛?)
            if _is_editor():
                with st.form("add_stock_note_form"):
                    nd = st.date_input("鏃ユ湡", value=date.today())
                    nt = st.text_input("妯欓")
                    ndt = st.text_area("鍏у")
                    ni = st.selectbox("鏂瑰悜", ["鍒╁", "鍒╃┖", "涓€?])
                    if st.form_submit_button("鏂板鍊嬭偂绛嗚"):
                        add_stock_note(t, str(nd), nt, ndt, ni, 0.6, "", [])
                        st.rerun()
            if notes:
                st.dataframe(_safe_df_rows(notes, ["note_date", "impact", "event_title", "event_detail"]), use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("RSS v2 鏂拌仦鍖叆")
        default_kw = "缇庝紛鎴扮埈, 浠ヤ紛鎴扮埈, 涓澅琛濈獊, 鑽峰鑼叉捣宄? 鐭虫补, 澶╃劧姘? 鏈夎壊閲戝爆, 鍖栬偉, spacex, 鑵︽鎺ュ彛, 閲忓瓙, gtc"
        saved_kw = get_meta("rss_keywords", default_kw)
        saved_feeds = get_meta("rss_feeds", "\n".join(DEFAULT_RSS_FEEDS))
        kw_text = st.text_area("闂滈嵉瀛楋紙閫楄櫉鍒嗛殧锛?, value=saved_kw, key="rss_kw")
        feeds_text = st.text_area("RSS feeds锛堟瘡琛屼竴鍊嬶級", value=saved_feeds, height=120, key="rss_feeds")
        s1, s2 = st.columns(2)
        if s1.button("淇濆瓨 RSS 瑷畾", key="btn_save_rss_cfg"):
            set_meta("rss_keywords", kw_text.strip())
            set_meta("rss_feeds", feeds_text.strip())
            st.success("宸蹭繚瀛?RSS 瑷畾")
        if s2.button("淇濆瓨涓﹀悓姝ュ埌 GitHub", key="btn_save_rss_cfg_sync", disabled=not _is_editor()):
            set_meta("rss_keywords", kw_text.strip())
            set_meta("rss_feeds", feeds_text.strip())
            ok, msg = _sync_notes_to_github_now()
            if ok:
                st.success("宸蹭繚瀛樹甫鍚屾鍒?GitHub")
            else:
                st.error(f"鍚屾澶辨晽锛歿msg}")
        raw_keywords = [x.strip() for x in kw_text.split(",") if x.strip()]
        feeds = [x.strip() for x in feeds_text.splitlines() if x.strip()]
        lookback_for_refetch = int(st.session_state.get("rss_lookback", 48))
        max_items_for_refetch = int(st.session_state.get("rss_max_items", 30))
        custom = {r["keyword"]: r["expansions_list"] for r in list_keyword_synonyms() if r.get("enabled", True)}
        expanded, kmap = expand_keywords(raw_keywords, custom_synonyms=custom)
        st.caption(f"鎿村睍闂滈嵉瀛楋細{len(expanded)}锛堝師濮嬶細{len(raw_keywords)}锛?)

        unmapped = [k for k in raw_keywords if not kmap.get(k)]
        with st.expander("鏌ョ湅/绠＄悊鎿村睍闂滈嵉瀛?, expanded=False):
            if unmapped:
                st.warning(f"鏈槧灏勯棞閸靛瓧锛歿', '.join(unmapped)}")
                if st.button("AI 瑁滃叏鏈槧灏勯棞閸靛瓧涓﹀鍏ュ瓧鍏?):
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
                        st.success(f"宸茶鍏?{added} 鍊嬮棞閸靛瓧锛屼甫宸茶嚜鍕曢噸鎶?RSS銆?)
                        st.rerun()
                    else:
                        st.error(f"鐩墠鐒℃硶瑁滃叏锛歿last_err or '妯″瀷鏈洖鍌冲彲鐢ㄧ祼鏋?}")
            compact = st.checkbox("绮剧啊椤ず锛堟瘡瑭炲彧椤ず鍓?3 鍊嬶級", value=True)
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
        lookback = c1.number_input("鍥炵湅灏忔檪", min_value=6, max_value=240, value=48, step=6, key="rss_lookback")
        max_items = c2.number_input("鏈€澶氭柊鑱炴暩", min_value=10, max_value=300, value=30, step=10, key="rss_max_items")
        auto_ai = c3.checkbox("鍖叆寰岃嚜鍕?AI 鏄犲皠", value=True)
        auto_daily = st.checkbox("姣忔棩鑷嫊鎶?RSS锛堟瘡澶╅娆￠枊绔欒嚜鍕曞煼琛屼竴娆★級", value=(str(get_meta("rss_daily_auto_enabled", "0")) == "1"))
        set_meta("rss_daily_auto_enabled", "1" if auto_daily else "0")
        st.caption(f"姣忔棩鑷嫊锛歿'闁嬪暉' if auto_daily else '闂滈枆'} | 涓婃鍩疯锛歿get_meta('rss_daily_last_run_date', '-')}")

        ran, msg = _maybe_daily_auto_rss(raw_keywords, feeds, int(lookback), int(max_items), auto_ai, custom, df)
        if ran:
            st.success(f"浠婃棩鑷嫊鎶撳彇瀹屾垚锛歿msg}")

        d1, d2 = st.columns(2)
        if d1.button("鎶撳彇 RSS"):
            items = fetch_rss_items(raw_keywords, feeds, int(lookback), int(max_items), custom_synonyms=custom)
            st.session_state["rss_items"] = [x.__dict__ for x in items]

        rss_items = st.session_state.get("rss_items", [])
        if rss_items:
            rdf = pd.DataFrame(rss_items)
            st.dataframe(_safe_df_rows(rdf.to_dict("records"), ["published_at", "source", "title", "title_zh_tw", "matched_keywords", "matched_expanded", "link"]), use_container_width=True, hide_index=True)
            q = st.text_input("鍦ㄥ凡鎶撳彇鏂拌仦涓悳灏嬶紙涓嫳锛?, "")
            rdf2 = rdf.copy()
            if q.strip():
                ql = q.strip().lower()
                rdf2 = rdf2[rdf2["title"].astype(str).str.lower().str.contains(ql, na=False) | rdf2["title_zh_tw"].astype(str).str.lower().str.contains(ql, na=False)]
            picks = st.multiselect("閬告搰瑕佸尟鍏ョ殑鏂拌仦", rdf2["title"].tolist(), default=rdf2["title"].tolist()[: min(20, len(rdf2))])
            if d2.button("鍖叆閬镐腑鏂拌仦"):
                imported, merged, skipped, mapped = _import_rss_rows(rdf2, picks, auto_ai, df, show_progress=True)
                st.success(f"鏂板 {imported} / 鍚堜降 {merged} / 鐣ラ亷 {skipped} / AI 鍛戒腑 {mapped}")
                st.rerun()

        st.subheader("鐝炬湁鏅備簨浜嬩欢")
        events = list_macro_notes()
        st.dataframe(_safe_df_rows(events, ["id", "note_date", "impact", "event_title", "source_url"]), use_container_width=True, hide_index=True)
        if _is_editor() and events:
            eid = st.selectbox("閬告搰浜嬩欢 ID 濂楃敤 AI", [x["id"] for x in events])
            if st.button("灏嶉伕涓簨浠惰窇 AI 鏄犲皠"):
                row = next(x for x in events if x["id"] == eid)
                tc, hc = _run_ai_for_event(row, df)
                st.success(f"涓婚 {tc} 鍊嬶紝鍛戒腑 {hc} 绛?)
                st.rerun()

    with tabs[2]:
        st.subheader("鏅備簨涓婚锛堢啽搴︽帓鍚嶏級")
        tq = st.text_input("涓婚鎼滃皨", "")
        rows = list_themes_ranked()
        if tq.strip():
            rows = [x for x in rows if tq.strip().lower() in str(x["theme"]).lower()]
        st.dataframe(
            _safe_df_rows(rows, ["theme", "heat_score", "stock_count", "hit_count", "recent_date", "bull_count", "bear_count", "neutral_count"]).rename(
                columns={
                    "theme": "涓婚",
                    "heat_score": "鐔卞害鍒嗘暩",
                    "stock_count": "鑲＄エ鏁?,
                    "hit_count": "鍛戒腑鏁?,
                    "recent_date": "鏈€杩戞棩鏈?,
                    "bull_count": "鍒╁",
                    "bear_count": "鍒╃┖",
                    "neutral_count": "涓€?,
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("瑷堢畻瑾槑", expanded=False):
            st.markdown(
                "- `鑲＄エ鏁竊锛氫富椤屼笅鍘婚噸寰岀殑 ticker 鏁搁噺銆俓n"
                "- `鍛戒腑鏁竊锛氫富椤屽懡涓附绛嗘暩锛堝悓涓€ ticker 鍙洜涓嶅悓浜嬩欢閲嶈锛夈€俓n"
                "- 涓婚鍛戒腑鐢变簨浠?AI + 瀹堥杸瑕忓墖鍏卞悓姹哄畾銆俓n"
                "- 寤鸿鍏堟寜 `鏈€浣?confidence` 鍐嶈瀵熺祼鏋溿€?
            )
        if rows:
            picked = st.selectbox("鏌ョ湅涓婚", [x["theme"] for x in rows])
            c1, c2, c3 = st.columns(3)
            if c1.button("娓呯悊閬镐腑涓婚鍛戒腑", key="btn_clear_theme_hits", disabled=not _is_editor()):
                n = delete_event_theme_hits_by_theme(picked)
                st.success(f"宸叉竻鐞?{picked} 鍛戒腑锛歿n} 绛?)
                st.rerun()
            if c2.button("娓呯悊鍏ㄩ儴涓婚鍛戒腑", key="btn_clear_all_hits", disabled=not _is_editor()):
                n = clear_all_event_theme_hits()
                st.success(f"宸叉竻鐞嗗叏閮ㄥ懡涓細{n} 绛?)
                st.rerun()
            if c3.button("閲嶈窇鍏ㄩ儴浜嬩欢 AI", key="btn_rebuild_all_hits", disabled=not _is_editor()):
                _rebuild_all_theme_hits(df)
                st.rerun()
            if not _is_editor():
                st.caption("瑷妯″紡鍙煡鐪嬶紱閫插叆绶ㄨ集妯″紡寰屽彲浣跨敤娓呯悊/閲嶈窇鎸夐垥銆?)
            hits = list_event_theme_hits(theme_keyword=picked)
            st.dataframe(_safe_df_rows(hits, ["ticker", "theme", "impact", "confidence", "reason", "note_date", "event_title"]), use_container_width=True, hide_index=True)

    with tabs[3]:
        st.subheader("瀛楀吀 / 瑕忓墖")
        with st.form("kw_add_form"):
            k = st.text_input("闂滈嵉瀛?)
            ex = st.text_input("鎿村睍瑭烇紙鐢?| 鍒嗛殧锛?)
            en = st.checkbox("鍟熺敤", value=True)
            if st.form_submit_button("鏂板/鏇存柊闂滈嵉瀛?):
                upsert_keyword_synonym(k, [x.strip() for x in ex.split("|") if x.strip()], en)
                st.rerun()
        kws = list_keyword_synonyms()
        if kws:
            st.dataframe(_safe_df_rows(kws, ["keyword", "expansions", "enabled", "updated_at"]), use_container_width=True, hide_index=True)
            dk = st.selectbox("鍒櫎闂滈嵉瀛?, [""] + [x["keyword"] for x in kws])
            if st.button("鍒櫎闂滈嵉瀛?) and dk:
                delete_keyword_synonym(dk)
                st.rerun()

        with st.form("theme_rule_form"):
            raw = st.text_input("鍘熷涓婚")
            can = st.text_input("姝ｈ涓婚")
            en2 = st.checkbox("鍟熺敤瑕忓墖", value=True)
            if st.form_submit_button("鏂板/鏇存柊涓婚瑕忓墖"):
                upsert_theme_rule(raw, can, en2)
                st.rerun()
        rules = list_theme_rules()
        if rules:
            st.dataframe(_safe_df_rows(rules, ["raw_theme", "canonical_theme", "enabled", "updated_at"]), use_container_width=True, hide_index=True)

        cur = get_event_min_confidence()
        nv = st.slider("鏈€浣庝俊蹇冮柧鍊?, 0.0, 1.0, float(cur), 0.05)
        if st.button("淇濆瓨闁惧€?):
            set_event_min_confidence(float(nv))
            st.success("宸蹭繚瀛?)

    with tabs[4]:
        st.subheader("瑷畾 / 鍚屾")
        st.caption(f"鏈湴 DB锛歿DB_PATH}")
        payload = export_json_text()
        st.download_button("涓嬭級鏈湴鍌欎唤", payload.encode("utf-8"), "notes_export.json", "application/json")
        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        path = st.secrets.get("GITHUB_NOTES_PATH", "data/notes_export.json")
        if token and repo:
            c1, c2 = st.columns(2)
            if c1.button("鍚屾鍒?GitHub", disabled=not _is_editor()):
                upsert_file_content(token, repo, path, branch, payload, f"Update notes backup {datetime.now():%Y-%m-%d %H:%M:%S}")
                st.success("宸插悓姝?)
            if c2.button("寰?GitHub 閭勫師", disabled=not _is_editor()):
                txt, _ = get_file_content(token, repo, path, branch)
                if txt.strip():
                    import_json_text(txt, replace=True)
                    st.success("宸查倓鍘?)
                    st.rerun()
        st.caption(f"涓婃鍚屾锛歿get_meta('last_sync_to_github', '-')}")


if __name__ == "__main__":
    main()

