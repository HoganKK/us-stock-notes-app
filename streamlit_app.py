from __future__ import annotations

import re
import json
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
    list_themes_summary,
    macro_note_exists,
    replace_event_theme_hits,
    set_event_min_confidence,
    set_meta,
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


def _theme_rule_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for r in list_theme_rules():
        if r.get("enabled", True):
            m[str(r["raw_theme"]).strip().lower()] = str(r["canonical_theme"]).strip()
    return m


def _suggest_keyword_expansions_ai(term: str, current_expanded: list[str]) -> list[str]:
    api_key = st.secrets.get("KIMI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        return []
    base_url = st.secrets.get("KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    model = st.secrets.get("KIMI_MODEL", "kimi-for-coding")
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = (
        "你是金融新聞檢索助手。"
        "請根據輸入主題詞，輸出 6-12 個最有檢索價值的擴展關鍵字（中英混合，可包含同義詞/常見縮寫/相關專有名詞）。"
        "僅輸出 JSON 陣列字串，不要任何額外文字。"
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
            return []
        out = []
        seen = set(x.lower() for x in current_expanded)
        for x in arr:
            sx = str(x).strip()
            if not sx:
                continue
            if sx.lower() in seen:
                continue
            seen.add(sx.lower())
            out.append(sx)
        return out[:20]
    except Exception:
        return []


def _run_ai_for_event(event_row: dict, df: pd.DataFrame) -> tuple[int, int]:
    api_key = st.secrets.get("KIMI_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("缺少 KIMI_API_KEY 或 OPENAI_API_KEY（請到 Streamlit secrets 設定）")

    base_url = st.secrets.get("KIMI_BASE_URL", "https://api.kimi.com/coding/v1")
    model = st.secrets.get("KIMI_MODEL", "kimi-for-coding")

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
    for h in res.hits:
        conf = float(h.get("confidence", 0.0) or 0.0)
        if conf < min_conf:
            continue
        raw_theme = str(h.get("theme", "")).strip()
        h["theme"] = rule_map.get(raw_theme.lower(), raw_theme)
        refined.append(h)

    replace_event_theme_hits(int(event_row["id"]), refined)
    return len(set([x["theme"] for x in refined])), len(refined)


def _apply_filters(df: pd.DataFrame, f: dict) -> pd.DataFrame:
    out = df.copy()
    kw = str(f.get("keyword", "")).strip().lower()
    if kw:
        out = out[out["search_blob"].str.contains(kw, na=False)]

    sectors = f.get("sectors", []) or []
    if sectors:
        out = out[out["sector"].isin(sectors)]

    subsectors = f.get("subsectors", []) or []
    if subsectors:
        out = out[out["subsector"].isin(subsectors)]

    exchanges = f.get("exchanges", []) or []
    if exchanges:
        out = out[out["exchange"].isin(exchanges)]

    tags = f.get("ai_tags", []) or []
    if tags:
        mode = str(f.get("tag_mode", "any"))
        if mode == "all":
            out = out[out["tags"].astype(str).apply(lambda x: all(t in _split_tags(x) for t in tags))]
        else:
            out = out[out["tags"].astype(str).apply(lambda x: any(t in _split_tags(x) for t in tags))]

    return out


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
        if st.button("載入上傳檔案", key="btn_load_upload", disabled=uploaded is None):
            st.session_state["bundle"] = load_bundle_from_upload(uploaded)
            st.rerun()

        pwd = st.secrets.get("EDITOR_PASSWORD", "")
        ip = st.text_input("編輯密碼", type="password")
        if st.button("登入編輯模式", key="btn_login_editor"):
            st.session_state["editor_authed"] = bool(pwd and ip == pwd)
            st.rerun()
        if _is_editor():
            if st.button("登出編輯模式", key="btn_logout_editor"):
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
            apply_clicked = c1.form_submit_button("套用篩選", use_container_width=True)
            clear_clicked = c2.form_submit_button("清空篩選", use_container_width=True)

            if clear_clicked:
                st.session_state["filters"] = {
                    "keyword": "",
                    "sectors": [],
                    "subsectors": [],
                    "exchanges": [],
                    "ai_tags": [],
                    "tag_mode": "any",
                }
                st.rerun()
            if apply_clicked:
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
        k = st.text_input("頁內關鍵字搜尋（代號/公司/簡介/子板塊/標籤）", "")
        show = filtered_df.copy()
        if k.strip():
            show = show[show["search_blob"].str.contains(k.strip().lower(), na=False)]
        st.dataframe(
            show[["ticker", "company_name", "exchange", "sector", "subsector", "tags", "summary"]].rename(
                columns={
                    "ticker": "代號",
                    "company_name": "公司",
                    "exchange": "交易所",
                    "sector": "大分類",
                    "subsector": "最終子分類",
                    "tags": "AI 標籤",
                    "summary": "簡介",
                }
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
                st.dataframe(
                    _safe_df_rows(notes, ["note_date", "impact", "event_title", "event_detail"]),
                    use_container_width=True,
                    hide_index=True,
                )

    with tabs[1]:
        st.subheader("RSS v2 新聞匯入")
        default_kw = "美伊戰爭, 以伊戰爭, 中東衝突, 荷姆茲海峽, 石油, 天然氣, 有色金屬, 化肥, spacex, 腦機接口, 量子, gtc"
        kw_text = st.text_area("關鍵字（逗號分隔）", value=default_kw, key="rss_kw")
        feeds_text = st.text_area("RSS feeds（每行一個）", value="\n".join(DEFAULT_RSS_FEEDS), height=120, key="rss_feeds")

        raw_keywords = [x.strip() for x in kw_text.split(",") if x.strip()]
        custom = {r["keyword"]: r["expansions_list"] for r in list_keyword_synonyms() if r.get("enabled", True)}
        expanded, kmap = expand_keywords(raw_keywords, custom_synonyms=custom)
        st.caption(f"擴展關鍵字：{len(expanded)}（原始：{len(raw_keywords)}）")

        unmapped = [k for k in raw_keywords if not kmap.get(k)]
        with st.expander("查看/管理擴展關鍵字", expanded=False):
            if unmapped:
                st.warning(f"未映射關鍵字：{', '.join(unmapped)}")
                if st.button("AI 補全未映射關鍵字並寫入字典", key="btn_ai_expand_unmapped"):
                    added = 0
                    for uk in unmapped[:10]:
                        sug = _suggest_keyword_expansions_ai(uk, expanded)
                        if sug:
                            upsert_keyword_synonym(uk, sug, True)
                            added += 1
                    if added > 0:
                        st.success(f"已補全 {added} 個關鍵字，請再按一次『抓取 RSS』。")
                        st.rerun()
                    else:
                        st.info("目前無法補全（可能未設 API key，或模型暫時未回應）。")

            compact = st.checkbox("精簡顯示（每詞只顯示前 3 個）", value=True, key="kw_compact_view")
            for kx, vals in kmap.items():
                if not vals:
                    continue
                if compact:
                    show_vals = vals[:3]
                    more = len(vals) - len(show_vals)
                    st.caption(f"{kx} -> {', '.join(show_vals)}" + (f" ... (+{more})" if more > 0 else ""))
                else:
                    st.caption(f"{kx} -> {', '.join(vals)}")

        c1, c2, c3 = st.columns(3)
        lookback = c1.number_input("回看小時", min_value=6, max_value=240, value=72, step=6, key="rss_lookback")
        max_items = c2.number_input("最多新聞數", min_value=10, max_value=300, value=80, step=10, key="rss_max_items")
        auto_ai = c3.checkbox("匯入後自動 AI 映射", value=True, key="rss_auto_ai")

        d1, d2 = st.columns(2)
        if d1.button("抓取 RSS", key="btn_fetch_rss"):
            items = fetch_rss_items(
                keywords=raw_keywords,
                feeds=[x.strip() for x in feeds_text.splitlines() if x.strip()],
                lookback_hours=int(lookback),
                max_items=int(max_items),
                custom_synonyms=custom,
            )
            st.session_state["rss_items"] = [x.__dict__ for x in items]

        rss_items = st.session_state.get("rss_items", [])
        if rss_items:
            rdf = pd.DataFrame(rss_items)
            st.dataframe(
                _safe_df_rows(rdf.to_dict("records"), ["published_at", "source", "title", "title_zh_tw", "matched_keywords", "matched_expanded", "link"]),
                use_container_width=True,
                hide_index=True,
            )

            q = st.text_input("在已抓取新聞中搜尋（中英）", "", key="rss_search")
            rdf2 = rdf.copy()
            if q.strip():
                ql = q.strip().lower()
                rdf2 = rdf2[
                    rdf2["title"].astype(str).str.lower().str.contains(ql, na=False)
                    | rdf2["title_zh_tw"].astype(str).str.lower().str.contains(ql, na=False)
                ]

            picks = st.multiselect(
                "選擇要匯入的新聞",
                rdf2["title"].tolist(),
                default=rdf2["title"].tolist()[: min(20, len(rdf2))],
                key="rss_pick_titles",
            )

            if d2.button("匯入選中新聞", key="btn_import_rss"):
                existing = list_macro_notes()
                imported = merged = skipped = mapped = 0
                for _, r in rdf2.iterrows():
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
                        from notes_store import update_macro_note

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

                st.success(f"新增 {imported} / 合併 {merged} / 略過 {skipped} / AI 命中 {mapped}")
                st.rerun()

        st.subheader("現有時事事件")
        events = list_macro_notes()
        st.dataframe(
            _safe_df_rows(events, ["id", "note_date", "impact", "event_title", "source_url"]),
            use_container_width=True,
            hide_index=True,
        )

        if _is_editor() and events:
            eid = st.selectbox("選擇事件 ID 套用 AI", [x["id"] for x in events], key="event_pick_for_ai")
            if st.button("對選中事件跑 AI 映射", key="btn_event_apply_ai"):
                row = next(x for x in events if x["id"] == eid)
                tc, hc = _run_ai_for_event(row, df)
                st.success(f"主題 {tc} 個，命中 {hc} 筆")
                st.rerun()

    with tabs[2]:
        st.subheader("時事主題")
        tq = st.text_input("主題搜尋", "", key="theme_search")
        rows = list_themes_summary()
        if tq.strip():
            rows = [x for x in rows if tq.strip().lower() in str(x["theme"]).lower()]
        st.dataframe(
            _safe_df_rows(rows, ["theme", "stock_count", "hit_count"]).rename(
                columns={"theme": "主題", "stock_count": "股票數", "hit_count": "命中數"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        if rows:
            picked = st.selectbox("查看主題", [x["theme"] for x in rows], key="theme_pick")
            hits = list_event_theme_hits(theme_keyword=picked)
            st.dataframe(
                _safe_df_rows(hits, ["ticker", "theme", "impact", "confidence", "reason", "note_date", "event_title"]),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        st.subheader("字典 / 規則")

        with st.form("kw_add_form"):
            k = st.text_input("關鍵字")
            ex = st.text_input("擴展詞（用 | 分隔）")
            en = st.checkbox("啟用", value=True)
            if st.form_submit_button("新增/更新關鍵字字典"):
                upsert_keyword_synonym(k, [x.strip() for x in ex.split("|") if x.strip()], en)
                st.rerun()

        kws = list_keyword_synonyms()
        if kws:
            st.dataframe(
                _safe_df_rows(kws, ["keyword", "expansions", "enabled", "updated_at"]),
                use_container_width=True,
                hide_index=True,
            )
            dk = st.selectbox("刪除關鍵字", [""] + [x["keyword"] for x in kws], key="kw_delete_pick")
            if st.button("刪除關鍵字", key="btn_kw_delete") and dk:
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
            st.dataframe(
                _safe_df_rows(rules, ["raw_theme", "canonical_theme", "enabled", "updated_at"]),
                use_container_width=True,
                hide_index=True,
            )

        cur = get_event_min_confidence()
        nv = st.slider("最低信心閾值", 0.0, 1.0, float(cur), 0.05, key="conf_slider")
        if st.button("保存閾值", key="btn_save_conf"):
            set_event_min_confidence(float(nv))
            st.success("已保存")

    with tabs[4]:
        st.subheader("設定 / 同步")
        st.caption(f"本地 DB：{DB_PATH}")

        payload = export_json_text()
        st.download_button(
            "下載本地備份",
            payload.encode("utf-8"),
            "notes_export.json",
            "application/json",
            key="btn_download_backup",
        )

        token = st.secrets.get("GITHUB_TOKEN", "")
        repo = st.secrets.get("GITHUB_REPO", "")
        branch = st.secrets.get("GITHUB_BRANCH", "main")
        path = st.secrets.get("GITHUB_NOTES_PATH", "data/notes_export.json")

        if token and repo:
            c1, c2 = st.columns(2)
            if c1.button("同步到 GitHub", key="btn_sync_github", disabled=not _is_editor()):
                upsert_file_content(
                    token=token,
                    repo=repo,
                    path=path,
                    branch=branch,
                    content=payload,
                    message=f"Update notes backup {datetime.now():%Y-%m-%d %H:%M:%S}",
                )
                st.success("已同步")
            if c2.button("從 GitHub 還原", key="btn_restore_github", disabled=not _is_editor()):
                txt, _ = get_file_content(token=token, repo=repo, path=path, branch=branch)
                if txt.strip():
                    import_json_text(txt, replace=True)
                    st.success("已還原")
                    st.rerun()

        st.caption(f"上次同步：{get_meta('last_sync_to_github', '-')}")


if __name__ == "__main__":
    main()
