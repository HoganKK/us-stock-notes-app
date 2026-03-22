from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from data_loader import DEFAULT_INPUT_PATH, DataBundle, load_bundle_from_path, load_bundle_from_upload
from github_sync import get_file_content, upsert_file_content
from notes_store import (
    DB_PATH,
    add_macro_note,
    add_stock_note,
    delete_macro_note,
    delete_stock_note,
    export_json_text,
    get_meta,
    import_json_text,
    init_db,
    list_macro_notes,
    list_related_macro_notes,
    list_stock_notes,
    set_meta,
    update_macro_note,
    update_stock_note,
)


st.set_page_config(page_title="美股清單與筆記", page_icon="📈", layout="wide")


@st.cache_data(show_spinner=False)
def _load_default_bundle(path_str: str) -> DataBundle:
    return load_bundle_from_path(Path(path_str))


def _ensure_data_loaded() -> None:
    if "bundle" not in st.session_state:
        st.session_state["bundle"] = _load_default_bundle(str(DEFAULT_INPUT_PATH))
    b: DataBundle = st.session_state["bundle"]
    set_meta("data_source_name", b.source_name)
    set_meta("data_source_hash", b.source_hash)
    set_meta("data_source_sheet", b.source_sheet)
    set_meta("data_loaded_at", b.loaded_at)


def _is_editor() -> bool:
    return bool(st.session_state.get("editor_authed", False))


def _require_editor() -> bool:
    if not _is_editor():
        st.warning("此操作需要編輯權限。")
        return False
    return True


def _parse_tags(x: str) -> list[str]:
    return [t.strip() for t in str(x or "").replace("、", "|").split("|") if t.strip()]


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
    # Persist filter state explicitly so reruns won't reset selections.
    if "flt_keyword" not in st.session_state:
        st.session_state["flt_keyword"] = ""
    if "flt_sectors" not in st.session_state:
        st.session_state["flt_sectors"] = []
    if "flt_subsectors" not in st.session_state:
        st.session_state["flt_subsectors"] = []
    if "flt_exchanges" not in st.session_state:
        st.session_state["flt_exchanges"] = []
    if "flt_tags" not in st.session_state:
        st.session_state["flt_tags"] = []
    if "flt_tag_mode" not in st.session_state:
        st.session_state["flt_tag_mode"] = "任一命中"

    key = st.sidebar.text_input(
        "關鍵字搜尋（代號/公司/簡介/子板塊/標籤）",
        value=st.session_state["flt_keyword"],
        key="flt_keyword",
    )
    if st.sidebar.button("清空全部篩選"):
        st.session_state["flt_keyword"] = ""
        st.session_state["flt_sectors"] = []
        st.session_state["flt_subsectors"] = []
        st.session_state["flt_exchanges"] = []
        st.session_state["flt_tags"] = []
        st.session_state["flt_tag_mode"] = "任一命中"
        st.rerun()
    sectors = sorted([x for x in df["sector"].dropna().astype(str).unique().tolist() if x.strip()])
    subsectors = sorted([x for x in df["subsector"].dropna().astype(str).unique().tolist() if x.strip()])
    exchanges = sorted([x for x in df["exchange"].dropna().astype(str).unique().tolist() if x.strip()])

    selected_sectors = st.sidebar.multiselect(
        "大分類",
        sectors,
        default=[x for x in st.session_state["flt_sectors"] if x in sectors],
        key="flt_sectors",
    )
    selected_subsectors = st.sidebar.multiselect(
        "最終子分類",
        subsectors,
        default=[x for x in st.session_state["flt_subsectors"] if x in subsectors],
        key="flt_subsectors",
    )
    selected_exchanges = st.sidebar.multiselect(
        "交易所",
        exchanges,
        default=[x for x in st.session_state["flt_exchanges"] if x in exchanges],
        key="flt_exchanges",
    )

    all_tags = set()
    for tags in df["tags_list"].tolist():
        for t in tags:
            all_tags.add(t)
    tag_options = sorted(all_tags)
    selected_tags = st.sidebar.multiselect(
        "AI 小分類標籤",
        tag_options,
        default=[x for x in st.session_state["flt_tags"] if x in tag_options],
        key="flt_tags",
    )
    tag_mode = st.sidebar.radio(
        "標籤匹配模式",
        ["任一命中", "全部命中"],
        horizontal=True,
        key="flt_tag_mode",
    )

    filtered = df.copy()
    if key.strip():
        k = key.strip().lower()
        filtered = filtered[filtered["search_blob"].str.contains(k, na=False)]
    if selected_sectors:
        filtered = filtered[filtered["sector"].isin(selected_sectors)]
    if selected_subsectors:
        filtered = filtered[filtered["subsector"].isin(selected_subsectors)]
    if selected_exchanges:
        filtered = filtered[filtered["exchange"].isin(selected_exchanges)]
    if selected_tags:
        if tag_mode == "任一命中":
            filtered = filtered[filtered["tags_list"].map(lambda xs: any(t in xs for t in selected_tags))]
        else:
            filtered = filtered[filtered["tags_list"].map(lambda xs: all(t in xs for t in selected_tags))]

    st.sidebar.markdown(f"**篩選後筆數：{len(filtered):,}**")
    return filtered.reset_index(drop=True)


def _render_subtheme_search(df: pd.DataFrame) -> None:
    st.subheader("子板塊專用搜尋")
    q = st.text_input("搜尋關鍵字（匹配 最終子分類 + AI小分類標籤）", value="", key="subtheme_query")
    if not q.strip():
        st.caption("輸入關鍵字可快速定位主題與股票。")
        return
    qv = q.strip().lower()
    hit = df[
        df["subsector"].astype(str).str.lower().str.contains(qv, na=False)
        | df["tags"].astype(str).str.lower().str.contains(qv, na=False)
    ].copy()
    st.write(f"命中股票數：**{len(hit):,}**")
    if hit.empty:
        return
    top = (
        hit.groupby("subsector", dropna=False)["ticker"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.dataframe(top, use_container_width=True, hide_index=True)
    with c2:
        st.dataframe(
            hit[["ticker", "company_name", "subsector", "tags"]].head(120),
            use_container_width=True,
            hide_index=True,
        )


def _render_stock_notes_panel(ticker: str, sector: str, subsector: str) -> None:
    st.markdown("#### 個股筆記")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        f_impact = st.selectbox("影響方向", ["全部", "利多", "利空", "中性"], key=f"stock_note_impact_{ticker}")
    with c2:
        f_start = st.date_input("起始日期", value=None, key=f"stock_note_start_{ticker}")
    with c3:
        f_end = st.date_input("結束日期", value=None, key=f"stock_note_end_{ticker}")
    with c4:
        f_tag = st.text_input("標籤關鍵字", value="", key=f"stock_note_tag_{ticker}")

    notes = list_stock_notes(
        ticker=ticker,
        impact=f_impact,
        start_date=str(f_start) if isinstance(f_start, date) else None,
        end_date=str(f_end) if isinstance(f_end, date) else None,
        tag_keyword=f_tag.strip() or None,
    )
    st.caption(f"個股筆記：{len(notes)} 筆")

    if _is_editor():
        with st.expander("新增個股筆記", expanded=False):
            with st.form(f"add_stock_note_{ticker}"):
                n_date = st.date_input("日期", value=date.today())
                n_title = st.text_input("事件標題")
                n_detail = st.text_area("事件內容")
                n_impact = st.selectbox("影響", ["利多", "利空", "中性"])
                n_conf = st.slider("信心分數", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
                n_url = st.text_input("來源連結")
                n_tags = st.text_input("標籤（用 | 分隔）")
                ok = st.form_submit_button("新增")
                if ok:
                    add_stock_note(
                        ticker=ticker,
                        note_date=str(n_date),
                        event_title=n_title,
                        event_detail=n_detail,
                        impact=n_impact,
                        confidence=float(n_conf),
                        source_url=n_url,
                        tags=_parse_tags(n_tags),
                    )
                    st.success("已新增個股筆記")
                    st.rerun()

    for n in notes:
        title = f"[{n.get('note_date','')}] {n.get('impact','')} - {n.get('event_title','')}"
        with st.expander(title, expanded=False):
            st.write(n.get("event_detail", ""))
            st.caption(
                f"信心: {n.get('confidence', 0)} | 標籤: {n.get('tags','')} | 更新: {n.get('updated_at','')}"
            )
            if n.get("source_url"):
                st.markdown(f"[來源連結]({n['source_url']})")

            if _is_editor():
                with st.form(f"edit_stock_{n['id']}"):
                    e_date = st.text_input("日期", value=str(n.get("note_date", "")))
                    e_title = st.text_input("事件標題", value=str(n.get("event_title", "")))
                    e_detail = st.text_area("事件內容", value=str(n.get("event_detail", "")))
                    e_impact = st.selectbox(
                        "影響",
                        ["利多", "利空", "中性"],
                        index=["利多", "利空", "中性"].index(str(n.get("impact", "中性"))),
                    )
                    e_conf = st.slider(
                        "信心分數",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(n.get("confidence", 0.6)),
                        step=0.05,
                    )
                    e_url = st.text_input("來源連結", value=str(n.get("source_url", "")))
                    e_tags = st.text_input("標籤（|分隔）", value=str(n.get("tags", "")))
                    c1, c2 = st.columns(2)
                    save = c1.form_submit_button("儲存")
                    remove = c2.form_submit_button("刪除")
                    if save:
                        update_stock_note(
                            note_id=int(n["id"]),
                            note_date=e_date,
                            event_title=e_title,
                            event_detail=e_detail,
                            impact=e_impact,
                            confidence=float(e_conf),
                            source_url=e_url,
                            tags=_parse_tags(e_tags),
                        )
                        st.success("已更新")
                        st.rerun()
                    if remove:
                        delete_stock_note(int(n["id"]))
                        st.success("已刪除")
                        st.rerun()

    st.markdown("#### 關聯時事")
    macro = list_related_macro_notes(ticker=ticker, sector=sector, subsector=subsector)
    st.caption(f"關聯時事：{len(macro)} 筆")
    for m in macro[:80]:
        with st.expander(f"[{m.get('note_date','')}] {m.get('impact','')} - {m.get('event_title','')}"):
            st.write(m.get("event_detail", ""))
            st.caption(
                f"板塊: {m.get('affected_subsectors','')} | 大分類: {m.get('affected_sectors','')} | 代號: {m.get('affected_tickers','')}"
            )
            if m.get("source_url"):
                st.markdown(f"[來源連結]({m['source_url']})")


def _render_macro_notes_page(df: pd.DataFrame) -> None:
    st.subheader("時事事件筆記")
    impacts = ["全部", "利多", "利空", "中性"]
    c1, c2, c3 = st.columns(3)
    with c1:
        f_impact = st.selectbox("影響方向", impacts, key="macro_filter_impact")
    with c2:
        f_start = st.date_input("起始日期", value=None, key="macro_filter_start")
    with c3:
        f_end = st.date_input("結束日期", value=None, key="macro_filter_end")

    macro_notes = list_macro_notes(
        impact=f_impact,
        start_date=str(f_start) if isinstance(f_start, date) else None,
        end_date=str(f_end) if isinstance(f_end, date) else None,
    )
    st.caption(f"事件筆記：{len(macro_notes)} 筆")

    if _is_editor():
        sectors = sorted([x for x in df["sector"].astype(str).unique().tolist() if x.strip()])
        subsectors = sorted([x for x in df["subsector"].astype(str).unique().tolist() if x.strip()])
        with st.expander("新增時事事件", expanded=False):
            with st.form("add_macro_note"):
                n_date = st.date_input("日期", value=date.today(), key="add_macro_date")
                n_title = st.text_input("事件標題", key="add_macro_title")
                n_detail = st.text_area("事件內容", key="add_macro_detail")
                n_impact = st.selectbox("影響", ["利多", "利空", "中性"], key="add_macro_impact")
                n_sectors = st.multiselect("影響大分類", sectors, key="add_macro_sectors")
                n_subsectors = st.multiselect("影響子板塊", subsectors, key="add_macro_subsectors")
                n_tickers = st.text_input("影響代號（逗號分隔）", key="add_macro_tickers")
                n_url = st.text_input("來源連結", key="add_macro_url")
                ok = st.form_submit_button("新增時事")
                if ok:
                    add_macro_note(
                        note_date=str(n_date),
                        event_title=n_title,
                        event_detail=n_detail,
                        affected_sectors=n_sectors,
                        affected_subsectors=n_subsectors,
                        affected_tickers=[x.strip().upper() for x in n_tickers.split(",") if x.strip()],
                        impact=n_impact,
                        source_url=n_url,
                    )
                    st.success("已新增時事")
                    st.rerun()

    for m in macro_notes:
        with st.expander(f"[{m.get('note_date','')}] {m.get('impact','')} - {m.get('event_title','')}"):
            st.write(m.get("event_detail", ""))
            st.caption(
                f"影響大分類: {m.get('affected_sectors','')} | 影響子板塊: {m.get('affected_subsectors','')} | 影響代號: {m.get('affected_tickers','')}"
            )
            if m.get("source_url"):
                st.markdown(f"[來源連結]({m['source_url']})")
            if _is_editor():
                with st.form(f"edit_macro_{m['id']}"):
                    e_date = st.text_input("日期", value=str(m.get("note_date", "")))
                    e_title = st.text_input("事件標題", value=str(m.get("event_title", "")))
                    e_detail = st.text_area("事件內容", value=str(m.get("event_detail", "")))
                    e_impact = st.selectbox(
                        "影響",
                        ["利多", "利空", "中性"],
                        index=["利多", "利空", "中性"].index(str(m.get("impact", "中性"))),
                    )
                    e_sectors = st.text_input("影響大分類（|分隔）", value=str(m.get("affected_sectors", "")))
                    e_sub = st.text_input("影響子板塊（|分隔）", value=str(m.get("affected_subsectors", "")))
                    e_tic = st.text_input("影響代號（|分隔）", value=str(m.get("affected_tickers", "")))
                    e_url = st.text_input("來源連結", value=str(m.get("source_url", "")))
                    c1, c2 = st.columns(2)
                    save = c1.form_submit_button("儲存")
                    remove = c2.form_submit_button("刪除")
                    if save:
                        update_macro_note(
                            note_id=int(m["id"]),
                            note_date=e_date,
                            event_title=e_title,
                            event_detail=e_detail,
                            affected_sectors=[x.strip() for x in e_sectors.split("|") if x.strip()],
                            affected_subsectors=[x.strip() for x in e_sub.split("|") if x.strip()],
                            affected_tickers=[x.strip().upper() for x in e_tic.split("|") if x.strip()],
                            impact=e_impact,
                            source_url=e_url,
                        )
                        st.success("已更新")
                        st.rerun()
                    if remove:
                        delete_macro_note(int(m["id"]))
                        st.success("已刪除")
                        st.rerun()


def _render_sync_panel() -> None:
    st.subheader("備份與同步")
    st.caption(f"本地 DB：{DB_PATH}")
    local_json = export_json_text()
    st.download_button(
        "下載本地筆記備份 JSON",
        data=local_json.encode("utf-8"),
        file_name="notes_export.json",
        mime="application/json",
    )

    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    branch = st.secrets.get("GITHUB_BRANCH", "main")
    path = st.secrets.get("GITHUB_NOTES_PATH", "data/notes_export.json")

    st.write(f"GitHub 目標：`{repo}` / `{branch}` / `{path}`")
    if not token or not repo:
        st.info("尚未設定 GitHub secrets（GITHUB_TOKEN, GITHUB_REPO）。")
        return

    c1, c2 = st.columns(2)
    if c1.button("同步到 GitHub", disabled=not _is_editor()):
        if _require_editor():
            try:
                upsert_file_content(
                    token=token,
                    repo=repo,
                    path=path,
                    branch=branch,
                    text=local_json,
                    commit_message=f"Update notes backup {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                )
                set_meta("last_sync_to_github", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                st.success("已同步到 GitHub。")
            except Exception as e:
                st.error(f"同步失敗：{e}")

    if c2.button("從 GitHub 還原", disabled=not _is_editor()):
        if _require_editor():
            try:
                text, _sha = get_file_content(token=token, repo=repo, path=path, branch=branch)
                if not text.strip():
                    st.warning("GitHub 上尚無備份檔。")
                else:
                    import_json_text(text, replace=True)
                    set_meta("last_restore_from_github", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    st.success("已從 GitHub 還原。")
                    st.rerun()
            except Exception as e:
                st.error(f"還原失敗：{e}")

    st.caption(f"上次同步：{get_meta('last_sync_to_github', '-')}")
    st.caption(f"上次還原：{get_meta('last_restore_from_github', '-')}")


def main() -> None:
    init_db()
    _ensure_data_loaded()
    bundle: DataBundle = st.session_state["bundle"]
    df = bundle.schema_df.copy()

    st.title("📈 美股清單與筆記系統")
    st.caption(
        f"資料來源：{bundle.source_name} | 工作表：{bundle.source_sheet} | Hash：`{bundle.source_hash}` | 載入時間：{bundle.loaded_at}"
    )

    with st.sidebar:
        st.header("資料與權限")
        uploaded = st.file_uploader("上傳新 Excel（立即覆蓋本次會話資料）", type=["xlsx"])
        if st.button("載入上傳檔案", disabled=uploaded is None):
            st.session_state["bundle"] = load_bundle_from_upload(uploaded)
            st.success("已載入新檔案。")
            st.rerun()

        editor_pwd = st.secrets.get("EDITOR_PASSWORD", "")
        if editor_pwd:
            pwd = st.text_input("編輯密碼", type="password")
            if st.button("登入編輯模式"):
                if pwd == editor_pwd:
                    st.session_state["editor_authed"] = True
                    st.success("已進入編輯模式")
                else:
                    st.session_state["editor_authed"] = False
                    st.error("密碼錯誤")
        else:
            st.info("未設定 EDITOR_PASSWORD，當前為只讀模式。")
        if _is_editor():
            if st.button("登出編輯模式"):
                st.session_state["editor_authed"] = False
                st.rerun()
            st.success("你目前可編輯")
        else:
            st.caption("你目前為訪客只讀")

    filtered = _filter_df(df)

    tab1, tab2, tab3, tab4 = st.tabs(["股票清單", "個股詳情", "時事筆記", "設定/同步"])

    with tab1:
        _render_subtheme_search(filtered)
        st.subheader("篩選結果")
        display = filtered[["ticker", "company_name", "exchange", "sector", "subsector", "tags", "summary"]].copy()
        display = display.rename(
            columns={
                "ticker": "代號",
                "company_name": "公司",
                "exchange": "交易所",
                "sector": "大分類",
                "subsector": "最終子分類",
                "tags": "AI小分類標籤",
                "summary": "公司簡介",
            }
        )
        page_size = st.selectbox("每頁筆數", [30, 50, 100, 200], index=1)
        total_rows = len(display)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)
        page_no = st.number_input("頁碼", min_value=1, max_value=total_pages, value=1, step=1)
        start = (page_no - 1) * page_size
        end = start + page_size
        page_df = display.iloc[start:end].copy()
        st.dataframe(page_df, use_container_width=True, hide_index=True)
        st.caption(f"第 {page_no}/{total_pages} 頁，共 {total_rows:,} 筆")

        csv_data = display.to_csv(index=False).encode("utf-8-sig")
        st.download_button("匯出目前篩選結果 CSV", data=csv_data, file_name="filtered_stocks.csv", mime="text/csv")

        tickers = filtered["ticker"].tolist()
        if tickers:
            selected = st.selectbox("快速選股（跳到個股詳情）", tickers)
            if st.button("開啟個股詳情"):
                st.session_state["selected_ticker"] = selected
                st.rerun()

    with tab2:
        ticker_candidates = filtered["ticker"].tolist() or df["ticker"].tolist()
        default_ticker = st.session_state.get("selected_ticker", ticker_candidates[0] if ticker_candidates else "")
        if not ticker_candidates:
            st.warning("目前沒有可顯示股票。")
        else:
            ticker = st.selectbox("選擇個股", ticker_candidates, index=max(0, ticker_candidates.index(default_ticker) if default_ticker in ticker_candidates else 0))
            row = df[df["ticker"] == ticker].head(1).iloc[0]
            st.markdown(f"### {row['ticker']} - {row['company_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("交易所", row["exchange"])
            c2.metric("大分類", row["sector"])
            c3.metric("最終子分類", row["subsector"])
            st.write(row["summary"])
            st.caption(f"標籤：{row['tags']}")
            _render_stock_notes_panel(ticker=row["ticker"], sector=row["sector"], subsector=row["subsector"])

    with tab3:
        _render_macro_notes_page(df)

    with tab4:
        _render_sync_panel()


if __name__ == "__main__":
    main()
