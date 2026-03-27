#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from build_second_opinion_preview import _clean_preview_text, _norm_text, _tags_to_display, _truthy


DEFAULT_BASE = Path("E:/CHATGPT_CODEX/STOCK_NOTES/stock_fetcher/output/us_stocks_investable_themes_zh_tw_fast.xlsx")
DEFAULT_REVIEW = Path("E:/CHATGPT_CODEX/STOCK_NOTES/stock_fetcher/output/us_stocks_ai_review.xlsx")
DEFAULT_OUTPUT = Path("E:/CHATGPT_CODEX/STOCK_NOTES/stock_fetcher/output/us_stocks_investable_themes_zh_tw_hybrid_search_preview.xlsx")
TARGET_SHEETS = ["全部股票", "可投資清單", "排除清單"]
OPTICAL_COMM_KEYWORDS = ("光通訊", "光學通訊", "光纖通訊")


def _split_tag_text(text: Any) -> list[str]:
    parts = []
    seen = set()
    raw = str(text or "").replace("、", "|")
    for item in raw.split("|"):
        tag = _clean_preview_text(item)
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(tag)
    return parts


def _join_tags(tags: list[str]) -> str:
    return "、".join([t for t in tags if _norm_text(t)])


def _merge_unique(items: list[str]) -> list[str]:
    out = []
    seen = set()
    for item in items:
        text = _clean_preview_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _merge_alias_text(parts: list[str], sep: str = " | ") -> str:
    return sep.join(_merge_unique(parts))


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _build_review_map(review_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for _, row in review_df.fillna("").iterrows():
        ticker = _norm_text(row.get("ticker", "")).upper()
        if not ticker:
            continue
        out[ticker] = {
            "rebuild_mode": _truthy(row.get("review_rebuild_mode", False)),
            "new_major": _clean_preview_text(row.get("review_major_sector_new", "")) or _clean_preview_text(row.get("major_sector", "")),
            "new_subsector": _clean_preview_text(row.get("review_ai_subsector_new", "")) or _clean_preview_text(row.get("ai_subsector", "")) or _clean_preview_text(row.get("final_subsector", "")),
            "new_tags_list": _split_tag_text(row.get("review_ai_small_tags_new", "")) or _split_tag_text(row.get("ai_small_tags", "")),
            "new_summary": _clean_preview_text(row.get("review_summary_new", "")) or _clean_preview_text(row.get("company_summary_zh_tw", "")),
            "status": _norm_text(row.get("review_status", "")),
            "conflict_level": _norm_text(row.get("review_conflict_level", "")),
            "reason": _clean_preview_text(row.get("review_rebuild_reason", "")) or _clean_preview_text(row.get("review_notes", "")),
        }
    return out


def _update_sheet(df: pd.DataFrame, review_map: dict[str, dict[str, Any]], generated_at: str) -> tuple[pd.DataFrame, dict[str, int]]:
    d = df.copy().fillna("")
    if "代號" not in d.columns:
        return d, {"rows": 0, "rebuild_rows": 0, "alias_rows": 0}

    rebuild_rows = 0
    alias_rows = 0
    for idx in d.index:
        ticker = _norm_text(d.at[idx, "代號"]).upper()
        if not ticker or ticker not in review_map:
            continue

        rv = review_map[ticker]
        old_subsector = _clean_preview_text(d.at[idx, "最終子分類"]) if "最終子分類" in d.columns else ""
        old_tags_list = _split_tag_text(d.at[idx, "AI小分類標籤"]) if "AI小分類標籤" in d.columns else []
        old_summary = _clean_preview_text(d.at[idx, "公司簡介(繁中)"]) if "公司簡介(繁中)" in d.columns else ""

        if rv["rebuild_mode"]:
            if "大分類" in d.columns and rv["new_major"]:
                d.at[idx, "大分類"] = rv["new_major"]
            if "最終子分類" in d.columns and rv["new_subsector"]:
                d.at[idx, "最終子分類"] = rv["new_subsector"]
            if "AI小分類標籤" in d.columns and rv["new_tags_list"]:
                d.at[idx, "AI小分類標籤"] = _join_tags(rv["new_tags_list"])
            if "公司簡介(繁中)" in d.columns and rv["new_summary"]:
                d.at[idx, "公司簡介(繁中)"] = rv["new_summary"]
            d.at[idx, "搜尋別名最終子分類"] = rv["new_subsector"]
            d.at[idx, "搜尋別名AI標籤"] = _join_tags(rv["new_tags_list"])
            d.at[idx, "搜尋別名簡介"] = rv["new_summary"]
            d.at[idx, "第二模型顯示策略"] = "rebuild_採用新值"
            rebuild_rows += 1
        else:
            alias_subsector = _merge_alias_text([old_subsector, rv["new_subsector"]], sep=" | ")
            alias_tags = _join_tags(_merge_unique(old_tags_list + rv["new_tags_list"]))
            alias_summary = _merge_alias_text([old_summary, rv["new_summary"]], sep=" || ")
            original_blob = " ".join([old_subsector, _join_tags(old_tags_list), old_summary])
            if _contains_any(original_blob, OPTICAL_COMM_KEYWORDS):
                alias_tags = _join_tags(_merge_unique(["光通訊"] + _split_tag_text(alias_tags)))
            d.at[idx, "搜尋別名最終子分類"] = alias_subsector or old_subsector
            d.at[idx, "搜尋別名AI標籤"] = alias_tags or _join_tags(old_tags_list)
            d.at[idx, "搜尋別名簡介"] = alias_summary or old_summary
            d.at[idx, "第二模型顯示策略"] = "legacy_display_search_aliases"
            alias_rows += 1

        d.at[idx, "第二模型模式"] = "rebuild" if rv["rebuild_mode"] else "conservative"
        d.at[idx, "第二模型狀態"] = rv["status"]
        d.at[idx, "第二模型衝突等級"] = rv["conflict_level"]
        d.at[idx, "第二模型原因"] = rv["reason"]
        d.at[idx, "第二模型採納時間"] = generated_at

    return d, {"rows": len(d), "rebuild_rows": rebuild_rows, "alias_rows": alias_rows}


def build_preview(base_path: Path, review_path: Path, output_path: Path, backup_existing_output: bool) -> Path:
    if not base_path.exists():
        raise FileNotFoundError(f"Base workbook not found: {base_path}")
    if not review_path.exists():
        raise FileNotFoundError(f"Review workbook not found: {review_path}")

    if backup_existing_output and output_path.exists():
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_path.with_name(f"{output_path.stem}.backup_before_preview_refresh_{stamp}{output_path.suffix}")
        shutil.copy2(output_path, backup_path)
        print(f"[INFO] existing_output_backup={backup_path}")

    review_df = pd.read_excel(review_path, sheet_name="all_reviews").fillna("")
    review_map = _build_review_map(review_df)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    base_xls = pd.ExcelFile(base_path)
    written: dict[str, pd.DataFrame] = {}
    summary_rows: list[dict[str, Any]] = []
    total_rebuild_rows = 0
    total_alias_rows = 0
    for sheet in TARGET_SHEETS:
        if sheet not in base_xls.sheet_names:
            continue
        source_df = pd.read_excel(base_path, sheet_name=sheet).fillna("")
        updated_df, stats = _update_sheet(source_df, review_map, generated_at)
        written[sheet] = updated_df
        total_rebuild_rows += stats["rebuild_rows"]
        total_alias_rows += stats["alias_rows"]
        summary_rows.append(
            {
                "sheet": sheet,
                "rows": stats["rows"],
                "rebuild_rows": stats["rebuild_rows"],
                "alias_rows": stats["alias_rows"],
            }
        )

    meta_df = pd.DataFrame(
        [
            {"key": "generated_at", "value": generated_at},
            {"key": "base_workbook", "value": str(base_path)},
            {"key": "review_workbook", "value": str(review_path)},
            {"key": "hybrid_strategy", "value": "rebuild rows use new values; non-rebuild rows keep legacy display and merge old+new aliases for search"},
            {"key": "source_review_rows", "value": len(review_df)},
            {"key": "rebuild_rows", "value": int(review_df["review_rebuild_mode"].map(_truthy).sum())},
            {"key": "alias_search_rows", "value": int((~review_df["review_rebuild_mode"].map(_truthy)).sum())},
        ]
    )
    summary_df = pd.DataFrame(summary_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in written.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        meta_df.to_excel(writer, sheet_name="preview_metadata", index=False)
        summary_df.to_excel(writer, sheet_name="preview_summary", index=False)

    print(f"[DONE] hybrid_preview_output={output_path}")
    print(f"[INFO] rebuild_rows={total_rebuild_rows} alias_rows={total_alias_rows}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a hybrid search preview workbook for the Streamlit app.")
    p.add_argument("--base", default=str(DEFAULT_BASE), help="Base investable workbook path.")
    p.add_argument("--review", default=str(DEFAULT_REVIEW), help="Second-opinion review workbook path.")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output preview workbook path.")
    p.add_argument("--skip-backup-output", action="store_true", help="Skip backup if hybrid preview output already exists.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_preview(
        base_path=Path(args.base),
        review_path=Path(args.review),
        output_path=Path(args.out),
        backup_existing_output=not bool(args.skip_backup_output),
    )


if __name__ == "__main__":
    main()
