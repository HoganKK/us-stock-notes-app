#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_BASE = Path("E:/CHATGPT_CODEX/STOCK_NOTES/stock_fetcher/output/us_stocks_investable_themes_zh_tw_fast.xlsx")
DEFAULT_REVIEW = Path("E:/CHATGPT_CODEX/STOCK_NOTES/stock_fetcher/output/us_stocks_ai_review.xlsx")
DEFAULT_OUTPUT = Path("E:/CHATGPT_CODEX/STOCK_NOTES/stock_fetcher/output/us_stocks_investable_themes_zh_tw_second_opinion_preview.xlsx")
TARGET_SHEETS = ["全部股票", "可投資清單", "排除清單"]
TEXT_REPLACEMENTS = [
    ("用事業", "公用事業"),
    ("潔能源", "清潔能源"),
    ("潔交通", "清潔交通"),
    ("進空中交通", "先進空中交通"),
    ("進空中運輸", "先進空中運輸"),
    ("伏模組", "光伏模組"),
    ("伏支架", "光伏支架"),
    ("太陽能伏", "太陽能光伏"),
    ("電動車電基礎設施", "電動車充電基礎設施"),
    ("電動車快速電網絡", "電動車快速充電網絡"),
    ("電服務營運商", "充電服務營運商"),
    ("車隊電解決方案", "車隊充電解決方案"),
    ("直流快站", "直流快充站"),
    ("裝與衛生用品材料", "包裝與衛生用品材料"),
    ("目布局", "項目布局"),
    ("倉儲式大店", "倉儲式大型門店"),
    ("家改善", "家居改善"),
    ("店鑰匙", "門店鑰匙"),
]


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _truthy(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "1.0", "true", "yes", "y"}


def _clean_preview_text(value: Any) -> str:
    text = _norm_text(value)
    for old, new in TEXT_REPLACEMENTS:
        text = text.replace(old, new)
    cleanup_pairs = [
        ("公公用事業", "公用事業"),
        ("光光伏", "光伏"),
    ]
    for old, new in cleanup_pairs:
        text = text.replace(old, new)
    return text


def _tags_to_display(value: Any) -> str:
    parts = []
    seen = set()
    raw = str(value or "").replace("、", "|")
    for item in raw.split("|"):
        tag = _clean_preview_text(item)
        if not tag:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        parts.append(tag)
    return "、".join(parts)


def _build_review_map(review_df: pd.DataFrame) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for _, row in review_df.fillna("").iterrows():
        ticker = _norm_text(row.get("ticker", "")).upper()
        if not ticker:
            continue
        out[ticker] = {
            "major_sector": _clean_preview_text(row.get("review_major_sector_new", "")) or _clean_preview_text(row.get("major_sector", "")),
            "subsector": _clean_preview_text(row.get("review_ai_subsector_new", "")) or _clean_preview_text(row.get("ai_subsector", "")) or _clean_preview_text(row.get("final_subsector", "")),
            "tags_display": _tags_to_display(row.get("review_ai_small_tags_new", "")) or _tags_to_display(row.get("ai_small_tags", "")),
            "summary": _clean_preview_text(row.get("review_summary_new", "")) or _clean_preview_text(row.get("company_summary_zh_tw", "")),
            "mode": "rebuild" if _truthy(row.get("review_rebuild_mode", False)) else "conservative",
            "status": _norm_text(row.get("review_status", "")),
            "conflict_level": _norm_text(row.get("review_conflict_level", "")),
            "reason": _clean_preview_text(row.get("review_rebuild_reason", "")) or _clean_preview_text(row.get("review_notes", "")),
        }
    return out


def _update_sheet(df: pd.DataFrame, review_map: dict[str, dict[str, str]], generated_at: str) -> tuple[pd.DataFrame, int]:
    d = df.copy().fillna("")
    if "代號" not in d.columns:
        return d, 0

    updated_rows = 0
    for idx in d.index:
        ticker = _norm_text(d.at[idx, "代號"]).upper()
        if not ticker or ticker not in review_map:
            continue
        rv = review_map[ticker]
        if "大分類" in d.columns and rv["major_sector"]:
            d.at[idx, "大分類"] = rv["major_sector"]
        if "最終子分類" in d.columns and rv["subsector"]:
            d.at[idx, "最終子分類"] = rv["subsector"]
        if "AI小分類標籤" in d.columns and rv["tags_display"]:
            d.at[idx, "AI小分類標籤"] = rv["tags_display"]
        if "公司簡介(繁中)" in d.columns and rv["summary"]:
            d.at[idx, "公司簡介(繁中)"] = rv["summary"]
        updated_rows += 1

    d["第二模型模式"] = d["代號"].astype(str).str.upper().map(lambda x: review_map.get(x, {}).get("mode", ""))
    d["第二模型狀態"] = d["代號"].astype(str).str.upper().map(lambda x: review_map.get(x, {}).get("status", ""))
    d["第二模型衝突等級"] = d["代號"].astype(str).str.upper().map(lambda x: review_map.get(x, {}).get("conflict_level", ""))
    d["第二模型原因"] = d["代號"].astype(str).str.upper().map(lambda x: review_map.get(x, {}).get("reason", ""))
    d["第二模型採納時間"] = generated_at
    return d, updated_rows


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
    written = {}
    summary_rows: list[dict[str, Any]] = []
    for sheet in TARGET_SHEETS:
        if sheet not in base_xls.sheet_names:
            continue
        source_df = pd.read_excel(base_path, sheet_name=sheet).fillna("")
        updated_df, updated_rows = _update_sheet(source_df, review_map, generated_at)
        written[sheet] = updated_df
        summary_rows.append({"sheet": sheet, "rows": len(updated_df), "updated_rows": updated_rows})

    summary_rows.extend(
        [
            {"sheet": "metadata", "rows": len(review_df), "updated_rows": ""},
        ]
    )

    meta_df = pd.DataFrame(
        [
            {"key": "generated_at", "value": generated_at},
            {"key": "base_workbook", "value": str(base_path)},
            {"key": "review_workbook", "value": str(review_path)},
            {"key": "source_review_rows", "value": len(review_df)},
            {"key": "rebuild_rows", "value": int(review_df["review_rebuild_mode"].map(_truthy).sum())},
            {"key": "mismatch_rows", "value": int(review_df["review_conflict_level"].isin(["high", "medium", "low"]).sum())},
            {"key": "match_rows", "value": int((review_df["review_conflict_level"] == "none").sum())},
        ]
    )
    summary_df = pd.DataFrame(summary_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, sheet_df in written.items():
            sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
        meta_df.to_excel(writer, sheet_name="preview_metadata", index=False)
        summary_df.to_excel(writer, sheet_name="preview_summary", index=False)

    print(f"[DONE] preview_output={output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a second-opinion preview workbook for the Streamlit app.")
    p.add_argument("--base", default=str(DEFAULT_BASE), help="Base investable workbook path.")
    p.add_argument("--review", default=str(DEFAULT_REVIEW), help="Second-opinion review workbook path.")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output preview workbook path.")
    p.add_argument("--skip-backup-output", action="store_true", help="Skip backup if preview output already exists.")
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
