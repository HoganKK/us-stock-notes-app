#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


COL_MAP = {
    "ticker": "代號",
    "exchange": "交易所(原文)",
    "exchange_zh_tw": "交易所",
    "security_name": "公司名稱",
    "company_summary_zh_tw": "公司簡介(繁中)",
    "major_sector": "大分類(原文)",
    "major_sector_zh_tw": "大分類",
    "subsector": "子分類(原文)",
    "subsector_zh_tw": "子分類",
    "ai_subsector": "AI子分類(原文)",
    "ai_subsector_zh_tw": "AI子分類",
    "final_subsector": "最終子分類(原文)",
    "final_subsector_zh_tw": "最終子分類",
    "ai_small_tags": "AI小分類標籤(原文)",
    "ai_small_tags_zh_tw": "AI小分類標籤",
    "security_type": "證券類型",
    "generated_at": "生成時間",
    "sec_title": "SEC公司名稱",
    "cik": "CIK",
    "sic": "SIC",
    "sic_description": "SIC說明",
    "entity_type": "實體類型",
    "is_investable": "是否可投資",
    "exclude_reason": "排除原因",
}


SHEET_MAP = {
    "all_stocks": "全部股票",
    "investable_all": "可投資清單",
    "excluded_non_investable": "排除清單",
    "investable_summary": "可投資統計",
    "theme_index": "主題索引",
}


def _pick(df: pd.DataFrame, zh_col: str, raw_col: str) -> pd.Series:
    if zh_col in df.columns:
        base = df[zh_col].astype(str).fillna("").str.strip()
        if raw_col in df.columns:
            raw = df[raw_col].astype(str).fillna("").str.strip()
            return base.where(base != "", raw)
        return base
    if raw_col in df.columns:
        return df[raw_col].astype(str).fillna("").str.strip()
    return pd.Series([""] * len(df))


def _normalize_tags(s: Any) -> str:
    txt = str(s or "").strip()
    if not txt:
        return ""
    parts = [p.strip() for p in txt.split("|") if p.strip()]
    return "、".join(dict.fromkeys(parts))


def transform_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().fillna("")

    # Prefer zh_tw fields when available.
    if "exchange" in out.columns or "exchange_zh_tw" in out.columns:
        out["exchange_zh_tw"] = _pick(out, "exchange_zh_tw", "exchange")
    if "major_sector" in out.columns or "major_sector_zh_tw" in out.columns:
        out["major_sector_zh_tw"] = _pick(out, "major_sector_zh_tw", "major_sector")
    if "subsector" in out.columns or "subsector_zh_tw" in out.columns:
        out["subsector_zh_tw"] = _pick(out, "subsector_zh_tw", "subsector")
    if "ai_subsector" in out.columns or "ai_subsector_zh_tw" in out.columns:
        out["ai_subsector_zh_tw"] = _pick(out, "ai_subsector_zh_tw", "ai_subsector")
    if "final_subsector" in out.columns or "final_subsector_zh_tw" in out.columns:
        out["final_subsector_zh_tw"] = _pick(out, "final_subsector_zh_tw", "final_subsector")
    if "ai_small_tags" in out.columns or "ai_small_tags_zh_tw" in out.columns:
        out["ai_small_tags_zh_tw"] = _pick(out, "ai_small_tags_zh_tw", "ai_small_tags").map(_normalize_tags)

    # Normalize booleans for readability.
    if "is_investable" in out.columns:
        out["is_investable"] = out["is_investable"].map(
            lambda x: "是" if str(x).strip().lower() in {"true", "1", "yes"} else ("否" if str(x).strip() != "" else "")
        )

    # Keep a practical display order first, then append remaining columns.
    preferred = [
        "ticker",
        "security_name",
        "exchange_zh_tw",
        "major_sector_zh_tw",
        "final_subsector_zh_tw",
        "ai_small_tags_zh_tw",
        "company_summary_zh_tw",
        "is_investable",
        "exclude_reason",
        "generated_at",
    ]
    existing_preferred = [c for c in preferred if c in out.columns]
    remaining = [c for c in out.columns if c not in existing_preferred]
    out = out[existing_preferred + remaining]

    out = out.rename(columns={c: COL_MAP.get(c, c) for c in out.columns})
    return out


def run(input_path: Path, output_path: Path) -> None:
    sheets = pd.read_excel(input_path, sheet_name=None)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            zh_sheet = SHEET_MAP.get(name, name)
            zh_sheet = str(zh_sheet)[:31]
            out = transform_sheet(df)
            out.to_excel(writer, sheet_name=zh_sheet, index=False)

    print(f"[DONE] output={output_path}")
    print(f"[INFO] sheets={len(sheets)}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast no-AI Traditional Chinese export for workbook.")
    p.add_argument(
        "--in",
        dest="input_path",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes.xlsx",
        help="Input workbook path.",
    )
    p.add_argument(
        "--out",
        dest="output_path",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes_zh_tw_fast.xlsx",
        help="Output workbook path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(Path(args.input_path), Path(args.output_path))


if __name__ == "__main__":
    main()
