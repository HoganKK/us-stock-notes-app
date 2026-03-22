#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


BAD_SHEET_CHARS = re.compile(r"[:\\/*?\[\]]")


def _norm(x: Any) -> str:
    return str(x or "").strip()


def _split_tags(tag_text: str) -> list[str]:
    tags = []
    seen = set()
    for raw in _norm(tag_text).split("|"):
        t = raw.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        tags.append(t)
    return tags


def _is_noise_theme(tag: str) -> bool:
    t = _norm(tag)
    if not t:
        return True
    if len(t) < 2 or len(t) > 30:
        return True
    if re.search(r"\d{4}年?到期|到期票據|票據", t):
        return True
    if re.fullmatch(r"[0-9A-Za-z\-\./ ]+", t) and len(t) <= 3:
        return True
    return False


def _sheet_name(base: str, used: set[str]) -> str:
    s = BAD_SHEET_CHARS.sub("_", _norm(base))
    s = re.sub(r"\s+", " ", s).strip()
    s = s[:31] if s else "theme"
    if not s:
        s = "theme"
    if s not in used:
        used.add(s)
        return s
    i = 2
    while True:
        suffix = f"_{i}"
        trimmed = s[: max(1, 31 - len(suffix))] + suffix
        if trimmed not in used:
            used.add(trimmed)
            return trimmed
        i += 1


def _investable_filter_reason(row: pd.Series) -> str:
    name = _norm(row.get("security_name")).lower()
    sec_title = _norm(row.get("sec_title")).lower()
    text = f"{name} {sec_title}"
    tags = _norm(row.get("ai_small_tags_zh_tw") or row.get("ai_small_tags")).lower()

    reasons: list[str] = []

    if re.search(r"special purpose acquisition|blank check", text) or (
        "spac" in tags or "空白支票" in tags
    ):
        reasons.append("SPAC/空白支票")

    if re.search(r"\bwarrant(s)?\b|\bright(s)?\b|\bunit(s)?\b", text) or (
        "權證" in tags or "認股權證" in tags
    ):
        reasons.append("權證/權利/單位")

    if re.search(r"\bpreferred\b|preference share|depositary share|series [a-z]\b", text) or (
        "優先股" in tags
    ):
        reasons.append("優先股/存託股份")

    if re.search(r"closed[- ]end fund|income fund|bond fund|municipal fund|fund,? inc\.?$", text) or (
        "封閉式基金" in tags
    ):
        reasons.append("基金型證券")

    if re.search(r"trust certificate|exchange[- ]traded note|\betn\b", text):
        reasons.append("結構型票據/ETN")

    return ";".join(reasons)


def build_output(
    input_path: Path,
    output_path: Path,
    min_theme_count: int,
    max_theme_sheets: int,
    theme_col: str,
) -> None:
    df = pd.read_excel(input_path, sheet_name="all_stocks").fillna("")
    if theme_col not in df.columns:
        fallback = "ai_small_tags_zh_tw" if "ai_small_tags_zh_tw" in df.columns else "ai_small_tags"
        theme_col = fallback

    df["exclude_reason"] = df.apply(_investable_filter_reason, axis=1)
    df["is_investable"] = df["exclude_reason"].astype(str).str.strip() == ""

    investable = df[df["is_investable"]].copy().reset_index(drop=True)
    excluded = df[~df["is_investable"]].copy().reset_index(drop=True)

    # Theme counting (unique ticker per theme)
    theme_to_tickers: dict[str, set[str]] = {}
    for _, row in investable.iterrows():
        ticker = _norm(row.get("ticker"))
        if not ticker:
            continue
        tags = [t for t in _split_tags(_norm(row.get(theme_col))) if not _is_noise_theme(t)]
        for t in tags:
            theme_to_tickers.setdefault(t, set()).add(ticker)

    counter = Counter({k: len(v) for k, v in theme_to_tickers.items()})
    selected_themes = [
        t for t, c in counter.most_common() if c >= max(1, int(min_theme_count))
    ][: max(1, int(max_theme_sheets))]

    used_sheet_names = {
        "all_stocks",
        "investable_all",
        "excluded_non_investable",
        "investable_summary",
        "theme_index",
    }

    theme_index_rows = []
    theme_sheets: list[tuple[str, pd.DataFrame]] = []

    for theme in selected_themes:
        def has_theme(v: Any) -> bool:
            return theme in _split_tags(_norm(v))

        g = investable[investable[theme_col].map(has_theme)].copy()
        g = g.sort_values(["major_sector_zh_tw", "final_subsector_zh_tw", "ticker"], ascending=[True, True, True])
        sheet = _sheet_name(f"theme_{theme}", used_sheet_names)
        theme_index_rows.append(
            {
                "theme": theme,
                "company_count": int(counter.get(theme, 0)),
                "sheet_name": sheet,
            }
        )
        theme_sheets.append((sheet, g))

    summary = (
        investable.groupby(["major_sector_zh_tw", "final_subsector_zh_tw"], dropna=False)["ticker"]
        .count()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_stocks", index=False)
        investable.to_excel(writer, sheet_name="investable_all", index=False)
        excluded.to_excel(writer, sheet_name="excluded_non_investable", index=False)
        summary.to_excel(writer, sheet_name="investable_summary", index=False)
        pd.DataFrame(theme_index_rows).to_excel(writer, sheet_name="theme_index", index=False)
        for sheet, g in theme_sheets:
            g.to_excel(writer, sheet_name=sheet, index=False)

    print(f"[DONE] output={output_path}")
    print(f"[INFO] all_stocks={len(df)} investable={len(investable)} excluded={len(excluded)}")
    print(f"[INFO] themes_selected={len(theme_sheets)} min_theme_count={min_theme_count}")
    if theme_sheets:
        print("[INFO] top_themes:")
        for row in theme_index_rows[:20]:
            print(f"  - {row['theme']}: {row['company_count']} ({row['sheet_name']})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create investable-only workbook + multi-theme sheets from ai_small_tags."
    )
    p.add_argument(
        "--in",
        dest="input_path",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_subsectors_ai_zh_tw.xlsx",
        help="Input Excel path (must contain all_stocks).",
    )
    p.add_argument(
        "--out",
        dest="output_path",
        default="E:/VScode/my_python_project/stock_fetcher/output/us_stocks_investable_themes.xlsx",
        help="Output Excel path.",
    )
    p.add_argument(
        "--theme-col",
        default="ai_small_tags_zh_tw",
        help="Theme source column. Default uses ai_small_tags_zh_tw.",
    )
    p.add_argument(
        "--min-theme-count",
        type=int,
        default=12,
        help="Only create theme sheet if at least N companies have that theme.",
    )
    p.add_argument(
        "--max-theme-sheets",
        type=int,
        default=200,
        help="Maximum number of theme sheets to generate.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_output(
        input_path=Path(args.input_path),
        output_path=Path(args.output_path),
        min_theme_count=int(args.min_theme_count),
        max_theme_sheets=int(args.max_theme_sheets),
        theme_col=str(args.theme_col).strip(),
    )


if __name__ == "__main__":
    main()
