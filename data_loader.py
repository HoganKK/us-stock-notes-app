from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

DEFAULT_INPUT_CANDIDATES = [
    OUTPUT_DIR / "us_stocks_investable_themes_zh_tw_hybrid_search_preview.xlsx",
    OUTPUT_DIR / "us_stocks_investable_themes_zh_tw_second_opinion_preview.xlsx",
    OUTPUT_DIR / "us_stocks_investable_themes_zh_tw_fast.xlsx",
    OUTPUT_DIR / "us_stocks_investable_themes.xlsx",
    OUTPUT_DIR / "us_stocks_subsectors_ai_zh_tw.xlsx",
    OUTPUT_DIR / "us_stocks_subsectors.xlsx",
]
DEFAULT_INPUT_PATH = DEFAULT_INPUT_CANDIDATES[0]
PREFERRED_SHEETS = ["可投資清單", "investable_all", "all_stocks", "全部股票"]


@dataclass
class DataBundle:
    raw_df: pd.DataFrame
    schema_df: pd.DataFrame
    source_name: str
    source_hash: str
    loaded_at: str
    source_sheet: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def _pick_sheet(sheet_names: list[str]) -> str:
    for name in PREFERRED_SHEETS:
        if name in sheet_names:
            return name
    return sheet_names[0]


def resolve_default_input_path() -> Path:
    for p in DEFAULT_INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No default Excel found under ./output. Expected one of: "
        + ", ".join(x.name for x in DEFAULT_INPUT_CANDIDATES)
    )


def _coalesce(df: pd.DataFrame, candidates: list[str], default: str = "") -> pd.Series:
    out = pd.Series([default] * len(df), index=df.index, dtype="object")
    for c in candidates:
        if c in df.columns:
            s = df[c].astype(str).fillna("").str.strip()
            out = out.where(out.astype(str).str.strip() != "", s)
    return out.fillna(default).astype(str)


def _parse_tags(text: str) -> list[str]:
    parts = [p.strip() for p in str(text or "").replace("、", "|").split("|")]
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def to_internal_schema(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().fillna("")
    out = pd.DataFrame()
    out["ticker"] = _coalesce(d, ["代號", "ticker"]).str.upper()
    out["company_name"] = _coalesce(d, ["公司名稱", "security_name", "SEC公司名稱", "sec_title"])
    out["exchange"] = _coalesce(d, ["交易所", "exchange_zh_tw", "exchange"])
    out["sector"] = _coalesce(d, ["大分類", "major_sector_zh_tw", "major_sector"])
    out["subsector"] = _coalesce(d, ["最終子分類", "final_subsector_zh_tw", "final_subsector", "子分類"])
    out["tags"] = _coalesce(d, ["AI小分類標籤", "ai_small_tags_zh_tw", "ai_small_tags"])
    out["summary"] = _coalesce(d, ["公司簡介(繁中)", "company_summary_zh_tw"])
    out["subsector_aliases"] = _coalesce(d, ["搜尋別名最終子分類", "search_subsector_aliases"], default="").astype(str)
    out["tags_aliases"] = _coalesce(d, ["搜尋別名AI標籤", "search_tag_aliases"], default="").astype(str)
    out["summary_aliases"] = _coalesce(d, ["搜尋別名簡介", "search_summary_aliases"], default="").astype(str)
    out["generated_at"] = _coalesce(d, ["生成時間", "generated_at"])
    out["is_investable"] = _coalesce(d, ["是否可投資", "is_investable"])
    out["exclude_reason"] = _coalesce(d, ["排除原因", "exclude_reason"])

    out["subsector_aliases"] = out["subsector_aliases"].where(out["subsector_aliases"].astype(str).str.strip() != "", out["subsector"])
    out["tags_aliases"] = out["tags_aliases"].where(out["tags_aliases"].astype(str).str.strip() != "", out["tags"])
    out["summary_aliases"] = out["summary_aliases"].where(out["summary_aliases"].astype(str).str.strip() != "", out["summary"])

    out["tags_list"] = out["tags_aliases"].map(_parse_tags)
    out["search_blob"] = (
        out["ticker"].astype(str)
        + " "
        + out["company_name"].astype(str)
        + " "
        + out["summary"].astype(str)
        + " "
        + out["summary_aliases"].astype(str)
        + " "
        + out["subsector"].astype(str)
        + " "
        + out["subsector_aliases"].astype(str)
        + " "
        + out["tags"].astype(str)
        + " "
        + out["tags_aliases"].astype(str)
    ).str.lower()
    out = out.drop_duplicates(subset=["ticker"], keep="first").reset_index(drop=True)
    return out


def load_bundle_from_path(path: Path) -> DataBundle:
    path = Path(path)
    if not path.exists():
        path = resolve_default_input_path()
    content = path.read_bytes()
    xls = pd.ExcelFile(io.BytesIO(content))
    sheet = _pick_sheet(xls.sheet_names)
    raw_df = pd.read_excel(io.BytesIO(content), sheet_name=sheet).fillna("")
    schema_df = to_internal_schema(raw_df)
    return DataBundle(
        raw_df=raw_df,
        schema_df=schema_df,
        source_name=str(path),
        source_hash=_sha256_bytes(content),
        loaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source_sheet=sheet,
    )


def load_bundle_from_upload(uploaded_file: Any) -> DataBundle:
    content = uploaded_file.getvalue()
    xls = pd.ExcelFile(io.BytesIO(content))
    sheet = _pick_sheet(xls.sheet_names)
    raw_df = pd.read_excel(io.BytesIO(content), sheet_name=sheet).fillna("")
    schema_df = to_internal_schema(raw_df)
    return DataBundle(
        raw_df=raw_df,
        schema_df=schema_df,
        source_name=str(getattr(uploaded_file, "name", "uploaded.xlsx")),
        source_hash=_sha256_bytes(content),
        loaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        source_sheet=sheet,
    )
