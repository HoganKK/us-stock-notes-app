from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "output" / "notes.db"


def get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS stock_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                note_date TEXT NOT NULL,
                event_title TEXT NOT NULL,
                event_detail TEXT NOT NULL,
                impact TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                source_url TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS macro_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_date TEXT NOT NULL,
                event_title TEXT NOT NULL,
                event_detail TEXT NOT NULL,
                affected_sectors TEXT NOT NULL DEFAULT '',
                affected_subsectors TEXT NOT NULL DEFAULT '',
                affected_tickers TEXT NOT NULL DEFAULT '',
                impact TEXT NOT NULL,
                source_url TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS event_theme_hits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                note_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                theme TEXT NOT NULL,
                impact TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.5,
                reason TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            """
        )
        conn.commit()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _csv(items: list[str]) -> str:
    return "|".join([str(x).strip() for x in items if str(x).strip()])


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s or "").split("|") if x.strip()]


def set_meta(key: str, value: str, db_path: Path = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            "INSERT INTO app_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def get_meta(key: str, default: str = "", db_path: Path = DB_PATH) -> str:
    with closing(get_conn(db_path)) as conn:
        row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default


def add_stock_note(
    ticker: str,
    note_date: str,
    event_title: str,
    event_detail: str,
    impact: str,
    confidence: float,
    source_url: str,
    tags: list[str],
    db_path: Path = DB_PATH,
) -> int:
    now = _now()
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO stock_notes(
                ticker, note_date, event_title, event_detail, impact, confidence, source_url, tags, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.strip().upper(),
                note_date.strip(),
                event_title.strip(),
                event_detail.strip(),
                impact.strip(),
                float(confidence),
                source_url.strip(),
                _csv(tags),
                now,
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_stock_note(
    note_id: int,
    note_date: str,
    event_title: str,
    event_detail: str,
    impact: str,
    confidence: float,
    source_url: str,
    tags: list[str],
    db_path: Path = DB_PATH,
) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            """
            UPDATE stock_notes
            SET note_date=?, event_title=?, event_detail=?, impact=?, confidence=?, source_url=?, tags=?, updated_at=?
            WHERE id=?
            """,
            (
                note_date.strip(),
                event_title.strip(),
                event_detail.strip(),
                impact.strip(),
                float(confidence),
                source_url.strip(),
                _csv(tags),
                _now(),
                int(note_id),
            ),
        )
        conn.commit()


def delete_stock_note(note_id: int, db_path: Path = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute("DELETE FROM stock_notes WHERE id = ?", (int(note_id),))
        conn.commit()


def list_stock_notes(
    ticker: str | None = None,
    impact: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    tag_keyword: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM stock_notes WHERE 1=1"
    params: list[Any] = []
    if ticker:
        sql += " AND ticker = ?"
        params.append(ticker.strip().upper())
    if impact and impact != "全部":
        sql += " AND impact = ?"
        params.append(impact)
    if start_date:
        sql += " AND note_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND note_date <= ?"
        params.append(end_date)
    if tag_keyword:
        sql += " AND tags LIKE ?"
        params.append(f"%{tag_keyword.strip()}%")
    sql += " ORDER BY note_date DESC, updated_at DESC"

    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["tags_list"] = _parse_csv(r.get("tags", ""))
    return out


def add_macro_note(
    note_date: str,
    event_title: str,
    event_detail: str,
    affected_sectors: list[str],
    affected_subsectors: list[str],
    affected_tickers: list[str],
    impact: str,
    source_url: str,
    db_path: Path = DB_PATH,
) -> int:
    now = _now()
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute(
            """
            INSERT INTO macro_notes(
                note_date, event_title, event_detail, affected_sectors, affected_subsectors, affected_tickers, impact, source_url, created_at, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note_date.strip(),
                event_title.strip(),
                event_detail.strip(),
                _csv(affected_sectors),
                _csv(affected_subsectors),
                _csv([t.upper() for t in affected_tickers]),
                impact.strip(),
                source_url.strip(),
                now,
                now,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_macro_note(
    note_id: int,
    note_date: str,
    event_title: str,
    event_detail: str,
    affected_sectors: list[str],
    affected_subsectors: list[str],
    affected_tickers: list[str],
    impact: str,
    source_url: str,
    db_path: Path = DB_PATH,
) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            """
            UPDATE macro_notes
            SET note_date=?, event_title=?, event_detail=?, affected_sectors=?, affected_subsectors=?, affected_tickers=?,
                impact=?, source_url=?, updated_at=?
            WHERE id=?
            """,
            (
                note_date.strip(),
                event_title.strip(),
                event_detail.strip(),
                _csv(affected_sectors),
                _csv(affected_subsectors),
                _csv([t.upper() for t in affected_tickers]),
                impact.strip(),
                source_url.strip(),
                _now(),
                int(note_id),
            ),
        )
        conn.commit()


def delete_macro_note(note_id: int, db_path: Path = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute("DELETE FROM macro_notes WHERE id = ?", (int(note_id),))
        conn.commit()


def list_macro_notes(
    impact: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM macro_notes WHERE 1=1"
    params: list[Any] = []
    if impact and impact != "全部":
        sql += " AND impact = ?"
        params.append(impact)
    if start_date:
        sql += " AND note_date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND note_date <= ?"
        params.append(end_date)
    sql += " ORDER BY note_date DESC, updated_at DESC"

    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["affected_sectors_list"] = _parse_csv(r.get("affected_sectors", ""))
        r["affected_subsectors_list"] = _parse_csv(r.get("affected_subsectors", ""))
        r["affected_tickers_list"] = [x.upper() for x in _parse_csv(r.get("affected_tickers", ""))]
    return out


def list_related_macro_notes(
    ticker: str,
    sector: str,
    subsector: str,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    ticker = ticker.strip().upper()
    notes = list_macro_notes(db_path=db_path)
    out: list[dict[str, Any]] = []
    for n in notes:
        if ticker and ticker in n["affected_tickers_list"]:
            out.append(n)
            continue
        if sector and sector in n["affected_sectors_list"]:
            out.append(n)
            continue
        if subsector and subsector in n["affected_subsectors_list"]:
            out.append(n)
            continue
    return out


def replace_event_theme_hits(note_id: int, rows: list[dict[str, Any]], db_path: Path = DB_PATH) -> None:
    now = _now()
    with closing(get_conn(db_path)) as conn:
        conn.execute("DELETE FROM event_theme_hits WHERE note_id = ?", (int(note_id),))
        for r in rows:
            conn.execute(
                """
                INSERT INTO event_theme_hits(note_id, ticker, theme, impact, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(note_id),
                    str(r.get("ticker", "")).strip().upper(),
                    str(r.get("theme", "")).strip(),
                    str(r.get("impact", "中性")).strip() or "中性",
                    float(r.get("confidence", 0.5) or 0.5),
                    str(r.get("reason", "")).strip(),
                    now,
                ),
            )
        conn.commit()


def list_event_theme_hits(
    ticker: str | None = None,
    theme_keyword: str | None = None,
    db_path: Path = DB_PATH,
) -> list[dict[str, Any]]:
    sql = """
    SELECT h.*, m.note_date, m.event_title, m.event_detail, m.source_url
    FROM event_theme_hits h
    LEFT JOIN macro_notes m ON m.id = h.note_id
    WHERE 1=1
    """
    params: list[Any] = []
    if ticker:
        sql += " AND h.ticker = ?"
        params.append(ticker.strip().upper())
    if theme_keyword:
        sql += " AND h.theme LIKE ?"
        params.append(f"%{theme_keyword.strip()}%")
    sql += " ORDER BY m.note_date DESC, h.confidence DESC, h.id DESC"
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def list_themes_summary(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    sql = """
    SELECT theme, COUNT(DISTINCT ticker) AS stock_count, COUNT(*) AS hit_count
    FROM event_theme_hits
    GROUP BY theme
    ORDER BY stock_count DESC, hit_count DESC, theme ASC
    """
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


def export_all_as_dict(db_path: Path = DB_PATH) -> dict[str, Any]:
    with closing(get_conn(db_path)) as conn:
        stock_rows = [dict(r) for r in conn.execute("SELECT * FROM stock_notes ORDER BY id ASC").fetchall()]
        macro_rows = [dict(r) for r in conn.execute("SELECT * FROM macro_notes ORDER BY id ASC").fetchall()]
        theme_rows = [dict(r) for r in conn.execute("SELECT * FROM event_theme_hits ORDER BY id ASC").fetchall()]
        meta_rows = [dict(r) for r in conn.execute("SELECT * FROM app_meta ORDER BY key ASC").fetchall()]
    return {
        "stock_notes": stock_rows,
        "macro_notes": macro_rows,
        "event_theme_hits": theme_rows,
        "app_meta": meta_rows,
    }


def import_all_from_dict(obj: dict[str, Any], db_path: Path = DB_PATH, replace: bool = True) -> None:
    stock_rows = obj.get("stock_notes", []) or []
    macro_rows = obj.get("macro_notes", []) or []
    theme_rows = obj.get("event_theme_hits", []) or []
    meta_rows = obj.get("app_meta", []) or []
    with closing(get_conn(db_path)) as conn:
        if replace:
            conn.execute("DELETE FROM stock_notes")
            conn.execute("DELETE FROM macro_notes")
            conn.execute("DELETE FROM event_theme_hits")
            conn.execute("DELETE FROM app_meta")
        for r in stock_rows:
            conn.execute(
                """
                INSERT INTO stock_notes(id, ticker, note_date, event_title, event_detail, impact, confidence, source_url, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("id"),
                    str(r.get("ticker", "")).upper(),
                    r.get("note_date", ""),
                    r.get("event_title", ""),
                    r.get("event_detail", ""),
                    r.get("impact", "中性"),
                    float(r.get("confidence", 0.5) or 0.5),
                    r.get("source_url", ""),
                    r.get("tags", ""),
                    r.get("created_at", _now()),
                    r.get("updated_at", _now()),
                ),
            )
        for r in macro_rows:
            conn.execute(
                """
                INSERT INTO macro_notes(id, note_date, event_title, event_detail, affected_sectors, affected_subsectors, affected_tickers, impact, source_url, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("id"),
                    r.get("note_date", ""),
                    r.get("event_title", ""),
                    r.get("event_detail", ""),
                    r.get("affected_sectors", ""),
                    r.get("affected_subsectors", ""),
                    r.get("affected_tickers", ""),
                    r.get("impact", "中性"),
                    r.get("source_url", ""),
                    r.get("created_at", _now()),
                    r.get("updated_at", _now()),
                ),
            )
        for r in theme_rows:
            conn.execute(
                """
                INSERT INTO event_theme_hits(id, note_id, ticker, theme, impact, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    r.get("id"),
                    int(r.get("note_id", 0) or 0),
                    str(r.get("ticker", "")).upper(),
                    str(r.get("theme", "")),
                    str(r.get("impact", "中性")),
                    float(r.get("confidence", 0.5) or 0.5),
                    str(r.get("reason", "")),
                    str(r.get("created_at", _now())),
                ),
            )
        for r in meta_rows:
            if "key" in r:
                conn.execute(
                    "INSERT INTO app_meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (r["key"], str(r.get("value", ""))),
                )
        conn.commit()


def export_json_text(db_path: Path = DB_PATH) -> str:
    payload = export_all_as_dict(db_path=db_path)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def import_json_text(text: str, db_path: Path = DB_PATH, replace: bool = True) -> None:
    payload = json.loads(text)
    import_all_from_dict(payload, db_path=db_path, replace=replace)
