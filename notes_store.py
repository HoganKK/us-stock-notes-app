from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "output" / "notes.db"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _csv(items: list[str]) -> str:
    return "|".join([str(x).strip() for x in items if str(x).strip()])


def _parse_csv(s: str) -> list[str]:
    return [x.strip() for x in str(s or "").split("|") if x.strip()]


def _canon_text(s: str) -> str:
    return str(s or "").strip().lower()


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

            CREATE TABLE IF NOT EXISTS keyword_synonyms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                keyword TEXT NOT NULL UNIQUE,
                expansions TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS theme_canonical_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_theme TEXT NOT NULL UNIQUE,
                canonical_theme TEXT NOT NULL,
                enabled INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.commit()
    _seed_defaults(db_path)


def _seed_defaults(db_path: Path = DB_PATH) -> None:
    defaults = {
        "美伊戰爭": "iran war|iran|us|middle east conflict|hormuz|iran-israel",
        "以伊戰爭": "iran war|iran|israel|middle east conflict|hormuz",
        "美以伊": "iran war|iran|israel|us|middle east conflict|hormuz",
        "腦機接口": "brain-computer interface|bci|neuralink|brain chip",
        "spacex": "spacex|starlink|space economy|launch services",
        "量子": "quantum|quantum computing|qubit|quantum hardware",
    }
    with closing(get_conn(db_path)) as conn:
        for k, v in defaults.items():
            conn.execute(
                """
                INSERT INTO keyword_synonyms(keyword, expansions, enabled, updated_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(keyword) DO NOTHING
                """,
                (k, v, _now()),
            )
        conn.execute(
            "INSERT INTO app_meta(key, value) VALUES('event_min_confidence','0.55') ON CONFLICT(key) DO NOTHING"
        )
        conn.commit()


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
        conn.execute("DELETE FROM event_theme_hits WHERE note_id = ?", (int(note_id),))
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


def get_macro_note(note_id: int, db_path: Path = DB_PATH) -> dict[str, Any] | None:
    with closing(get_conn(db_path)) as conn:
        row = conn.execute("SELECT * FROM macro_notes WHERE id = ?", (int(note_id),)).fetchone()
    return dict(row) if row else None


def macro_note_exists(event_title: str, source_url: str = "", db_path: Path = DB_PATH) -> bool:
    title = str(event_title or "").strip()
    url = str(source_url or "").strip()
    if not title:
        return False
    with closing(get_conn(db_path)) as conn:
        if url:
            row = conn.execute("SELECT 1 FROM macro_notes WHERE source_url = ? LIMIT 1", (url,)).fetchone()
            if row:
                return True
        row2 = conn.execute("SELECT 1 FROM macro_notes WHERE event_title = ? LIMIT 1", (title,)).fetchone()
    return bool(row2)


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
    # Dedup by (ticker, theme, impact), keep highest confidence
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for r in rows:
        ticker = str(r.get("ticker", "")).strip().upper()
        theme = str(r.get("theme", "")).strip()
        impact = str(r.get("impact", "中性")).strip() or "中性"
        if not ticker or not theme:
            continue
        key = (ticker, theme, impact)
        conf = float(r.get("confidence", 0.5) or 0.5)
        if key not in best or conf > float(best[key].get("confidence", 0.0)):
            best[key] = {
                "ticker": ticker,
                "theme": theme,
                "impact": impact,
                "confidence": max(0.0, min(1.0, conf)),
                "reason": str(r.get("reason", "")).strip(),
            }
    with closing(get_conn(db_path)) as conn:
        conn.execute("DELETE FROM event_theme_hits WHERE note_id = ?", (int(note_id),))
        for r in best.values():
            conn.execute(
                """
                INSERT INTO event_theme_hits(note_id, ticker, theme, impact, confidence, reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(note_id),
                    r["ticker"],
                    r["theme"],
                    r["impact"],
                    float(r["confidence"]),
                    r["reason"],
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
        kw = str(theme_keyword).strip()
        aliases = [kw]
        # include canonical aliases from rules (raw/canonical both directions)
        with closing(get_conn(db_path)) as conn:
            rules = conn.execute(
                "SELECT raw_theme, canonical_theme FROM theme_canonical_rules WHERE enabled=1"
            ).fetchall()
        low = _canon_text(kw)
        for r in rules:
            raw = str(r["raw_theme"] or "").strip()
            can = str(r["canonical_theme"] or "").strip()
            if not raw or not can:
                continue
            if _canon_text(raw) == low:
                aliases.append(can)
            if _canon_text(can) == low:
                aliases.append(raw)
        aliases = sorted(set([x for x in aliases if x]))
        if aliases:
            sql += " AND (" + " OR ".join(["h.theme LIKE ?"] * len(aliases)) + ")"
            params.extend([f"%{x}%" for x in aliases])
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


def list_themes_ranked(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    """
    Theme heat ranking with:
    - alias merge using theme_canonical_rules
    - score = confidence * recency_weight
      where recency_weight = max(0.2, 1.0 - days_since_note/14)
    """
    with closing(get_conn(db_path)) as conn:
        hit_rows = conn.execute(
            """
            SELECT h.theme, h.ticker, h.confidence, h.impact, m.note_date
            FROM event_theme_hits h
            LEFT JOIN macro_notes m ON m.id = h.note_id
            """
        ).fetchall()
        rules = conn.execute(
            "SELECT raw_theme, canonical_theme FROM theme_canonical_rules WHERE enabled=1"
        ).fetchall()

    rule_map: dict[str, str] = {}
    for r in rules:
        raw = str(r["raw_theme"] or "").strip()
        can = str(r["canonical_theme"] or "").strip()
        if raw and can:
            rule_map[_canon_text(raw)] = can

    grouped: dict[str, dict[str, Any]] = {}
    now = datetime.now()
    for r in hit_rows:
        theme_raw = str(r["theme"] or "").strip()
        if not theme_raw:
            continue
        theme = rule_map.get(_canon_text(theme_raw), theme_raw)

        note_date = str(r["note_date"] or "")[:10]
        days = 14.0
        try:
            d = datetime.strptime(note_date, "%Y-%m-%d")
            days = max(0.0, (now - d).days)
        except Exception:
            days = 14.0
        recency_weight = max(0.2, 1.0 - (days / 14.0))
        conf = float(r["confidence"] or 0.5)
        score = conf * recency_weight

        g = grouped.setdefault(
            theme,
            {
                "theme": theme,
                "stock_set": set(),
                "hit_count": 0,
                "score": 0.0,
                "recent_date": note_date,
                "bull_count": 0,
                "bear_count": 0,
                "neutral_count": 0,
            },
        )
        g["stock_set"].add(str(r["ticker"] or "").upper())
        g["hit_count"] += 1
        g["score"] += score
        if note_date and (not g["recent_date"] or note_date > g["recent_date"]):
            g["recent_date"] = note_date
        impact = str(r["impact"] or "")
        if "利多" in impact:
            g["bull_count"] += 1
        elif "利空" in impact:
            g["bear_count"] += 1
        else:
            g["neutral_count"] += 1

    out = []
    for v in grouped.values():
        out.append(
            {
                "theme": v["theme"],
                "stock_count": len(v["stock_set"]),
                "hit_count": int(v["hit_count"]),
                "heat_score": round(float(v["score"]), 4),
                "recent_date": v["recent_date"],
                "bull_count": int(v["bull_count"]),
                "bear_count": int(v["bear_count"]),
                "neutral_count": int(v["neutral_count"]),
            }
        )
    out.sort(key=lambda x: (x["heat_score"], x["stock_count"], x["hit_count"]), reverse=True)
    return out


def delete_event_theme_hits_by_theme(theme: str, db_path: Path = DB_PATH) -> int:
    t = str(theme or "").strip()
    if not t:
        return 0
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute("DELETE FROM event_theme_hits WHERE theme = ?", (t,))
        conn.commit()
        return int(cur.rowcount or 0)


def clear_all_event_theme_hits(db_path: Path = DB_PATH) -> int:
    with closing(get_conn(db_path)) as conn:
        cur = conn.execute("DELETE FROM event_theme_hits")
        conn.commit()
        return int(cur.rowcount or 0)


def list_keyword_synonyms(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM keyword_synonyms ORDER BY enabled DESC, keyword ASC"
        ).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["expansions_list"] = _parse_csv(r.get("expansions", ""))
        r["enabled"] = bool(int(r.get("enabled", 1)))
    return out


def upsert_keyword_synonym(keyword: str, expansions: list[str], enabled: bool = True, db_path: Path = DB_PATH) -> None:
    k = str(keyword or "").strip()
    if not k:
        return
    ex = _csv(expansions)
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO keyword_synonyms(keyword, expansions, enabled, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(keyword) DO UPDATE SET expansions=excluded.expansions, enabled=excluded.enabled, updated_at=excluded.updated_at
            """,
            (k, ex, 1 if enabled else 0, _now()),
        )
        conn.commit()


def delete_keyword_synonym(keyword: str, db_path: Path = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute("DELETE FROM keyword_synonyms WHERE keyword = ?", (str(keyword).strip(),))
        conn.commit()


def list_theme_rules(db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    with closing(get_conn(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM theme_canonical_rules ORDER BY enabled DESC, raw_theme ASC"
        ).fetchall()
    out = [dict(r) for r in rows]
    for r in out:
        r["enabled"] = bool(int(r.get("enabled", 1)))
    return out


def upsert_theme_rule(raw_theme: str, canonical_theme: str, enabled: bool = True, db_path: Path = DB_PATH) -> None:
    raw = str(raw_theme or "").strip()
    can = str(canonical_theme or "").strip()
    if not raw or not can:
        return
    with closing(get_conn(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO theme_canonical_rules(raw_theme, canonical_theme, enabled, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(raw_theme) DO UPDATE SET canonical_theme=excluded.canonical_theme, enabled=excluded.enabled, updated_at=excluded.updated_at
            """,
            (raw, can, 1 if enabled else 0, _now()),
        )
        conn.commit()


def delete_theme_rule(raw_theme: str, db_path: Path = DB_PATH) -> None:
    with closing(get_conn(db_path)) as conn:
        conn.execute("DELETE FROM theme_canonical_rules WHERE raw_theme = ?", (str(raw_theme).strip(),))
        conn.commit()


def get_event_min_confidence(db_path: Path = DB_PATH) -> float:
    try:
        return float(get_meta("event_min_confidence", "0.55", db_path=db_path))
    except Exception:
        return 0.55


def set_event_min_confidence(v: float, db_path: Path = DB_PATH) -> None:
    vv = max(0.0, min(1.0, float(v)))
    set_meta("event_min_confidence", str(vv), db_path=db_path)


def export_all_as_dict(db_path: Path = DB_PATH) -> dict[str, Any]:
    with closing(get_conn(db_path)) as conn:
        stock_rows = [dict(r) for r in conn.execute("SELECT * FROM stock_notes ORDER BY id ASC").fetchall()]
        macro_rows = [dict(r) for r in conn.execute("SELECT * FROM macro_notes ORDER BY id ASC").fetchall()]
        theme_rows = [dict(r) for r in conn.execute("SELECT * FROM event_theme_hits ORDER BY id ASC").fetchall()]
        kw_rows = [dict(r) for r in conn.execute("SELECT * FROM keyword_synonyms ORDER BY id ASC").fetchall()]
        rule_rows = [dict(r) for r in conn.execute("SELECT * FROM theme_canonical_rules ORDER BY id ASC").fetchall()]
        meta_rows = [dict(r) for r in conn.execute("SELECT * FROM app_meta ORDER BY key ASC").fetchall()]
    return {
        "stock_notes": stock_rows,
        "macro_notes": macro_rows,
        "event_theme_hits": theme_rows,
        "keyword_synonyms": kw_rows,
        "theme_canonical_rules": rule_rows,
        "app_meta": meta_rows,
    }


def import_all_from_dict(obj: dict[str, Any], db_path: Path = DB_PATH, replace: bool = True) -> None:
    stock_rows = obj.get("stock_notes", []) or []
    macro_rows = obj.get("macro_notes", []) or []
    theme_rows = obj.get("event_theme_hits", []) or []
    kw_rows = obj.get("keyword_synonyms", []) or []
    rule_rows = obj.get("theme_canonical_rules", []) or []
    meta_rows = obj.get("app_meta", []) or []
    with closing(get_conn(db_path)) as conn:
        if replace:
            conn.execute("DELETE FROM stock_notes")
            conn.execute("DELETE FROM macro_notes")
            conn.execute("DELETE FROM event_theme_hits")
            conn.execute("DELETE FROM keyword_synonyms")
            conn.execute("DELETE FROM theme_canonical_rules")
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
        for r in kw_rows:
            conn.execute(
                """
                INSERT INTO keyword_synonyms(id, keyword, expansions, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    r.get("id"),
                    str(r.get("keyword", "")),
                    str(r.get("expansions", "")),
                    int(r.get("enabled", 1) or 1),
                    str(r.get("updated_at", _now())),
                ),
            )
        for r in rule_rows:
            conn.execute(
                """
                INSERT INTO theme_canonical_rules(id, raw_theme, canonical_theme, enabled, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    r.get("id"),
                    str(r.get("raw_theme", "")),
                    str(r.get("canonical_theme", "")),
                    int(r.get("enabled", 1) or 1),
                    str(r.get("updated_at", _now())),
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
    return json.dumps(export_all_as_dict(db_path=db_path), ensure_ascii=False, indent=2)


def import_json_text(text: str, db_path: Path = DB_PATH, replace: bool = True) -> None:
    payload = json.loads(text)
    import_all_from_dict(payload, db_path=db_path, replace=replace)
