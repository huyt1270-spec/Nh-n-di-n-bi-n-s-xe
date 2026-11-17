"""Lưu trữ và tra cứu kết quả nhận diện bằng SQLite."""
import datetime as dt
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .plate_rules import normalize

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plate       TEXT NOT NULL,
    raw_text    TEXT,
    ocr_conf    REAL,
    det_conf    REAL,
    valid       INTEGER,
    province_ok INTEGER,
    source      TEXT,
    crop_path   TEXT,
    created_at  TEXT NOT NULL
)"""


class PlateDB:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def add(self, plate, raw_text="", ocr_conf=0.0, det_conf=0.0, valid=True,
            province_ok=True, source="", crop_path=""):
        now = dt.datetime.now().isoformat(sep=" ", timespec="seconds")
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO detections (plate, raw_text, ocr_conf, det_conf, valid, province_ok,"
                " source, crop_path, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (plate, raw_text, ocr_conf, det_conf, int(valid), int(province_ok),
                 source, str(crop_path), now))
            return cur.lastrowid

    def search(self, plate="", date_from=None, date_to=None, limit=500):
        sql, args = "SELECT * FROM detections WHERE 1=1", []
        if plate:
            sql += " AND REPLACE(REPLACE(REPLACE(plate,'-',''),'.',''),' ','') LIKE ?"
            args.append(f"%{normalize(plate)}%")
        if date_from:
            sql += " AND date(created_at) >= date(?)"
            args.append(str(date_from))
        if date_to:
            sql += " AND date(created_at) <= date(?)"
            args.append(str(date_to))
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._conn() as c:
            return [dict(r) for r in c.execute(sql, args).fetchall()]

    def count(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
