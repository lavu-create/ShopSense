"""
ShopSense Disk-Backed SQLite Catalog
====================================
Provides memory-efficient, indexed metadata lookups without loading
the full 213k-product catalog into RAM at API startup.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Any, Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_clean.parquet"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "processed" / "catalog.db"


def build_catalog_db(
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    db_path: Path = DEFAULT_DB_PATH,
    overwrite: bool = False,
) -> Path:
    """
    Builds the SQLite catalog database from cleaned metadata parquet.
    Creates an indexed B-tree on parent_asin (PRIMARY KEY) and main_category.
    """
    if db_path.exists() and not overwrite and db_path.stat().st_size > 1024 * 1024:
        return db_path

    if not parquet_path.exists():
        raise FileNotFoundError(f"Metadata parquet not found at {parquet_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(parquet_path)

    cols = [
        "parent_asin", "title", "main_category", "categories_flat",
        "price", "store", "average_rating", "rating_number"
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS catalog (
        parent_asin TEXT PRIMARY KEY,
        title TEXT,
        main_category TEXT,
        categories_flat TEXT,
        price REAL,
        store TEXT,
        average_rating REAL,
        rating_number INTEGER
    )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_catalog_category ON catalog(main_category)")

    records = df[cols].values.tolist()
    cur.executemany("""
    INSERT OR REPLACE INTO catalog (
        parent_asin, title, main_category, categories_flat,
        price, store, average_rating, rating_number
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, records)

    conn.commit()
    conn.close()
    return db_path


class CatalogDB:
    """
    Thread-safe, read-only interface to the SQLite catalog database.
    Supports single and batch lookups for product enrichment and category routing.
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists() or self.db_path.stat().st_size < 1024:
            build_catalog_db(db_path=self.db_path)
        self._local = threading.local()

    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            uri = f"file:{self.db_path.resolve()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def get_item(self, parent_asin: str) -> Optional[Dict[str, Any]]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT parent_asin, title, main_category, categories_flat, price "
            "FROM catalog WHERE parent_asin = ?",
            (parent_asin,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)

    def get_items(self, parent_asins: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        if not parent_asins:
            return {}
        conn = self._get_connection()
        cur = conn.cursor()
        asins_list = list(parent_asins)
        res: Dict[str, Dict[str, Any]] = {}
        for i in range(0, len(asins_list), 900):
            chunk = asins_list[i : i + 900]
            placeholders = ",".join(["?"] * len(chunk))
            cur.execute(
                f"SELECT parent_asin, title, main_category, categories_flat, price "
                f"FROM catalog WHERE parent_asin IN ({placeholders})",
                chunk,
            )
            for row in cur.fetchall():
                res[row["parent_asin"]] = dict(row)
        return res

    def has_item(self, parent_asin: str) -> bool:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM catalog WHERE parent_asin = ? LIMIT 1", (parent_asin,))
        return cur.fetchone() is not None

    def get_category(self, parent_asin: str) -> Optional[str]:
        conn = self._get_connection()
        cur = conn.cursor()
        cur.execute("SELECT main_category FROM catalog WHERE parent_asin = ?", (parent_asin,))
        row = cur.fetchone()
        return row["main_category"] if row and row["main_category"] else None

    def get_categories(self, parent_asins: Sequence[str]) -> Dict[str, str]:
        if not parent_asins:
            return {}
        conn = self._get_connection()
        cur = conn.cursor()
        res: Dict[str, str] = {}
        asins_list = list(parent_asins)
        for i in range(0, len(asins_list), 900):
            chunk = asins_list[i : i + 900]
            placeholders = ",".join(["?"] * len(chunk))
            cur.execute(
                f"SELECT parent_asin, main_category FROM catalog WHERE parent_asin IN ({placeholders})",
                chunk,
            )
            for row in cur.fetchall():
                if row["main_category"]:
                    res[row["parent_asin"]] = row["main_category"]
        return res

    def get(self, parent_asin: str, default: Any = None) -> Any:
        item = self.get_item(parent_asin)
        return item if item is not None else (default if default is not None else {})

    def __contains__(self, parent_asin: str) -> bool:
        return self.has_item(parent_asin)


if __name__ == "__main__":
    db = build_catalog_db(overwrite=True)
    print(f"Catalog DB built successfully: {db} ({db.stat().st_size / (1024*1024):.1f} MB)")
