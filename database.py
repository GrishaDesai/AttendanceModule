"""SQLite storage for academy members and their face embeddings."""
import sqlite3
import pickle
import numpy as np

DB_PATH = "members.db"


def init_db(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS members (
            member_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def add_member(conn, member_id: str, name: str, embedding: np.ndarray):
    blob = pickle.dumps(embedding.astype(np.float32))
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO members (member_id, name, embedding) VALUES (?, ?, ?)",
        (member_id, name, blob),
    )
    conn.commit()


def get_all_members(conn):
    """Returns list of (member_id, name, embedding: np.ndarray)."""
    c = conn.cursor()
    c.execute("SELECT member_id, name, embedding FROM members")
    rows = c.fetchall()
    return [(mid, name, pickle.loads(blob)) for mid, name, blob in rows]


def delete_member(conn, member_id: str):
    c = conn.cursor()
    c.execute("DELETE FROM members WHERE member_id = ?", (member_id,))
    conn.commit()