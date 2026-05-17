

import sqlite3
import hashlib
import pandas as pd
from datetime import datetime

DB_PATH = "traffic_app.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            email       TEXT,
            role        TEXT DEFAULT "user",
            created_at  TEXT
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id          INTEGER,
            filename         TEXT,
            total_frames     INTEGER,
            max_vehicles     INTEGER,
            congested_frames INTEGER,
            congestion_rate  REAL,
            lat              REAL  DEFAULT 22.5726,
            lng              REAL  DEFAULT 88.3639,
            location_name    TEXT  DEFAULT "Kolkata",
            threshold        INTEGER DEFAULT 15,
            timestamp        TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    try:
        hashed = _hash(  "admin123")
        c.execute('''
            INSERT INTO users (username, password, email, role, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', ("admin", hashed, "admin@traffic.com", "admin",
              datetime.now().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  

    conn.close()


def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username: str, password: str, email: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute('''
            INSERT INTO users (username, password, email, role, created_at)
            VALUES (?, ?, ?, "user", ?)
        ''', (username, _hash(password), email, datetime.now().isoformat()))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError:
        return False, "Username already exists!"
    finally:
        conn.close()


def login_user(username: str, password: str):
    """Returns (id, username, role) tuple or None."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        SELECT id, username, role FROM users
        WHERE username = ? AND password = ?
    ''', (username, _hash(password)))
    user = c.fetchone()
    conn.close()
    return user


def log_detection(user_id, filename, total_frames, max_vehicles,
                  congested_frames, congestion_rate,
                  lat, lng, location_name, threshold):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO detection_logs
            (user_id, filename, total_frames, max_vehicles,
             congested_frames, congestion_rate,
             lat, lng, location_name, threshold, timestamp)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    ''', (user_id, filename, total_frames, max_vehicles,
          congested_frames, congestion_rate,
          lat, lng, location_name, threshold,
          datetime.now().isoformat()))
    conn.commit()
    conn.close()


def get_user_history(user_id) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('''
            SELECT * FROM detection_logs
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', conn, params=(user_id,))
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_all_logs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('''
            SELECT dl.*, u.username
            FROM detection_logs dl
            JOIN users u ON dl.user_id = u.id
            ORDER BY dl.timestamp DESC
        ''', conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def get_all_users() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query('''
            SELECT id, username, email, role, created_at
            FROM users
            ORDER BY created_at DESC
        ''', conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


def delete_user(user_id: int):
    """Delete a non-admin user."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE id = ? AND role != 'admin'", (user_id,))
    conn.commit()
    conn.close()
