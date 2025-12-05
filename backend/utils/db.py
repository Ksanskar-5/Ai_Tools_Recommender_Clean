"""
db.py — MySQL integration for AI Recommender
Handles users, AI data, logs, and feedback.
"""

import mysql.connector
from mysql.connector import Error
import hashlib
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ======================================================
# MySQL CONNECTION
# ======================================================
def get_connection():
    """Establish a MySQL connection."""
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="San777@tyhfs",
        database="ai_recommender"
    )


def _hash(password: str) -> str:
    """Return SHA-256 hash of password."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


# ======================================================
# USER AUTHENTICATION
# ======================================================
def verify_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """Verify email + password and return user info."""
    try:
        conn = get_connection()
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        row = cur.fetchone()

        if row and row.get("password_hash") == _hash(password):
            return {
                "user_id": row["user_id"],
                "name": row["name"],
                "email": row["email"]
            }
        return None

    except Error as e:
        logger.error(f"Error verifying user: {e}")
        return None

    finally:
        if conn: conn.close()


def register_user(name: str, email: str, password: str) -> Optional[int]:
    """Register a new user. Enforces unique email."""
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Unique email constraint
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255) UNIQUE,
                password_hash VARCHAR(255)
            )
        """)

        cur.execute("""
            INSERT INTO users (name, email, password_hash)
            VALUES (%s, %s, %s)
        """, (name, email, _hash(password)))

        conn.commit()
        return cur.lastrowid

    except Error as e:
        logger.error(f"Error registering user: {e}")
        return None

    finally:
        if conn: conn.close()


# ======================================================
# AI DATA
# ======================================================
def get_all_ai() -> List[Dict[str, Any]]:
    """Fetch all AI tools."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM ai_data")
    rows = cur.fetchall()
    conn.close()
    return rows


# ======================================================
# USER LOGS (implicit feedback)
# ======================================================
def log_user_action(user_id: int, ai_id: int, query: str, action: str,
                    feedback: Optional[float] = None,
                    dwell_time: Optional[int] = None):

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT,
                ai_id INT,
                query TEXT,
                action VARCHAR(30),
                feedback FLOAT,
                dwell_time INT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            INSERT INTO user_data (user_id, ai_id, query, action, feedback, dwell_time)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (user_id, ai_id, query, action, feedback, dwell_time))

        conn.commit()

    except Exception as e:
        logger.error(f"Failed to log user action: {e}")

    finally:
        if conn: conn.close()


# ======================================================
# BOOKMARKS
# ======================================================
def get_bookmarks(user_id: int):
    """Return bookmarked AI tools."""
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT a.*
        FROM ai_data a
        JOIN feedback f ON f.ai_id = a.ai_id
        WHERE f.user_id = %s AND f.bookmark = 1
    """, (user_id,))

    rows = cur.fetchall()
    conn.close()
    return rows


# ======================================================
# FEEDBACK (likes, dislikes, rating, comment, bookmark)
# ======================================================
def save_feedback(
    user_id: int,
    ai_id: int,
    feedback: Optional[int] = None,
    rating: Optional[int] = None,
    comment: Optional[str] = None,
    sentiment: Optional[float] = None,
    bookmark: Optional[int] = None
):
    """
    Safely upserts feedback without overwriting fields with NULL.
    """
    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                user_id INT,
                ai_id INT,
                feedback INT,
                rating TINYINT NULL,
                comment TEXT NULL,
                sentiment FLOAT NULL,
                bookmark TINYINT DEFAULT 0,
                PRIMARY KEY (user_id, ai_id)
            )
        """)

        # normalize feedback (1 / -1)
        stored_feedback = 1 if feedback == 1 else (-1 if feedback == -1 else None)

        # build conditional update query
        updates = []
        params = []

        if stored_feedback is not None:
            updates.append("feedback = %s")
            params.append(stored_feedback)

        if rating is not None:
            updates.append("rating = %s")
            params.append(rating)

        if comment is not None:
            updates.append("comment = %s")
            params.append(comment)

        if sentiment is not None:
            updates.append("sentiment = %s")
            params.append(sentiment)

        if bookmark is not None:
            updates.append("bookmark = %s")
            params.append(bookmark)

        # If no updates requested → do nothing
        if not updates:
            return

        # Insert placeholder row if not exists
        cur.execute("""
            INSERT IGNORE INTO feedback (user_id, ai_id)
            VALUES (%s, %s)
        """, (user_id, ai_id))

        # Apply updates safely
        sql = f"""
            UPDATE feedback
            SET {', '.join(updates)}
            WHERE user_id = %s AND ai_id = %s
        """

        params.extend([user_id, ai_id])
        cur.execute(sql, params)

        conn.commit()

    except Exception as e:
        logger.error(f"MySQL feedback save failed: {e}")
        raise

    finally:
        if conn: conn.close()


# ======================================================
# EXPLICIT FEEDBACK (likes/dislikes)
# ======================================================
def get_user_feedback(user_id: int) -> Dict[int, float]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT ai_id, feedback
        FROM feedback
        WHERE user_id = %s AND feedback IS NOT NULL
    """, (user_id,))

    rows = cur.fetchall()
    conn.close()

    return {r["ai_id"]: r["feedback"] for r in rows}


# ======================================================
# IMPLICIT FEEDBACK
# ======================================================
def get_user_actions(user_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("""
        SELECT ai_id, action, dwell_time, query, timestamp
        FROM user_data
        WHERE user_id = %s
    """, (user_id,))

    rows = cur.fetchall()
    conn.close()
    return rows


# ======================================================
# COLLABORATIVE FILTERING SUPPORT
# ======================================================
def get_similar_users_feedback(user_id: int) -> List[tuple]:
    conn = get_connection()
    cur = conn.cursor(dictionary=True)

    cur.execute("SELECT DISTINCT user_id FROM feedback WHERE user_id != %s", (user_id,))
    other_users = [r["user_id"] for r in cur.fetchall()]

    conn.close()

    base = get_user_feedback(user_id)
    results = []

    for other in other_users:
        f2 = get_user_feedback(other)
        common = set(base) & set(f2)

        if not common:
            continue

        dot = sum(base[i] * f2[i] for i in common)
        denom = (len(base) * len(f2)) ** 0.5
        sim = dot / denom if denom else 0

        results.append((other, sim, f2))

    return sorted(results, key=lambda x: x[1], reverse=True)[:5]
