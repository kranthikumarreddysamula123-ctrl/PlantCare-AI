
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE = os.path.join(BASE_DIR, "prediction_history.db")


def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    connection = get_connection()

    connection.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        )
    """)

    connection.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            heatmap_path TEXT,
            overlay_path TEXT,
            created_at TEXT NOT NULL,
            user_id INTEGER REFERENCES users(id)
        )
    """)

    # Add user_id to an existing predictions table if missing
    columns = [
        row["name"]
        for row in connection.execute(
            "PRAGMA table_info(predictions)"
        ).fetchall()
    ]

    if "user_id" not in columns:
        connection.execute("""
            ALTER TABLE predictions
            ADD COLUMN user_id INTEGER REFERENCES users(id)
        """)

    connection.commit()
    connection.close()


def register_user(username, email, password, role="user"):
    connection = get_connection()

    try:
        hashed_password = generate_password_hash(password)

        connection.execute("""
            INSERT INTO users (
                username, email, password, role, created_at
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            username,
            email,
            hashed_password,
            role,
            datetime.now().strftime("%d-%m-%Y %I:%M %p")
        ))

        connection.commit()
        return True

    except sqlite3.IntegrityError:
        return False

    finally:
        connection.close()


def login_user(email, password):
    connection = get_connection()

    user = connection.execute("""
        SELECT *
        FROM users
        WHERE email = ?
    """, (email,)).fetchone()

    connection.close()

    if user and check_password_hash(user["password"], password):
        return user

    return None


def get_user_by_id(user_id):
    connection = get_connection()

    user = connection.execute("""
        SELECT id, username, email, role
        FROM users
        WHERE id = ?
    """, (user_id,)).fetchone()

    connection.close()
    return user


def save_prediction(
    image_path,
    disease,
    confidence,
    heatmap_path,
    overlay_path,
    user_id=None
):
    connection = get_connection()

    connection.execute("""
        INSERT INTO predictions (
            image_path,
            disease,
            confidence,
            heatmap_path,
            overlay_path,
            created_at,
            user_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        image_path,
        disease,
        confidence,
        heatmap_path,
        overlay_path,
        datetime.now().strftime("%d-%m-%Y %I:%M %p"),
        user_id
    ))

    connection.commit()
    connection.close()


def get_prediction_history(user_id=None, is_admin=False):
    connection = get_connection()

    if is_admin:
        records = connection.execute("""
            SELECT predictions.*, users.username
            FROM predictions
            LEFT JOIN users ON predictions.user_id = users.id
            ORDER BY predictions.id DESC
        """).fetchall()

    elif user_id is not None:
        records = connection.execute("""
            SELECT *
            FROM predictions
            WHERE user_id = ?
            ORDER BY id DESC
        """, (user_id,)).fetchall()

    else:
        records = []

    connection.close()
    return records


def get_all_users():
    connection = get_connection()

    users = connection.execute("""
        SELECT id, username, email, role, created_at
        FROM users
        ORDER BY id DESC
    """).fetchall()

    connection.close()
    return users
