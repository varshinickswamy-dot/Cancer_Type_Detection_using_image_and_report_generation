import sqlite3
from datetime import datetime
import os

DB_PATH = "backend/cancer_records.db"


# ---------------------------------------------------
# Create Database + Table
# ---------------------------------------------------

def init_db():
    os.makedirs("backend", exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT,
            cancer_type TEXT,
            result TEXT,
            confidence REAL,
            image_path TEXT,
            gradcam_path TEXT,
            report_path TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


# ---------------------------------------------------
# Insert New Record
# ---------------------------------------------------

def insert_record(patient_name, cancer_type, result, confidence,
                  image_path, gradcam_path, report_path):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            patient_name,
            cancer_type,
            result,
            confidence,
            image_path,
            gradcam_path,
            report_path,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        patient_name,
        cancer_type,
        result,
        confidence,
        image_path,
        gradcam_path,
        report_path,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    conn.commit()
    conn.close()


# ---------------------------------------------------
# Get All Records
# ---------------------------------------------------

def get_all_records():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
    rows = cursor.fetchall()

    conn.close()
    return rows


# ---------------------------------------------------
# Get Single Record
# ---------------------------------------------------

def get_record_by_id(record_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM predictions WHERE id=?", (record_id,))
    row = cursor.fetchone()

    conn.close()
    return row


# ---------------------------------------------------
# Delete Record
# ---------------------------------------------------

def delete_record(record_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("DELETE FROM predictions WHERE id=?", (record_id,))

    conn.commit()
    conn.close()