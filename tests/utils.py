import sqlite3

REQUIRED_TABLES = {"student_info_basic", "student_info_detailed"}


def missing_required_tables(db_path="student_data.db") -> set:
    """Check for missing required tables in SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        return REQUIRED_TABLES - existing_tables
    except sqlite3.Error as e:
        print(f"[DB Check Error] {e}")
        return REQUIRED_TABLES
