"""
Loads CSV files and stores them in a local SQLite database for use in the SQL agent.
"""

import os
import sqlite3
import pandas as pd


# ----------------------------
# Config
# ----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "knowledgebase-data")
DB_PATH = os.path.join(BASE_DIR, "student_data.db")


# ----------------------------
# Data Ingestion Logic
# ----------------------------
def load_data():
    """
    Loads two CSV files and inserts them into a SQLite database as separate tables.
    """
    print("Loading CSV files...")

    try:
        df_detailed = pd.read_csv(os.path.join(DATA_DIR, "Expanded_data_with_more_features.csv")).iloc[:, 1:]
        df_basic = pd.read_csv(os.path.join(DATA_DIR, "Original_data_with_more_rows.csv")).iloc[:, 1:]
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return

    print("Connecting to SQLite...")
    try:
        conn = sqlite3.connect(DB_PATH)

        df_detailed.to_sql("student_info_detailed", conn, if_exists="replace", index=False)
        df_basic.to_sql("student_info_basic", conn, if_exists="replace", index=False)

        conn.commit()
        print("Tables loaded into SQLite successfully.")
    except Exception as e:
        print(f"Error loading tables into database: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    load_data()
