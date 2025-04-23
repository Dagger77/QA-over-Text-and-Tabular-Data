import pandas as pd
import sqlite3
import os


def load_data():
    table_path = os.path.join(os.path.dirname(__file__), "..", "data", "knowledgebase-data")

    # Load CSVs
    df1 = pd.read_csv(os.path.join(table_path, 'Expanded_data_with_more_features.csv'))
    df2 = pd.read_csv(os.path.join(table_path, 'Original_data_with_more_rows.csv'))
    df1 = df1.drop(df1.columns[0], axis=1)
    df2 = df2.drop(df2.columns[0], axis=1)

    db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "student_data.db"))
    conn = sqlite3.connect(db_path)

    df1.to_sql("student_info_detailed", conn, if_exists="replace", index=False)
    df2.to_sql("student_info_basic", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()
    print("Tables loaded into SQLite successfully.")


if __name__ == "__main__":
    load_data()
