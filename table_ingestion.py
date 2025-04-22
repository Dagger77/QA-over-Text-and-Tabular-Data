import pandas as pd
import sqlite3

table_path = "./knowledgebase-data"

# Load CSVs
df1 = pd.read_csv(table_path + '/Expanded_data_with_more_features.csv')
df2 = pd.read_csv(table_path + '/Original_data_with_more_rows.csv')
df1 = df1.drop(df1.columns[0], axis=1)
df2 = df2.drop(df2.columns[0], axis=1)

conn = sqlite3.connect("student_data.db")

df1.to_sql("student_info_detailed", conn, if_exists="replace", index=False)
df2.to_sql("student_info_basic", conn, if_exists="replace", index=False)

conn.commit()
conn.close()
print("Tables loaded into SQLite successfully.")