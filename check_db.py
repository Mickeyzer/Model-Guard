import sqlite3, os

db = 'data/modelguard.db'
print('DB exists:', os.path.exists(db))
con = sqlite3.connect(db)
cur = con.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print('Tables:', tables)

for t in tables:
    cur.execute(f"SELECT COUNT(*) FROM {t}")
    print(f"  {t}: {cur.fetchone()[0]} rows")

if 'defense_log' in tables:
    cur.execute("SELECT * FROM defense_log ORDER BY timestamp DESC LIMIT 5")
    rows = cur.fetchall()
    print("Last 5 defense_log rows:")
    for r in rows:
        print(" ", r)

con.close()
