import hashlib, json, sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_logs(game_id TEXT, game_date TEXT, player TEXT, player_key TEXT,
  team TEXT, opp TEXT, home INT, min REAL, pts REAL, reb REAL, ast REAL, stl REAL, blk REAL, tov REAL,
  fg3m REAL, fg3a REAL, fga REAL, fta REAL, oreb REAL, starter INT, PRIMARY KEY(game_id, player_key));
CREATE TABLE IF NOT EXISTS events(game_key TEXT PRIMARY KEY, odds_event_id TEXT, home TEXT, away TEXT, tipoff TEXT);
CREATE TABLE IF NOT EXISTS odds_snapshots(fetched_at TEXT, game_key TEXT, book TEXT, market TEXT,
  player TEXT, side TEXT, line REAL, odds REAL);
CREATE INDEX IF NOT EXISTS ix_odds ON odds_snapshots(game_key, fetched_at);
CREATE TABLE IF NOT EXISTS injuries(fetched_at TEXT, team TEXT, player_key TEXT, player TEXT, status TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS predictions(id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, created_at TEXT,
  tipoff TEXT, game_key TEXT, market TEXT, player TEXT, side TEXT, line REAL, book TEXT, odds REAL,
  implied REAL, fair REAL, model_p REAL, p_use REAL, push_p REAL, ev REAL, edge_pp REAL, proj REAL,
  proj_min REAL, decision TEXT, stake_frac REAL, reasons TEXT, card TEXT, prev_hash TEXT, hash TEXT);
CREATE TABLE IF NOT EXISTS results(pred_id INTEGER PRIMARY KEY, settled_at TEXT, actual REAL, outcome TEXT, clv_pp REAL);
CREATE TABLE IF NOT EXISTS period_scores(game_date TEXT, team TEXT, opp TEXT, home INT, q1 REAL, q2 REAL, q3 REAL, q4 REAL, ot REAL, PRIMARY KEY(game_date, team));
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""
FMT = "%Y-%m-%dT%H:%M:%SZ"
def iso(dt): return dt.astimezone(timezone.utc).strftime(FMT)
def parse(s): return datetime.strptime(s, FMT).replace(tzinfo=timezone.utc)
def now_utc(): return datetime.now(timezone.utc)

def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con

def get_meta(con, k):
    r = con.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    return r[0] if r else None
def set_meta(con, k, v):
    con.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, str(v))); con.commit()

HASHED = ["created_at", "tipoff", "game_key", "market", "player", "side", "line", "book", "odds", "model_p", "decision"]
def _digest(prev, row):
    return hashlib.sha256((prev + json.dumps({k: row[k] for k in HASHED}, sort_keys=True, default=str)).encode()).hexdigest()

def insert_prediction(con, row):
    """Append-only; each row's hash covers the previous row's hash (tamper-evident log)."""
    if row["created_at"] >= row["tipoff"]:
        raise ValueError("refusing to record a prediction at or after tipoff")
    prev = con.execute("SELECT hash FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
    row["prev_hash"] = prev[0] if prev else ""
    row["hash"] = _digest(row["prev_hash"], row)
    cols = [r[1] for r in con.execute("PRAGMA table_info(predictions)") if r[1] != "id"]
    con.execute(f"INSERT INTO predictions({','.join(cols)}) VALUES({','.join('?'*len(cols))})", [row.get(c) for c in cols])

def verify_chain(con):
    prev = ""
    for r in con.execute("SELECT * FROM predictions ORDER BY id"):
        if r["prev_hash"] != prev or r["hash"] != _digest(prev, r):
            return False, r["id"]
        prev = r["hash"]
    return True, None
