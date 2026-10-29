"""Data collection. Primary: nba_api (official stats.nba.com). Fallback: ESPN. Odds: The Odds API."""
import json, re, time, unicodedata, urllib.parse, urllib.request
from datetime import timedelta
import pandas as pd
from . import config as C, store

ALIASES = {"LA Clippers": "Los Angeles Clippers"}
def norm_team(n): return ALIASES.get(n, n)
_SUFFIX = {"jr", "sr", "ii", "iii", "iv", "v"}
def pkey(name):
    s = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode().lower()
    return " ".join(p for p in re.sub(r"[^a-z ]", " ", s).split() if p not in _SUFFIX)

def season_str(now):
    y = now.year if now.month >= 10 else now.year - 1
    return f"{y}-{str(y + 1)[2:]}"

def get_json(url, params=None, retries=3, timeout=25):
    if params: url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.load(r)
        except Exception:
            if i == retries - 1: raise
            time.sleep(2 ** i)

LOG_COLS = ["game_id", "game_date", "player", "player_key", "team", "opp", "home", "min", "pts", "reb",
            "ast", "stl", "blk", "tov", "fg3m", "fg3a", "fga", "fta", "oreb", "starter"]

# ---------------- box scores: official NBA stats ----------------
def logs_nba_api(season):
    from nba_api.stats.endpoints import leaguegamelog
    df = leaguegamelog.LeagueGameLog(season=season, player_or_team_abbreviation="P",
                                     season_type_all_star="Regular Season", timeout=45).get_data_frames()[0]
    abbr = dict(zip(df.TEAM_ABBREVIATION, df.TEAM_NAME))
    df = df[df.MIN.fillna(0) > 0].copy()
    opp = df.MATCHUP.str.extract(r"(?:vs\.|@)\s*(\w+)")[0].map(abbr)
    out = pd.DataFrame({
        "game_id": df.GAME_ID.astype(str), "game_date": pd.to_datetime(df.GAME_DATE).dt.strftime("%Y-%m-%d"),
        "player": df.PLAYER_NAME, "player_key": df.PLAYER_NAME.map(pkey), "team": df.TEAM_NAME.map(norm_team),
        "opp": opp.map(lambda x: norm_team(x) if isinstance(x, str) else None),
        "home": (~df.MATCHUP.str.contains("@")).astype(int), "min": df.MIN.astype(float),
        "pts": df.PTS, "reb": df.REB, "ast": df.AST, "stl": df.STL, "blk": df.BLK, "tov": df.TOV,
        "fg3m": df.FG3M, "fg3a": df.FG3A, "fga": df.FGA, "fta": df.FTA, "oreb": df.OREB, "starter": None})
    return out.dropna(subset=["opp"])[LOG_COLS]

# ---------------- box scores: ESPN fallback (best effort) ----------------
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
def _split(s):
    try: a, b = str(s).split("-"); return float(a), float(b)
    except Exception: return 0.0, 0.0

def logs_espn(dates):
    rows = []
    for d in dates:
        sb = get_json(f"{ESPN}/scoreboard", {"dates": d.replace("-", "")})
        for ev in sb.get("events", []):
            if not ev["status"]["type"].get("completed"): continue
            s = get_json(f"{ESPN}/summary", {"event": ev["id"]})
            names = [norm_team(t["team"]["displayName"]) for t in s["boxscore"]["players"]]
            home = {norm_team(c["team"]["displayName"]): c["homeAway"] == "home"
                    for c in s["header"]["competitions"][0]["competitors"]}
            for tb, team in zip(s["boxscore"]["players"], names):
                opp = [n for n in names if n != team][0]
                for grp in tb["statistics"]:
                    lab = grp.get("labels") or []
                    for a in grp.get("athletes", []):
                        if a.get("didNotPlay") or not a.get("stats"): continue
                        v = dict(zip(lab, a["stats"]))
                        try: mins = float(v.get("MIN", 0))
                        except ValueError: continue
                        if mins <= 0: continue
                        fgm, fga = _split(v.get("FG")); m3, a3 = _split(v.get("3PT")); _, fta = _split(v.get("FT"))
                        nm = a["athlete"]["displayName"]
                        f = lambda k: float(v.get(k, 0) or 0)
                        rows.append([ev["id"], d, nm, pkey(nm), team, opp, int(home.get(team, False)), mins,
                                     f("PTS"), f("REB"), f("AST"), f("STL"), f("BLK"), f("TO"), m3, a3, fga, fta,
                                     f("OREB"), int(bool(a.get("starter")))])
    return pd.DataFrame(rows, columns=LOG_COLS)

def update_logs(con, now, max_age_hours=6, force=False):
    last = store.get_meta(con, "logs_updated")
    if last and not force and (now - store.parse(last)).total_seconds() < max_age_hours * 3600:
        return "logs fresh"
    frames, src = [], "nba_api"
    seasons = [season_str(now)]
    if con.execute("SELECT COUNT(*) FROM player_logs").fetchone()[0] == 0:
        y = int(seasons[0][:4]); seasons.insert(0, f"{y-1}-{str(y)[2:]}")
    try:
        for s in seasons: frames.append(logs_nba_api(s))
    except Exception as e:
        src = f"espn fallback ({type(e).__name__})"
        mx = con.execute("SELECT MAX(game_date) FROM player_logs").fetchone()[0]
        start = (pd.Timestamp(mx) + pd.Timedelta(days=1)) if mx else (pd.Timestamp(now.date()) - pd.Timedelta(days=14))
        days = pd.date_range(start, pd.Timestamp(now.date()) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        frames = [logs_espn(list(days))]
    df = pd.concat(frames) if frames else pd.DataFrame(columns=LOG_COLS)
    if len(df):
        con.executemany(f"INSERT OR REPLACE INTO player_logs VALUES({','.join('?'*len(LOG_COLS))})",
                        df[LOG_COLS].astype(object).where(df[LOG_COLS].notna(), None).values.tolist())
        con.commit()
    store.set_meta(con, "logs_updated", store.iso(now))
    return f"logs updated via {src}: {len(df)} rows"

# ---------------- injuries ----------------
def update_injuries(con, now):
    data = get_json(f"{ESPN}/injuries")
    ts, rows = store.iso(now), []
    for t in data.get("injuries", []):
        team = norm_team(t.get("displayName", ""))
        for i in t.get("injuries", []):
            nm = i.get("athlete", {}).get("displayName", "")
            rows.append((ts, team, pkey(nm), nm, str(i.get("status", "")).lower(),
                         i.get("shortComment") or i.get("longComment") or ""))
    con.executemany("INSERT INTO injuries VALUES(?,?,?,?,?,?)", rows); con.commit()
    return f"{len(rows)} injury entries"

def latest_injuries(con):
    ts = con.execute("SELECT MAX(fetched_at) FROM injuries").fetchone()[0]
    out = {}
    if ts:
        for r in con.execute("SELECT * FROM injuries WHERE fetched_at=?", (ts,)):
            s = r["status"]
            st = ("out" if s in ("out", "suspension") else "doubtful" if "doubt" in s else
                  "probable" if "probable" in s else "questionable")   # day-to-day, questionable, etc.
            out.setdefault(r["team"], {})[r["player_key"]] = (st, r["detail"])
    return out

# ---------------- odds ----------------
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
def refresh_events(con):
    n = 0
    for e in get_json(f"{BASE}/events", {"apiKey": C.ODDS_API_KEY}):
        tip = store.parse(e["commence_time"].replace("Z", "Z")[:19] + "Z")
        home, away = norm_team(e["home_team"]), norm_team(e["away_team"])
        gk = f"{tip.astimezone(ET).strftime('%Y-%m-%d')}:{away}@{home}"
        con.execute("INSERT OR REPLACE INTO events VALUES(?,?,?,?,?)", (gk, e["id"], home, away, store.iso(tip))); n += 1
    con.commit(); return n

from zoneinfo import ZoneInfo
ET = ZoneInfo("America/New_York")

def parse_odds(data, game_key, home, fetched_at):
    rows = []
    for bk in data.get("bookmakers", []):
        for m in bk.get("markets", []):
            mk = m["key"]; base, _per = C.split_market(mk)
            if any(o["name"] == "Draw" for o in m["outcomes"]): continue      # 3-way market: 2-way devig would be wrong
            for o in m["outcomes"]:
                player = ""
                if mk in C.PROP_MARKETS: player, side, gl = o.get("description", ""), o["name"], o.get("point")
                elif base == "h2h": side, gl = norm_team(o["name"]), 0.0
                elif base == "spreads":
                    side = norm_team(o["name"]); gl = o["point"] if side == home else -o["point"]
                elif base == "totals": side, gl = o["name"], o.get("point")
                else: continue
                if gl is None: continue
                rows.append((fetched_at, game_key, bk["key"], mk, player, side, float(gl), float(o["price"])))
    return rows

def poll_odds(con, ev, now):
    ts, rows = store.iso(now), []
    d = get_json(f"{BASE}/events/{ev['odds_event_id']}/odds", {"apiKey": C.ODDS_API_KEY, "regions": C.REGIONS,
                 "markets": ",".join(C.GAME_MARKETS + C.PERIOD_MARKETS + list(C.PROP_MARKETS)), "oddsFormat": "decimal"})
    rows = parse_odds(d, ev["game_key"], ev["home"], ts)
    con.executemany("INSERT INTO odds_snapshots VALUES(?,?,?,?,?,?,?,?)", rows); con.commit()
    return len(rows)

def poll_interval_minutes(hours_to_tip):
    """Cheap early, frequent near tipoff (props cost 1 API credit per market per region per game)."""
    if hours_to_tip > 14: return None
    if hours_to_tip > 4: return 120
    if hours_to_tip > 1: return 30
    return 10

def update_period_scores(con, now):
    """Quarter scores from the ESPN scoreboard (one call per date). Needed to settle and to learn period scoring splits."""
    mx = con.execute("SELECT MAX(game_date) FROM period_scores").fetchone()[0]
    end = pd.Timestamp(now.astimezone(ET).date()) - pd.Timedelta(days=1)
    start = pd.Timestamp(mx) + pd.Timedelta(days=1) if mx else end - pd.Timedelta(days=30)
    n = 0
    for d in pd.date_range(start, end).strftime("%Y-%m-%d"):
        for ev in get_json(f"{ESPN}/scoreboard", {"dates": d.replace("-", "")}).get("events", []):
            if not ev["status"]["type"].get("completed"): continue
            comp = ev["competitions"][0]["competitors"]; teams = [norm_team(c["team"]["displayName"]) for c in comp]
            for c, opp in zip(comp, reversed(teams)):
                ls = [float(x.get("value", 0)) for x in c.get("linescores", [])]
                if len(ls) < 4: continue
                con.execute("INSERT OR REPLACE INTO period_scores VALUES(?,?,?,?,?,?,?,?,?)",
                            (d, norm_team(c["team"]["displayName"]), opp, int(c["homeAway"] == "home"), *ls[:4], sum(ls[4:]))); n += 1
    con.commit(); return f"{n} team period-score rows"
