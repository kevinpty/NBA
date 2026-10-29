"""Offline end-to-end demo on SYNTHETIC data (no network): collect -> predict -> lock -> settle -> evaluate."""
import itertools, os
from datetime import timedelta
import numpy as np, pandas as pd
from . import store, pipeline, report, evaluate, sources
from .sources import pkey, ET

TEAMS = ["Boston Celtics", "Denver Nuggets", "Miami Heat", "Phoenix Suns", "Milwaukee Bucks", "Dallas Mavericks"]
rng = np.random.default_rng(7)
NAMES = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India"]

def make_world():
    world = {}
    for t in TEAMS:
        ps = []
        for i, mpg in enumerate([35, 33, 31, 28, 25, 22, 18, 14, 10]):
            ps.append(dict(name=f"{t.split()[-1]} {NAMES[i]}", min=mpg, pts=.55 * (1.15 - i * .08), reb=.2 * (1 + (i % 3) * .3),
                           ast=.15 * (1 + (i % 2) * .6), fg3m=.07, stl=.03, blk=.02, tov=.07))
        world[t] = dict(players=ps, pace=98 + rng.normal(0, 2), off=0 + rng.normal(0, 3))
    return world

def game_rows(world, gid, date, home, away):
    pace = (world[home]["pace"] + world[away]["pace"]) / 2 + rng.normal(0, 2); rows = []
    for t, o, is_home in [(home, away, 1), (away, home, 0)]:
        for p in world[t]["players"]:
            m = float(np.clip(rng.normal(p["min"], 3), 4, 44)); k = lambda s: float(rng.negative_binomial(15, 15 / (15 + max(p[s] * m * pace / 98 * (1 + world[t]["off"] / 60), 1e-3))))
            pts, fg3m = k("pts"), k("fg3m")
            rows.append(dict(game_id=gid, game_date=date, player=p["name"], player_key=pkey(p["name"]), team=t, opp=o, home=is_home, min=m,
                             pts=pts, reb=k("reb"), ast=k("ast"), stl=k("stl"), blk=k("blk"), tov=k("tov"), fg3m=fg3m, fg3a=fg3m * 2.5,
                             fga=pts * .8, fta=pts * .2, oreb=k("reb") * .25, starter=None))
    return rows

def period_rows(rows, date):
    out = []
    tot = pd.DataFrame(rows).groupby(["team", "opp", "home"]).pts.sum().reset_index()
    for _, r in tot.iterrows():
        q = rng.multinomial(int(round(r.pts)), [.25, .25, .25, .25])
        out.append((date, r.team, r.opp, int(r.home), *[float(x) for x in q], 0.0))
    return out

def synth_odds(world, ev, now, move=0.0):
    """Three books around the truth; bookB deliberately misprices one prop so the agent has something to find."""
    home, away = ev["home"], ev["away"]; rows = []; ts = store.iso(now)
    def add(book, mk, player, side, line, odds): rows.append((ts, ev["game_key"], book, mk, player, side, float(line), float(odds)))
    for book, sk, vig in [("pinnacle", 0.0, .03), ("bookA", .01, .05), ("bookB", .015, .05)]:
        for t in (home, away):
            for p in world[t]["players"][:4]:
                for mk, f in [("player_points", lambda p: p["pts"]), ("player_rebounds", lambda p: p["reb"]), ("player_assists", lambda p: p["ast"]),
                              ("player_points_rebounds_assists", lambda p: p["pts"] + p["reb"] + p["ast"])]:
                    mean = f(p) * p["min"] * (1 + move); line = np.floor(mean) + .5
                    q = float(np.clip(.5 + rng.normal(sk, .015), .3, .7))
                    if book == "bookB" and p["name"].endswith("Bravo") and mk == "player_points": q -= .07   # planted mispricing on the Over's opposite
                    add(book, mk, p["name"], "Over", line, 1 / (q * (1 + vig / 2))); add(book, mk, p["name"], "Under", line, 1 / ((1 - q) * (1 + vig / 2)))
        spread = round(-2.5 + rng.normal(0, .3) * 1) - .5
        for mk, s1, s2, gl in [("h2h", home, away, 0.0), ("spreads", home, away, -4.5), ("totals", "Over", "Under", 224.5),
                               ("h2h_h1", home, away, 0.0), ("spreads_h1", home, away, -2.5), ("totals_h1", "Over", "Under", 112.5),
                               ("h2h_q1", home, away, 0.0), ("spreads_q1", home, away, -1.5), ("totals_q1", "Over", "Under", 56.5)]:
            q = float(np.clip(.5 + rng.normal(0, .015), .4, .6))
            add(book, mk, "", s1, gl, 1 / (q * (1 + vig / 2))); add(book, mk, "", s2, gl, 1 / ((1 - q) * (1 + vig / 2)))
    return rows

def run():
    path = "demo.db"
    if os.path.exists(path): os.remove(path)
    con = store.connect(path); now = store.now_utc()
    world = make_world(); today = now.astimezone(ET).date()
    days = [(today - timedelta(days=d)).strftime("%Y-%m-%d") for d in range(60, 0, -1)]
    gid = 0; allrows = []
    for d in days:
        order = list(rng.permutation(TEAMS))
        for a, b in zip(order[::2], order[1::2]):
            gid += 1; gr = game_rows(world, f"G{gid}", d, a, b); allrows += gr
            con.executemany("INSERT INTO period_scores VALUES(?,?,?,?,?,?,?,?,?)", period_rows(gr, d))
    df = pd.DataFrame(allrows)
    con.executemany(f"INSERT INTO player_logs VALUES({','.join('?'*len(sources.LOG_COLS))})", df[sources.LOG_COLS].astype(object).where(df[sources.LOG_COLS].notna(), None).values.tolist())
    tip = now + timedelta(hours=3); home, away = "Boston Celtics", "Denver Nuggets"
    ev = dict(game_key=f"{tip.astimezone(ET).strftime('%Y-%m-%d')}:{away}@{home}", odds_event_id="demo", home=home, away=away, tipoff=store.iso(tip))
    con.execute("INSERT INTO events VALUES(?,?,?,?,?)", tuple(ev.values()))
    ts = store.iso(now - timedelta(hours=2))
    con.execute("INSERT INTO injuries VALUES(?,?,?,?,?,?)", (ts, away, pkey("Nuggets Bravo"), "Nuggets Bravo", "day-to-day", "ankle - game-time decision"))
    con.execute("INSERT INTO injuries VALUES(?,?,?,?,?,?)", (ts, home, pkey("Celtics Charlie"), "Celtics Charlie", "out", "knee"))
    for when, mv in [(now - timedelta(hours=2), 0.0), (now, 0.01)]:           # two odds snapshots -> line movement + closing line
        con.executemany("INSERT INTO odds_snapshots VALUES(?,?,?,?,?,?,?,?)", synth_odds(world, ev, when, mv)); con.commit()
        rid, n = pipeline.run_cycle(con, now=when, offline=True, log=lambda *_: None)
        print(f"cycle @ {store.iso(when)}: {n} prediction rows written")
    report.daily_report(con, "reports_demo.md", top=3)
    print("\nhash chain:", "intact" if store.verify_chain(con)[0] else "BROKEN")
    # play the game, settle after the fact, evaluate
    fr = game_rows(world, "GDEMO", ev["game_key"].split(":")[0], home, away); final = pd.DataFrame(fr)
    con.executemany("INSERT INTO period_scores VALUES(?,?,?,?,?,?,?,?,?)", period_rows(fr, ev["game_key"].split(":")[0]))
    con.executemany(f"INSERT INTO player_logs VALUES({','.join('?'*len(sources.LOG_COLS))})", final[sources.LOG_COLS].astype(object).where(final[sources.LOG_COLS].notna(), None).values.tolist()); con.commit()
    print(f"\nsettled {pipeline.settle(con, now=tip + timedelta(hours=5))} predictions\n")
    evaluate.performance(con)
