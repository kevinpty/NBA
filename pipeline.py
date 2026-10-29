import json, zlib
import numpy as np, pandas as pd
from scipy.stats import norm
from . import config as C, store, sources
from .market import devig_power, ev as ev_fn
from .models import (Ctx, STATS, game_projection, p_over_normal, project_player, simulate, prob_over)
from .decide import decide

def load_logs(con): return pd.read_sql("SELECT * FROM player_logs", con)

def latest_batch(con, gk):
    ts = con.execute("SELECT MAX(fetched_at) FROM odds_snapshots WHERE game_key=?", (gk,)).fetchone()[0]
    if not ts: return None, None
    return pd.read_sql("SELECT * FROM odds_snapshots WHERE game_key=? AND fetched_at=?", con, params=(gk, ts)), ts

def consensus(piv, s1, s2):
    """No-vig P(s1) per book (power method); sharp book if present, else mean. Returns (p, n_books)."""
    if s1 not in piv.columns or s2 not in piv.columns: return None, 0
    both = piv.dropna(subset=[s1, s2])
    if both.empty: return None, 0
    fair = {b: devig_power([1 / r[s1], 1 / r[s2]])[0] for b, r in both.iterrows()}
    sharp = [fair[b] for b in C.SHARP_BOOKS if b in fair]
    return (sharp[0] if sharp else float(np.mean(list(fair.values())))), len(fair)

def maybe_insert(con, row):
    last = con.execute("SELECT decision, odds, model_p FROM predictions WHERE game_key=? AND market=? AND player=? AND side=? AND line=? "
                       "ORDER BY id DESC LIMIT 1", (row["game_key"], row["market"], row["player"], row["side"], row["line"])).fetchone()
    if last and last[0] == row["decision"] and last[1] == row["odds"] and abs(last[2] - row["model_p"]) < 0.005:
        return False
    store.insert_prediction(con, row); return True

def period_fracs(con):
    f = dict(C.TOTAL_FRAC)
    df = pd.read_sql("SELECT q1,q2,q3,q4 FROM period_scores", con)
    if len(df) >= 200:
        q = df.sum() / df.sum().sum()
        for i in range(1, 5): f[f"q{i}"] = float(q[f"q{i}"])
        f["h1"], f["h2"] = f["q1"] + f["q2"], f["q3"] + f["q4"]
    return f, dict(C.MARGIN_FRAC)

def predict_game(con, ev, ctx, inj, now, run_id):
    odds, ts = latest_batch(con, ev["game_key"])
    stats = dict(rows=0, skipped_no_fair=0, skipped_unmatched=0)
    if odds is None: return stats
    home, away, tip = ev["home"], ev["away"], store.parse(ev["tipoff"])
    if now >= tip: return stats
    date_et = tip.astimezone(sources.ET).strftime("%Y-%m-%d")
    gp = game_projection(ctx, home, away, date_et)
    if gp is None: return stats
    ih, ia = inj.get(home, {}), inj.get(away, {})
    outs = {home: {k for k, (s, _) in ih.items() if s in ("out", "doubtful")}, away: {k for k, (s, _) in ia.items() if s in ("out", "doubtful")}}
    rh, ra = ctx.rest(home, date_et, home), ctx.rest(away, date_et, home)
    from .models import ALTITUDE
    margin = gp["margin"] + ctx.injury_points(away, outs[away]) - ctx.injury_points(home, outs[home])
    margin += (-1.0 if rh["b2b"] else 0) + (1.0 if ra["b2b"] else 0) - (.4 if rh["three_in_four"] else 0) + (.4 if ra["three_in_four"] else 0)
    margin += (.3 if ra["km"] > 2500 else 0) - (.3 if rh["km"] > 2500 else 0) + ALTITUDE.get(home, 0)
    total = gp["total"] - 0.3 * (ctx.injury_points(home, outs[home]) + ctx.injury_points(away, outs[away]))
    n_team = min(gp["n_home"], gp["n_away"])

    # game-script anchor: market spread whose no-vig prob is closest to 50/50
    best, sp_abs = 9, abs(margin)
    sp = odds[odds.market == "spreads"]
    for gl, g in sp.groupby("line"):
        p, nb = consensus(g.pivot_table(index="book", columns="side", values="odds", aggfunc="max"), home, away)
        if p is not None and abs(p - .5) < best: best, sp_abs = abs(p - .5), abs(gl)

    first = pd.read_sql("SELECT * FROM odds_snapshots WHERE game_key=? AND fetched_at=(SELECT MIN(fetched_at) FROM odds_snapshots WHERE game_key=?)",
                        con, params=(ev["game_key"], ev["game_key"]))
    med = lambda df: df[df.side.isin(["Over", home])].groupby(["market", "player"]).line.median()
    open_med, cur_med = med(first), med(odds)

    tfrac, mfrac = period_fracs(con)
    proj_cache = {}
    def player_sim(pname):
        pk = sources.pkey(pname)
        if pk in proj_cache: return proj_cache[pk]
        res = None; rec = ctx.recent.get(pk)
        if rec and rec["team"] in (home, away):
            team = rec["team"]; opp = away if team == home else home
            r_t = rh if team == home else ra
            env = dict(pace_factor=gp["pace"] / gp["lg_pace"], b2b=r_t["b2b"], spread_abs=sp_abs, out_keys=outs[team] - {pk})
            pr = project_player(ctx, pk, team, opp, env)
            if pr: res = (pr, simulate(pr, zlib.crc32(f"{ev['game_key']}|{pk}".encode())), team, opp)
        proj_cache[pk] = res; return res

    for (mk, player, gl), g in odds.groupby(["market", "player", "line"]):
        is_prop = mk in C.PROP_MARKETS
        base, per = C.split_market(mk)
        if not is_prop and base is None: continue
        s1, s2 = ("Over", "Under") if (is_prop or base == "totals") else (home, away)
        piv = g.pivot_table(index="book", columns="side", values="odds", aggfunc="max")
        fair1, nb = consensus(piv, s1, s2)
        if fair1 is None or nb < 2: stats["skipped_no_fair"] += 1; continue
        proj = pmin = None; status = "ok"; n_games = n_team; tm_flag = False; notes = []; risks = []; h2h = {}
        if is_prop:
            ps = player_sim(player)
            if ps is None: stats["skipped_unmatched"] += 1; continue
            pr, sim, team, opp = ps; pk = sources.pkey(player)
            st, detail = (ih if team == home else ia).get(pk, ("ok", ""))
            if st == "out": continue
            if st in ("questionable", "doubtful"): status = "unresolved"; risks.append(f"status {st}: {detail[:80]}")
            tmi = (ih if team == home else ia)
            un = [ctx.recent[k]["name"] for k, (s, _) in tmi.items() if s in ("questionable", "doubtful") and k != pk and ctx.recent.get(k, {}).get("mpg", 0) >= 25]
            tm_flag = bool(un)
            if un: risks.append("unresolved rotation player(s): " + ", ".join(un[:3]))
            x = sim[C.PROP_MARKETS[mk]]; po, pp, pu = prob_over(x, gl)
            n_games, notes, h2h = pr["n"], pr["notes"], pr["h2h"]
            proj, pmin = float(x.mean()), pr["mu_m"]
            risks.append(f"minutes uncertainty: sd {pr['sd_m']:.1f} around {pr['mu_m']:.0f}")
            if sp_abs > 10: risks.append(f"blowout risk (spread {sp_abs:.1f})")
            card_extra = dict(proj=proj, p10=float(np.percentile(x, 10)), p90=float(np.percentile(x, 90)))
            probs = {"Over": (po, pp), "Under": (pu, pp)}
        else:
            ft, fm = (tfrac[per], mfrac[per]) if per else (1.0, 1.0)
            mu_mg, sd_mg, mu_tot, sd_tot = margin * fm, C.MARGIN_SD * ft ** .5, total * ft, C.TOTAL_SD * ft ** .5
            if base == "h2h":
                if per: o, p, u = p_over_normal(mu_mg, sd_mg, 0.0); probs = {home: (o, p), away: (u, p)}   # ties push
                else: ph = 1 - norm.cdf(0, mu_mg, sd_mg); probs = {home: (ph, 0.0), away: (1 - ph, 0.0)}
                proj = mu_mg
            elif base == "spreads":
                o, p, u = p_over_normal(mu_mg, sd_mg, -gl); probs = {home: (o, p), away: (u, p)}; proj = mu_mg
            else:
                o, p, u = p_over_normal(mu_tot, sd_tot, gl); probs = {"Over": (o, p), "Under": (u, p)}; proj = mu_tot
            notes = [f"proj margin {margin:+.1f}, total {total:.1f}, pace {gp['pace']:.1f} vs lg {gp['lg_pace']:.1f}",
                     f"rest: {home} {rh['days']}d{' B2B' if rh['b2b'] else ''}, {away} {ra['days']}d{' B2B' if ra['b2b'] else ''}, {away} travel {ra['km']:.0f} km"]
            if per: notes.append(f"{per.upper()}: full-game projection scaled by time share (total x{ft:.3f}, margin x{fm:.3f})")
            risks.append(f"model margin sd {C.MARGIN_SD}, total sd {C.TOTAL_SD} are priors; injury impact is a rough heuristic")
            if per: risks.append("no segment-specific rotation/pace model yet: period lines are scaled from the full game")
            card_extra = dict(proj=proj)
        om, cm = open_med.get((mk, player)), cur_med.get((mk, player))
        for side, (mp, push) in probs.items():
            if side not in piv.columns: continue
            bo, bb = float(piv[side].max()), piv[side].idxmax()
            fair = fair1 if side == s1 else 1 - fair1
            p_use = fair + C.SHRINK_TO_MARKET * (mp - fair)
            dec, why, stake = decide("prop" if is_prop else "period" if per else "game", mp, p_use, push, bo, fair, n_games, status, pmin, tm_flag)
            card = dict(no_vig=fair, model=mp, p_use=p_use, implied=1 / bo, edge_vs_implied=mp - 1 / bo, edge_vs_fair=mp - fair,
                        ev=ev_fn(mp, push, bo), ev_shrunk=ev_fn(p_use, push, bo), proj_min=pmin, matchup=notes, risks=risks + (why if dec == "PASS" else []),
                        n_games=n_games, h2h=h2h, books=nb, line_open=om, line_now=cm, **card_extra)
            row = dict(run_id=run_id, created_at=store.iso(now), tipoff=ev["tipoff"], game_key=ev["game_key"], market=mk, player=player,
                       side=side, line=float(gl), book=bb, odds=bo, implied=1 / bo, fair=fair, model_p=mp, p_use=p_use, push_p=push,
                       ev=card["ev"], edge_pp=card["edge_vs_implied"], proj=proj, proj_min=pmin, decision=dec, stake_frac=stake,
                       reasons=json.dumps(why), card=json.dumps(card, default=float))
            if maybe_insert(con, row): stats["rows"] += 1
    con.commit(); return stats

def run_cycle(con, now=None, offline=False, log=print):
    now = now or store.now_utc()
    if not offline:
        for name, fn in [("logs", lambda: sources.update_logs(con, now)), ("injuries", lambda: sources.update_injuries(con, now)),
                         ("events", lambda: f"{sources.refresh_events(con)} events"),
                         ("periods", lambda: sources.update_period_scores(con, now))]:
            try: log(f"[{name}] {fn()}")
            except Exception as e: log(f"[{name}] FAILED ({type(e).__name__}: {e}) - continuing with stored data")
    evs = [dict(r) for r in con.execute("SELECT * FROM events WHERE tipoff > ? ORDER BY tipoff", (store.iso(now),))]
    if not offline:
        for ev in evs:
            hrs = (store.parse(ev["tipoff"]) - now).total_seconds() / 3600
            gap = sources.poll_interval_minutes(hrs)
            last = con.execute("SELECT MAX(fetched_at) FROM odds_snapshots WHERE game_key=?", (ev["game_key"],)).fetchone()[0]
            if gap and (not last or (now - store.parse(last)).total_seconds() >= gap * 60 - 30):
                try: log(f"[odds] {ev['game_key']}: {sources.poll_odds(con, ev, now)} lines")
                except Exception as e: log(f"[odds] {ev['game_key']} FAILED ({type(e).__name__})")
    logs, inj, run_id, ctxs = load_logs(con), sources.latest_injuries(con), now.strftime("%Y%m%dT%H%M%S"), {}
    total = 0
    for ev in evs:
        if (store.parse(ev["tipoff"]) - now).total_seconds() > 30 * 3600: continue
        d = store.parse(ev["tipoff"]).astimezone(sources.ET).strftime("%Y-%m-%d")
        ctx = ctxs.setdefault(d, Ctx(logs, d))
        s = predict_game(con, ev, ctx, inj, now, run_id)
        log(f"[predict] {ev['game_key']}: +{s['rows']} rows (no-fair skipped {s['skipped_no_fair']}, unmatched players {s['skipped_unmatched']})")
        total += s["rows"]
    return run_id, total

# ---------------- settlement + CLV ----------------
def _closing_fair(con, r):
    ts = con.execute("SELECT MAX(fetched_at) FROM odds_snapshots WHERE game_key=? AND fetched_at<?", (r["game_key"], r["tipoff"])).fetchone()[0]
    if not ts: return None
    g = pd.read_sql("SELECT * FROM odds_snapshots WHERE game_key=? AND fetched_at=? AND market=? AND player=? AND line=?",
                    con, params=(r["game_key"], ts, r["market"], r["player"], r["line"]))
    if g.empty: return None
    ev = con.execute("SELECT home, away FROM events WHERE game_key=?", (r["game_key"],)).fetchone()
    s1, s2 = ("Over", "Under") if r["side"] in ("Over", "Under") else (ev["home"], ev["away"])
    p, nb = consensus(g.pivot_table(index="book", columns="side", values="odds", aggfunc="max"), s1, s2)
    if p is None: return None
    return p if r["side"] == s1 else 1 - p

def period_pts(con, date, team, per):
    r = con.execute("SELECT * FROM period_scores WHERE game_date=? AND team=?", (date, team)).fetchone()
    if not r: return None
    return {"q1": r["q1"], "q2": r["q2"], "q3": r["q3"], "q4": r["q4"], "h1": r["q1"] + r["q2"], "h2": r["q3"] + r["q4"]}[per]   # excludes OT

def settle(con, now=None):
    now = now or store.now_utc(); n = 0
    rows = con.execute("SELECT p.* FROM predictions p LEFT JOIN results r ON r.pred_id=p.id WHERE r.pred_id IS NULL AND p.tipoff<?",
                       (store.iso(now - pd.Timedelta(hours=4)),)).fetchall()
    for r in rows:
        date, mt = r["game_key"].split(":", 1); away, home = mt.split("@")
        actual = outcome = None; base, per = C.split_market(r["market"])
        if r["market"] in C.PROP_MARKETS:
            q = con.execute("SELECT * FROM player_logs WHERE game_date=? AND player_key=?", (date, sources.pkey(r["player"]))).fetchone()
            if q is None: outcome = "void"                         # DNP: books refund
            else:
                st = C.PROP_MARKETS[r["market"]]
                actual = q["pts"] + q["reb"] + q["ast"] if st == "pra" else q[st]
        else:
            if per:
                hp, ap = period_pts(con, date, home, per), period_pts(con, date, away, per)
                if hp is None or ap is None: continue              # period scores not in yet; retry next run
                mg, tot = hp - ap, hp + ap
            else:
                gid = con.execute("SELECT game_id FROM player_logs WHERE game_date=? AND team=?", (date, home)).fetchone()
                if gid is None: continue
                pts = dict(con.execute("SELECT team, SUM(pts) FROM player_logs WHERE game_id=? GROUP BY team", (gid[0],)).fetchall())
                if home not in pts or away not in pts: continue
                mg, tot = pts[home] - pts[away], pts[home] + pts[away]
        if outcome is None:
            if r["market"] in C.PROP_MARKETS or base == "totals":
                a = actual if actual is not None else tot; actual = a
                outcome = "push" if a == r["line"] else "win" if ((a > r["line"]) == (r["side"] == "Over")) else "loss"
            elif base == "h2h":
                actual = mg
                outcome = "push" if mg == 0 else "win" if ((mg > 0) == (r["side"] == home)) else "loss"
            else:
                actual = mg; v = mg + r["line"]
                outcome = "push" if v == 0 else "win" if ((v > 0) == (r["side"] == home)) else "loss"
        cf = _closing_fair(con, r) if r["decision"] == "BET" else None
        clv = None if cf is None else cf - r["implied"]
        con.execute("INSERT OR REPLACE INTO results VALUES(?,?,?,?,?)", (r["id"], store.iso(now), actual, outcome, clv)); n += 1
    con.commit(); return n
