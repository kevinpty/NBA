"""Independent statistical model: team strength, rest/travel, player minutes+rates, joint simulation."""
import math
import numpy as np, pandas as pd
from scipy.stats import norm, nbinom
from . import config as C

STATS = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov"]
DISP = {"pts": 40, "reb": 20, "ast": 14, "fg3m": 15, "stl": 6, "blk": 4, "tov": 10}  # NB dispersion priors; tune via calibration
TEAM_COLS = ["pts", "poss", "reb", "ast", "stl", "blk", "tov", "fg3m"]
COORD = {"Atlanta Hawks": (33.76, -84.40), "Boston Celtics": (42.37, -71.06), "Brooklyn Nets": (40.68, -73.97),
 "Charlotte Hornets": (35.23, -80.84), "Chicago Bulls": (41.88, -87.67), "Cleveland Cavaliers": (41.50, -81.69),
 "Dallas Mavericks": (32.79, -96.81), "Denver Nuggets": (39.75, -105.01), "Detroit Pistons": (42.34, -83.06),
 "Golden State Warriors": (37.77, -122.39), "Houston Rockets": (29.75, -95.36), "Indiana Pacers": (39.76, -86.16),
 "Los Angeles Clippers": (34.04, -118.27), "Los Angeles Lakers": (34.04, -118.27), "Memphis Grizzlies": (35.14, -90.05),
 "Miami Heat": (25.78, -80.19), "Milwaukee Bucks": (43.04, -87.92), "Minnesota Timberwolves": (44.98, -93.28),
 "New Orleans Pelicans": (29.95, -90.08), "New York Knicks": (40.75, -73.99), "Oklahoma City Thunder": (35.46, -97.52),
 "Orlando Magic": (28.54, -81.38), "Philadelphia 76ers": (39.90, -75.17), "Phoenix Suns": (33.45, -112.07),
 "Portland Trail Blazers": (45.53, -122.67), "Sacramento Kings": (38.58, -121.50), "San Antonio Spurs": (29.43, -98.44),
 "Toronto Raptors": (43.64, -79.38), "Utah Jazz": (40.77, -111.90), "Washington Wizards": (38.90, -77.02)}
ALTITUDE = {"Denver Nuggets": 0.8, "Utah Jazz": 0.4}   # points of margin for the visitor

def _km(a, b):
    if a not in COORD or b not in COORD: return 0.0
    (la1, lo1), (la2, lo2) = COORD[a], COORD[b]
    p1, p2 = math.radians(la1), math.radians(la2)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))

def _wavg(x, hl):
    x = np.asarray(x, float); w = 0.5 ** (np.arange(len(x))[::-1] / hl)
    return float(np.average(x, weights=w))

def _prior_corr():
    R = np.eye(len(STATS)); ix = {s: i for i, s in enumerate(STATS)}
    for a, b, v in [("pts", "reb", .05), ("pts", "ast", .10), ("pts", "fg3m", .55), ("pts", "tov", .15), ("reb", "ast", -.05),
                    ("reb", "fg3m", -.10), ("reb", "blk", .10), ("ast", "tov", .30), ("ast", "fg3m", .10)]:
        R[ix[a], ix[b]] = R[ix[b], ix[a]] = v
    return R
PRIOR_R = _prior_corr()

def _psd(R):
    w, v = np.linalg.eigh(R); R = v @ np.diag(np.clip(w, 1e-3, None)) @ v.T
    d = np.sqrt(np.diag(R)); return R / np.outer(d, d)

class Ctx:
    """Everything the models know as of a given date (only games strictly before it)."""
    def __init__(self, logs, asof_date):
        L = logs[(logs.game_date < asof_date) & (logs["min"] > 0)].sort_values("game_date").copy()
        self.L = L
        g = L.groupby(["game_id", "team"]).agg(date=("game_date", "first"), opp=("opp", "first"), home=("home", "first"),
              mins=("min", "sum"), pts=("pts", "sum"), fga=("fga", "sum"), fta=("fta", "sum"), oreb=("oreb", "sum"),
              tov=("tov", "sum"), reb=("reb", "sum"), ast=("ast", "sum"), stl=("stl", "sum"), blk=("blk", "sum"),
              fg3m=("fg3m", "sum")).reset_index()
        g["poss"] = (g.fga + .44 * g.fta - g.oreb + g.tov) * 240 / g.mins.clip(lower=1)
        o = g[["game_id", "team"] + TEAM_COLS].rename(columns={"team": "opp", **{c: "o_" + c for c in TEAM_COLS}})
        m = g.merge(o, on=["game_id", "opp"]).sort_values("date")
        m["pace"] = (m.poss + m.o_poss) / 2
        m["ortg"] = 100 * m.pts / m.poss; m["drtg"] = 100 * m.o_pts / m.poss
        self.games = m
        self.lg = dict(pace=m.pace.mean() if len(m) else 98.0, ortg=m.ortg.mean() if len(m) else 114.0)
        self.lg_ratio = {s: (m["o_" + s].sum() / m.poss.sum()) if len(m) else 1.0 for s in STATS}
        self.teams = {}
        for t, d in m.groupby("team"):
            n = len(d); lam = n / (n + 8)
            sh = lambda v, base: lam * v + (1 - lam) * base
            allowed = {s: 1 + (n / (n + 20)) * (_wavg(d["o_" + s] / d.poss, 12) / self.lg_ratio[s] - 1) for s in STATS}
            self.teams[t] = dict(n=n, pace=sh(_wavg(d.pace, 12), self.lg["pace"]), ortg=sh(_wavg(d.ortg, 12), self.lg["ortg"]),
                                 drtg=sh(_wavg(d.drtg, 12), self.lg["ortg"]), allowed=allowed,
                                 last_date=d.date.iloc[-1], last_home=bool(d.home.iloc[-1]), last_opp=d.opp.iloc[-1],
                                 dates=list(d.date))
        self.by_player = {k: d for k, d in L.groupby("player_key")}
        self.recent = {}
        for k, d in self.by_player.items():
            r = d.tail(10)
            self.recent[k] = dict(team=d.team.iloc[-1], n=len(d), mpg=r["min"].mean(), pts=r.pts.mean(), reb=r.reb.mean(), ast=r.ast.mean(),
                                  name=d.player.iloc[-1])
        reg = L.groupby("player_key").filter(lambda d: len(d) >= 5 and d["min"].mean() >= 15)
        tot = reg["min"].sum() or 1.0
        self.lr = {s: reg[s].sum() / tot if len(reg) else 0.1 for s in STATS}

    def roster(self, team): return [k for k, r in self.recent.items() if r["team"] == team]

    def rest(self, team, game_date, venue):
        t = self.teams.get(team)
        if not t: return dict(days=3, b2b=False, three_in_four=False, km=0.0)
        days = (pd.Timestamp(game_date) - pd.Timestamp(t["last_date"])).days - 1
        recent = [d for d in t["dates"] if (pd.Timestamp(game_date) - pd.Timestamp(d)).days <= 3]
        prev_loc = team if t["last_home"] else t["last_opp"]
        return dict(days=days, b2b=days == 0, three_in_four=len(recent) >= 2 and days <= 1, km=_km(prev_loc, venue))

    def injury_points(self, team, out_keys):
        """Rough margin cost of absences: Hollinger-style game score above a replacement rate. Heuristic, tune it."""
        tot = 0.0
        for k in out_keys:
            d = self.by_player.get(k)
            if d is None or self.recent[k]["team"] != team: continue
            r = d.tail(15); m = r["min"].sum()
            if m <= 0: continue
            gs = (r.pts + .7 * r.reb + .7 * r.ast + r.stl + .7 * r.blk - r.tov - .5 * r.fga).sum() / m
            tot += max(0.0, gs - 0.30) * r["min"].mean() * 0.55
        return tot

def game_projection(ctx, home, away, venue_date):
    th, ta = ctx.teams.get(home), ctx.teams.get(away)
    if not th or not ta: return None
    pace = th["pace"] * ta["pace"] / ctx.lg["pace"]
    pts_h = pace * (th["ortg"] + ta["drtg"] - ctx.lg["ortg"]) / 100
    pts_a = pace * (ta["ortg"] + th["drtg"] - ctx.lg["ortg"]) / 100
    return dict(pace=pace, margin=pts_h - pts_a + C.HOME_COURT_PTS, total=pts_h + pts_a,
                lg_pace=ctx.lg["pace"], n_home=th["n"], n_away=ta["n"])

def p_over_normal(mu, sd, line):
    """P(X>line), P(X==line), P(X<line) for an integer-valued X approximated by a normal."""
    integer = abs(line - round(line)) < 1e-9
    hi = line + 0.5 if integer else line
    over = 1 - norm.cdf(hi, mu, sd)
    if not integer: return over, 0.0, 1 - over
    push = norm.cdf(line + .5, mu, sd) - norm.cdf(line - .5, mu, sd)
    return over, push, 1 - over - push

def project_player(ctx, pk, team, opp, env):
    d = ctx.by_player.get(pk)
    if d is None or len(d) < 5: return None
    n = len(d); mins = d["min"].values
    k = min(n, 15); wm = 0.5 ** (np.arange(k)[::-1] / 6)
    mu_m = float(np.average(mins[-k:], weights=wm))
    sd_m = max(3.0, float(math.sqrt(np.average((mins[-k:] - mu_m) ** 2, weights=wm))))
    notes = []
    w = 0.5 ** (np.arange(n)[::-1] / 10); den = float((w * mins).sum())
    rates = {s: ((w * d[s].values).sum() + 150 * ctx.lr[s]) / (den + 150) for s in STATS}
    # teammates out -> usage and minutes redistribution
    out = [p for p in env["out_keys"] if p in ctx.roster(team) and p != pk]
    if out:
        rot = [p for p in ctx.roster(team) if p not in env["out_keys"] and ctx.recent[p]["mpg"] >= 10]
        tm = lambda key: max(sum(ctx.recent[p][key] for p in rot), 1.0)
        fpts = 1 + .6 * sum(ctx.recent[p]["pts"] for p in out) / tm("pts")
        fast = 1 + .6 * sum(ctx.recent[p]["ast"] for p in out) / tm("ast")
        freb = 1 + .4 * sum(ctx.recent[p]["reb"] for p in out) / tm("reb")
        rates["pts"] *= fpts; rates["ast"] *= fast; rates["reb"] *= freb
        rates["fg3m"] *= 1 + .8 * (fpts - 1); rates["tov"] *= 1 + .5 * (fast - 1)
        out_min = sum(ctx.recent[p]["mpg"] for p in out)
        if mu_m >= 15:
            add = min(5.0, 0.35 * out_min * mu_m / max(sum(ctx.recent[p]["mpg"] for p in rot), 1.0)); mu_m += add
            notes.append(f"{len(out)} teammate(s) out ({', '.join(ctx.recent[p]['name'] for p in out[:3])}): +{add:.1f} min, usage up")
    if env["b2b"]: mu_m *= 0.97; notes.append("back-to-back: minutes trimmed 3%")
    sp = env["spread_abs"]
    if mu_m >= 24 and sp > 8:
        cut = min(4.0, 0.25 * (sp - 8)); mu_m -= cut; sd_m += 0.15 * (sp - 8)
        notes.append(f"spread {sp:.1f}: blowout risk, -{cut:.1f} min")
    oppf = ctx.teams.get(opp, {}).get("allowed", {s: 1.0 for s in STATS})
    pf = env["pace_factor"]
    notes.append(f"matchup pace {pf:.2f}x league")
    for s in STATS:
        rates[s] *= pf * oppf[s]
    big = [(s, oppf[s]) for s in ("pts", "reb", "ast") if abs(oppf[s] - 1) >= 0.04]
    for s, f in big: notes.append(f"{opp} allow {100*(f-1):+.0f}% {s}/poss vs league (n={ctx.teams[opp]['n']} games)")
    # joint structure: shrunk empirical correlation of per-minute stats
    tail = d.tail(40)
    if len(tail) >= 15:
        X = np.column_stack([(tail[s] / tail["min"]).values for s in STATS])
        with np.errstate(all="ignore"): Rs = np.nan_to_num(np.corrcoef(X.T))
        lam = len(tail) / (len(tail) + 25); R = _psd(lam * Rs + (1 - lam) * PRIOR_R)
    else: R = PRIOR_R
    h = d[d.opp == opp]
    h2h = dict(n=len(h), **{s: float(h[s].mean()) for s in ("pts", "reb", "ast")}) if len(h) else dict(n=0)
    return dict(n=n, mu_m=mu_m, sd_m=sd_m, rates=rates, R=R, notes=notes, h2h=h2h, recent_min=float(mins[-5:].mean()))

def simulate(proj, seed, n=None):
    n = n or C.SIMS; rng = np.random.default_rng(seed)
    M = np.clip(rng.normal(proj["mu_m"], proj["sd_m"], n), 0, 48)
    env = np.exp(rng.normal(0, 0.06, n))                       # shared game-state noise (pace, flow)
    u = norm.cdf(rng.multivariate_normal(np.zeros(len(STATS)), proj["R"], n))
    out = {}
    for i, s in enumerate(STATS):
        mean = np.clip(proj["rates"][s] * M * env, 1e-6, None); r = DISP[s]
        out[s] = nbinom.ppf(np.clip(u[:, i], 1e-9, 1 - 1e-9), r, r / (r + mean))
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    out["_min"] = M
    return out

def prob_over(x, line):
    return float((x > line).mean()), float((x == line).mean()), float((x < line).mean())
