import json
import pandas as pd
from . import store

def card_text(r):
    c = json.loads(r["card"]); fnum = lambda x, f: "n/a" if x is None else format(x, f)
    side = r["side"] if r["side"] in ("Over", "Under") else r["side"]
    line = r["line"] if not r["market"].startswith("spreads") or r["side"] in ("Over", "Under") else r["line"]
    who = f"{r['player']} " if r["player"] else ""
    L = [f"{r['game_key']} | {r['market']} | {who}{side} {line:+g}" if r["market"].startswith("spreads") else f"{r['game_key']} | {r['market']} | {who}{side} {line:g}",
         f"  Best price        {r['odds']:.2f} @ {r['book']}  (implied {c['implied']*100:.2f}%)",
         f"  No-vig market     {c['no_vig']*100:.1f}%   ({c['books']} books)",
         f"  Model probability {c['model']*100:.1f}%   (market-shrunk {c['p_use']*100:.1f}%)",
         f"  Probability edge  {c['edge_vs_implied']*100:+.2f} pp vs implied, {c['edge_vs_fair']*100:+.2f} pp vs no-vig",
         f"  EV per $100       {c['ev']*100:+.2f} (market-shrunk {c['ev_shrunk']*100:+.2f})",
         f"  Projection        {fnum(c.get('proj'), '.1f')}" + (f"  (10-90%: {c['p10']:.0f}-{c['p90']:.0f})" if "p10" in c else "")
         + (f"   minutes {c['proj_min']:.1f}" if c.get("proj_min") else ""),
         "  Matchup           " + "; ".join(c["matchup"])]
    h = c.get("h2h") or {}
    if h.get("n"): L.append(f"  Head-to-head      n={h['n']} games vs this opponent (pts {h['pts']:.1f}, reb {h['reb']:.1f}, ast {h['ast']:.1f})"
                            + (" - tiny sample, NOT used in projection" if h["n"] < 5 else " - displayed only, not used in projection"))
    mv = "" if c.get("line_open") is None or c.get("line_now") is None else f" | line open {c['line_open']:g} -> now {c['line_now']:g}"
    L += ["  Main risk         " + "; ".join(c["risks"]), f"  Sample            {c['n_games']} games{mv}",
          f"  Decision          {r['decision']}" + (f"  stake {r['stake_frac']*100:.2f}% of bankroll" if r["decision"] == "BET" else "")]
    return "\n".join(L)

def latest_run(con, run_id=None):
    run_id = run_id or con.execute("SELECT run_id FROM predictions ORDER BY id DESC LIMIT 1").fetchone()
    run_id = run_id[0] if run_id and not isinstance(run_id, str) else run_id
    return pd.read_sql("SELECT * FROM predictions WHERE run_id=? ORDER BY ev DESC", con, params=(run_id,)) if run_id else pd.DataFrame()

def daily_report(con, path=None, top=12, log=print):
    """Current best view: the newest pre-tipoff row per bet key, for games that have not started."""
    now = store.iso(store.now_utc())
    df = pd.read_sql("SELECT * FROM predictions WHERE tipoff>? ORDER BY id", con, params=(now,))
    if df.empty: log("No open predictions."); return ""
    df = df.groupby(["game_key", "market", "player", "side", "line"], as_index=False).tail(1)
    bets = df[df.decision == "BET"].sort_values("ev", ascending=False)
    out = [f"# NBA agent report - {len(bets)} bet candidates, {int((df.decision=='PASS').sum())} passes\n"]
    for _, r in bets.head(top).iterrows(): out.append("```\n" + card_text(r) + "\n```")
    if len(bets) == 0: out.append("No bets: nothing cleared the edge, data-quality and lineup-certainty rules. That is a valid outcome.")
    def cat(x):
        r = json.loads(x)[0]
        for k, v in [("status unresolved", "player status unresolved"), ("only ", "too little data"), ("projected minutes", "minutes/role uncertain"),
                     ("model differs", "model vs market gap too large (likely missing info)"), ("edge ", "edge below threshold"), ("EV ", "EV below threshold")]:
            if k in r: return v
        return "other"
    reasons = pd.Series([cat(x) for x in df[df.decision == "PASS"].reasons]).value_counts().head(6)
    out.append("\nWhy it passed:\n" + "\n".join(f"- {k}: {v}" for k, v in reasons.items()))
    text = "\n".join(out)
    if path: open(path, "w").write(text)
    log(text); return text
