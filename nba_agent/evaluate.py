"""Quality of probability estimates: calibration, sample size, drawdown, per-market, CLV."""
import numpy as np, pandas as pd

def final_rows(con):
    """Last prediction per (game, market, player, side, line) made BEFORE tipoff, joined to its result."""
    df = pd.read_sql("SELECT p.*, r.outcome, r.actual, r.clv_pp FROM predictions p LEFT JOIN results r ON r.pred_id=p.id "
                     "WHERE p.created_at < p.tipoff", con)
    if df.empty: return df
    return df.sort_values("created_at").groupby(["game_key", "market", "player", "side", "line"], as_index=False).tail(1)

def performance(con, log=print):
    df = final_rows(con)
    s = df[df.outcome.isin(["win", "loss"])].copy() if len(df) else df
    if s is None or len(s) == 0:
        log("No settled predictions yet."); return
    s["y"] = (s.outcome == "win").astype(float)
    log(f"Settled predictions: {len(s)} | model Brier {((s.model_p-s.y)**2).mean():.4f} | market Brier {((s.fair-s.y)**2).mean():.4f} "
        f"(lower is better; model must beat the market to have an edge)")
    if len(s) < 300: log(f"  sample size {len(s)} is small: treat all numbers below as noisy")
    s["band"] = pd.cut(s.model_p, np.arange(0, 1.01, .1), include_lowest=True)
    log("\nCalibration by model-probability band:")
    for b, g in s.groupby("band", observed=True):
        log(f"  {str(b):<12} n={len(g):<5} predicted {g.model_p.mean():.3f}  observed {g.y.mean():.3f}")
    log("\nBy market:")
    for m, g in s.groupby("market"):
        log(f"  {m:<32} n={len(g):<5} brier model {((g.model_p-g.y)**2).mean():.4f} vs market {((g.fair-g.y)**2).mean():.4f}")
    b = df[(df.decision == "BET") & df.outcome.isin(["win", "loss", "push"])].sort_values("tipoff").copy()
    if len(b) == 0: log("\nNo settled BET recommendations yet."); return
    b["u"] = np.where(b.outcome == "win", b.odds - 1, np.where(b.outcome == "loss", -1.0, 0.0))
    eq = b.u.cumsum(); dd = (eq.cummax() - eq).max()
    log(f"\nBET recommendations (flat 1u): n={len(b)} | profit {b.u.sum():+.2f}u | ROI {b.u.mean()*100:+.1f}% | max drawdown {dd:.2f}u")
    c = b.clv_pp.dropna()
    if len(c): log(f"Avg closing-line value: {c.mean()*100:+.2f} pp over {len(c)} bets (positive = beating the close)")
    for m, g in b.groupby("market"): log(f"  {m:<32} n={len(g):<4} ROI {g.u.mean()*100:+.1f}%")
