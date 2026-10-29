"""Bet-or-pass rules. The agent is explicitly allowed (and expected) to abstain."""
from . import config as C
from .market import ev, kelly_fraction

def decide(kind, model_p, p_use, push_p, odds, fair, n_games, status, proj_min, teammate_flag):
    edge = model_p - 1 / odds
    thr = C.MIN_EDGE_PP[kind] + (0.02 if teammate_flag else 0.0)
    if status == "unresolved":
        return "PASS", ["player status unresolved (questionable/doubtful) - wait for confirmation"], 0.0
    if n_games < C.MIN_GAMES:
        return "PASS", [f"only {n_games} games of data (need {C.MIN_GAMES})"], 0.0
    if proj_min is not None and proj_min < 12:
        return "PASS", [f"projected minutes {proj_min:.0f} - role too small/uncertain"], 0.0
    if abs(model_p - fair) > C.MAX_MODEL_MARKET_GAP:
        return "PASS", [f"model differs from no-vig market by {abs(model_p-fair)*100:.0f} pp - likely missing information"], 0.0
    if edge < thr:
        return "PASS", [f"edge {edge*100:+.1f} pp below {thr*100:.0f} pp threshold"], 0.0
    e = ev(p_use, push_p, odds)
    if e < C.MIN_EV:
        return "PASS", [f"EV {e*100:+.1f}% after shrinking toward market is below {C.MIN_EV*100:.0f}%"], 0.0
    return "BET", [f"edge {edge*100:+.1f} pp, EV {e*100:+.1f}% (market-shrunk)"], kelly_fraction(p_use, push_p, odds, C.KELLY_FRACTION, C.MAX_STAKE)
