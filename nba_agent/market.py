"""Implied probability, vig removal, EV, Kelly."""

def implied(odds): return 1.0 / odds

def devig_power(probs):
    """Find k with sum(p_i^k)=1. Shrinks longshots more than plain normalization."""
    lo, hi = 0.05, 30.0
    for _ in range(200):
        k = (lo + hi) / 2
        if sum(p ** k for p in probs) > 1: lo = k
        else: hi = k
    return [p ** k for p in probs]

def ev(p_win, p_push, odds):
    """Expected return per 1 staked; a push refunds the stake."""
    return p_win * (odds - 1) - (1 - p_win - p_push)

def kelly_fraction(p_win, p_push, odds, frac, cap):
    b = odds - 1
    if b <= 0: return 0.0
    f = ev(p_win, p_push, odds) / b
    return min(max(0.0, f) * frac, cap)
