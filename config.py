import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # Read local .env; environment variables take precedence.
except ImportError:
    pass

DB_PATH = os.getenv("NBA_DB", "nba_agent.db")
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
REGIONS = "us,eu"                       # eu brings Pinnacle (sharp anchor)
GAME_MARKETS = ["h2h", "spreads", "totals"]
PROP_MARKETS = {                        # Odds API market key -> stat
    "player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
    "player_threes": "fg3m", "player_steals": "stl", "player_blocks": "blk",
    "player_turnovers": "tov", "player_points_rebounds_assists": "pra",
}
SHARP_BOOKS = ("pinnacle",)

# --- decision rules (all tunable; validate with the backtest/calibration report) ---
MIN_EDGE_PP = {"prop": 0.04, "game": 0.03, "period": 0.04}   # raw model prob minus raw implied prob
MIN_EV = 0.03                                # EV after shrinking model toward market
MIN_GAMES = 12                               # recent games needed for a player projection
MAX_MODEL_MARKET_GAP = 0.15                  # bigger gap = probably missing information -> pass
SHRINK_TO_MARKET = 0.5                       # p_use = fair + 0.5*(model - fair)
KELLY_FRACTION, MAX_STAKE = 0.25, 0.02       # fraction of bankroll

# --- model priors ---
SIMS = 20000
MARGIN_SD, TOTAL_SD = 12.0, 18.5
HOME_COURT_PTS = 2.2

# --- halves / quarters (each period market costs 1 Odds API credit per region per game per poll) ---
_p = os.getenv("NBA_PERIODS", "h1,q1")           # e.g. "h1,h2,q1,q2,q3,q4" or "all"
PERIODS = ["h1", "h2", "q1", "q2", "q3", "q4"] if _p == "all" else [x for x in _p.split(",") if x]
PERIOD_MARKETS = [f"{b}_{p}" for p in PERIODS for b in ("h2h", "spreads", "totals")]
TOTAL_FRAC = {"h1": .5, "h2": .5, "q1": .25, "q2": .25, "q3": .25, "q4": .25}    # priors; learned from period_scores once n>=200
MARGIN_FRAC = {"h1": .5, "h2": .5, "q1": .25, "q2": .25, "q3": .25, "q4": .25}

def split_market(mk):
    """'spreads_h1' -> ('spreads','h1'); 'totals' -> ('totals', None); props -> (None, None)."""
    for b in ("h2h", "spreads", "totals"):
        if mk == b: return b, None
        if mk.startswith(b + "_") and mk[len(b) + 1:] in ("h1", "h2", "q1", "q2", "q3", "q4"): return b, mk[len(b) + 1:]
    return None, None
