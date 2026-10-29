# Sports betting analytics agent - NBA module

Research tool. It estimates probabilities and flags where the model disagrees with the market; it does not
guarantee profit. Check your local laws, and never stake money you can't afford to lose.

## Setup
    pip install -r requirements.txt
    # Copy `.env.example` to `.env` and put your your new The Odds API key in `.env`
    # Keep .env private; it is excluded from Git by .gitignore.
    python -m compileall -q nba_agent run.py && python run.py demo             # offline smoke test on synthetic data

## Run automatically
    # crontab -e   (every 10 min; the agent decides internally whether a game is due for a refresh)
    */10 * * * * cd /path/to/sports_agent && ODDS_API_KEY=... python run.py cycle >> agent.log 2>&1
    30 8 * * *   cd /path/to/sports_agent && python run.py settle >> agent.log 2>&1
Or `python run.py loop` under systemd/Docker. Poll cadence per game: none >14h out, 2h, then 30 min inside 4h, 10 min inside 1h.

## What each cycle does
1. Refresh box scores (nba_api; ESPN fallback on failure), injuries (ESPN), schedule + odds (The Odds API).
2. Build team ratings, rest/travel, per-player minutes and per-minute rates, then simulate joint stat outcomes
   (shared minutes + game-state noise + shrunk empirical correlations, so PRA is modelled jointly).
3. For every line: implied prob, no-vig prob (sharp book if present, else consensus), model prob, edge, EV, Kelly stake.
4. Decide BET or PASS (see decide.py: unresolved status, thin sample, tiny role, model-vs-market gap, edge/EV thresholds).
5. Append to `predictions` (hash-chained, refuses rows at/after tipoff). `python run.py verify` checks the chain.
6. `settle` grades results, computes closing-line value. `perf` shows calibration, Brier vs market, per-market ROI, drawdown.

## Commands
cycle | loop | settle | report | perf | verify | demo

## Halves and quarters
Moneyline, spread and total for h1, h2, q1-q4. Default is `NBA_PERIODS=h1,q1` because every extra market costs
1 Odds API credit per region per game per poll; use `NBA_PERIODS=all` if your plan can afford it. Period lines are
priced by scaling the full-game projection by each period's time share (total shares are learned from ESPN quarter
scores once 200+ team-games are stored). Period moneylines treat ties as pushes; 3-way markets are skipped.
Halves settle on regulation only (OT excluded) - confirm your book's rule.

## Not built yet (be aware)
- Segment-specific models for periods (starters' rest minutes, Q1/Q4 pace, rotation patterns) and period player props.
- Play-by-play, on/off lineups, shot zones, defensive assignments, rim protection, tracking data.
- Confirmed starters (only a minutes proxy); official NBA injury-report PDF (ESPN injuries used).
- Live fetchers (nba_api, ESPN, Odds API) were written to documented response formats but NOT tested against live
  endpoints. Expect to fix small parsing issues on first run.
- Dispersion priors, sd priors, injury-impact heuristic, and decision thresholds are untuned guesses. Do not bet real
  money until `perf` shows the model beating the market's Brier score over hundreds of settled predictions.
