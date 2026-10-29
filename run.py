"""CLI: python run.py [cycle|loop|settle|report|perf|verify|demo]"""
import os, sys, time
from nba_agent import config as C, store, pipeline, report, evaluate

def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "cycle"
    if cmd == "demo":
        from nba_agent import demo; return demo.run()
    con = store.connect(C.DB_PATH)
    if cmd == "cycle":
        rid, n = pipeline.run_cycle(con); print(f"run {rid}: {n} new/changed prediction rows")
        os.makedirs("reports", exist_ok=True)
        report.daily_report(con, f"reports/{rid[:8]}.md")
    elif cmd == "loop":                     # keeps itself running; cadence per game is handled inside run_cycle
        while True:
            try: pipeline.run_cycle(con); pipeline.settle(con)
            except Exception as e: print("cycle error:", e)
            time.sleep(600)
    elif cmd == "settle":
        from nba_agent import sources
        print(sources.update_logs(con, store.now_utc(), force=True)); print(sources.update_period_scores(con, store.now_utc())); print(f"settled {pipeline.settle(con)} predictions")
    elif cmd == "report": report.daily_report(con)
    elif cmd == "perf": evaluate.performance(con)
    elif cmd == "verify": print("hash chain intact" if store.verify_chain(con)[0] else f"TAMPERING at row {store.verify_chain(con)[1]}")

if __name__ == "__main__": main()
