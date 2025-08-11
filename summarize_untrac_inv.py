#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, glob, json, argparse, datetime
import pandas as pd

def load_first_last_eval_loss(trainer_state_path: str):
    with open(trainer_state_path, "r") as f:
        state = json.load(f)
    logs = [x for x in state.get("log_history", []) if "eval_loss" in x]
    if not logs:
        return None, None
    first = logs[0]["eval_loss"]
    last  = logs[-1]["eval_loss"]
    return float(first), float(last)

def load_from_results_like(dir_path: str):
    for pattern in ("results*.json", "metrics*.json"):
        for fp in glob.glob(os.path.join(dir_path, pattern)):
            try:
                with open(fp, "r") as f:
                    d = json.load(f)
                orig = d.get("orig_eval_loss")
                unld = d.get("unlearned_eval_loss")
                if orig is not None and unld is not None:
                    return float(orig), float(unld)
            except Exception:
                pass
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Summarize UnTrac-Inv runs into CSV + Markdown.")
    ap.add_argument("--runs_dir", default="untrac_inv_outputs", help="Directory containing per-split outputs.")
    ap.add_argument("--out_csv", default="untrac_inv_summary.csv", help="CSV output path.")
    ap.add_argument("--out_md",  default="untrac_inv_report.md", help="Markdown report path.")
    args = ap.parse_args()

    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs_dir, "*"))):
        if not os.path.isdir(run_dir): 
            continue
        dataset = os.path.basename(run_dir)
        ts_path = os.path.join(run_dir, "trainer_state.json")

        orig, unld = (None, None)
        if os.path.isfile(ts_path):
            orig, unld = load_first_last_eval_loss(ts_path)
        if orig is None or unld is None:
            # fallback to results/metrics json
            orig, unld = load_from_results_like(run_dir)

        if orig is None or unld is None:
            print(f"[WARN] skip {dataset}: no eval_loss found.")
            continue

        delta = unld - orig
        rows.append({
            "dataset": dataset,
            "orig_eval_loss": round(orig, 6),
            "unlearned_eval_loss": round(unld, 6),
            "delta_eval_loss": round(delta, 6),
            "run_dir": run_dir
        })

    if not rows:
        print(f"[ERROR] No runs found under {args.runs_dir}.")
        return

    df = pd.DataFrame(rows).sort_values("delta_eval_loss", ascending=False)
    df.to_csv(args.out_csv, index=False)
    print(f"[OK] Saved CSV -> {args.out_csv}")

    # Markdown 
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append(f"# UnTrac-Inv Influence Summary  \nGenerated at {now}\n")
    lines.append(f"- Runs directory: `{args.runs_dir}`")
    lines.append(f"- Rows: **{len(df)}**\n")
    lines.append("## Ranking by Δ eval loss (descending)\n")
    lines.append("| # | dataset | orig_eval_loss | unlearned_eval_loss | Δ eval loss |")
    lines.append("|---:|---|---:|---:|---:|")
    for i, r in enumerate(df.to_dict(orient="records"), start=1):
        lines.append(f"| {i} | {r['dataset']} | {r['orig_eval_loss']} | {r['unlearned_eval_loss']} | **{r['delta_eval_loss']}** |")
    lines.append("\n**Note:** Δ eval loss = final − first (higher ⇒ more influential).")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved report -> {args.out_md}")

if __name__ == "__main__":
    main()
