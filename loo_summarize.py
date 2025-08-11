#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import pandas as pd

def last_eval_loss(trainer_state_path: str) -> float:
    with open(trainer_state_path, "r") as f:
        state = json.load(f)
    logs = [x for x in state.get("log_history", []) if "eval_loss" in x]
    if not logs:
        raise ValueError(f"No eval_loss found in {trainer_state_path}")
    return float(logs[-1]["eval_loss"])

def main():
    ap = argparse.ArgumentParser(description="Summarize LOO Δ eval loss.")
    ap.add_argument("--root", default="model/t5_00_loo_00", help="LOO runs root dir")
    ap.add_argument("--baseline_dir", default="all", help="Subdir name for baseline (no leave-out)")
    ap.add_argument("--out", default="loo_summary.csv", help="Output CSV filename (saved under --root)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    base_path = os.path.join(root, args.baseline_dir, "trainer_state.json")
    if not os.path.isfile(base_path):
        raise SystemExit(f"Baseline not found: {base_path}\n"
                         f"Tip: run the ALL baseline first (no --loo_dataset_names).")

    base_loss = last_eval_loss(base_path)

    rows = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        if not os.path.isdir(run_dir): 
            continue
        if name == args.baseline_dir:
            continue
        ts = os.path.join(run_dir, "trainer_state.json")
        if not os.path.isfile(ts):
            print(f"[skip] {name}: no trainer_state.json")
            continue
        leave_loss = last_eval_loss(ts)
        rows.append({
            "dataset": name,
            "eval_loss_all": round(base_loss, 6),
            "eval_loss_leaveout": round(leave_loss, 6),
            "delta_eval_loss": round(leave_loss - base_loss, 6),
        })

    if not rows:
        raise SystemExit(f"No LOO runs found under {root}")

    df = pd.DataFrame(rows).sort_values("delta_eval_loss", ascending=False)
    out_csv = os.path.join(root, args.out)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved → {out_csv}")

if __name__ == "__main__":
    main()
