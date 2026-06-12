"""Paired statistical comparison: Greedy k=3 vs Random k=3 on the same 500 NAVSIM scenes.

For each of 500 scenes, compute:
  - PDMS_greedy(scene) − PDMS_random(scene)
  - Same for sub-metrics

Then run paired Wilcoxon signed-rank test for per-scene differences.
Report: median delta, IQR, p-value, effect size (Cohen's d), N wins/losses/ties.
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import stats


LOGS = Path("/home/irteam/ws/alpamayo_pruning_share/scripts/logs")

METRIC_COLS = [
    "score",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
]

DISPLAY = {
    "score": "PDMS",
    "no_at_fault_collisions": "NCC",
    "drivable_area_compliance": "DAC",
    "driving_direction_compliance": "Direction",
    "traffic_light_compliance": "TLC",
    "ego_progress": "ego_progress",
    "time_to_collision_within_bound": "TTC",
    "lane_keeping": "Lane",
    "history_comfort": "Comfort_hist",
    "two_frame_extended_comfort": "Comfort_2f",
}


def load(p: Path) -> dict[str, dict]:
    """Returns {token: {col: float, ...}} for scene rows only."""
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            tok = r.get("token", "")
            if tok == "average_all_frames" or not tok:
                continue
            out[tok] = {c: float(r[c]) if r.get(c) not in ("", None) else float("nan")
                        for c in METRIC_COLS}
    return out


def paired(a: list[float], b: list[float]):
    """Return paired diff stats."""
    d = np.array(a) - np.array(b)
    n = len(d)
    mean = float(d.mean())
    median = float(np.median(d))
    std = float(d.std(ddof=1))
    iqr = float(np.percentile(d, 75) - np.percentile(d, 25))
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())
    # Wilcoxon (paired, ignore ties)
    nonzero = d[d != 0]
    if len(nonzero) > 0:
        w_stat, w_p = stats.wilcoxon(nonzero, alternative="greater")
    else:
        w_stat, w_p = float("nan"), 1.0
    # Cohen's d (paired)
    cohen_d = mean / std if std > 0 else float("nan")
    return dict(n=n, mean=mean, median=median, std=std, iqr=iqr,
                wins=wins, losses=losses, ties=ties,
                w_p=float(w_p), cohen_d=float(cohen_d))


def report(name_a: str, name_b: str, a_data: dict, b_data: dict):
    common = sorted(set(a_data) & set(b_data))
    print(f"\n=== {name_a}  vs  {name_b}  (paired, n={len(common)} scenes) ===")
    print(f"{'Metric':<14} {'mean Δ':>9} {'median Δ':>9} {'wins':>6} {'losses':>7} {'ties':>5} {'cohen_d':>8} {'wilcox_p':>10}")
    print('-' * 80)
    for col in METRIC_COLS:
        a_vals = [a_data[t][col] for t in common]
        b_vals = [b_data[t][col] for t in common]
        # drop NaN pairs
        keep = [(av, bv) for av, bv in zip(a_vals, b_vals)
                if not (np.isnan(av) or np.isnan(bv))]
        if not keep: continue
        a_clean = [x for x, _ in keep]; b_clean = [x for _, x in keep]
        s = paired(a_clean, b_clean)
        sig = ""
        if s["w_p"] < 0.001: sig = "***"
        elif s["w_p"] < 0.01: sig = "**"
        elif s["w_p"] < 0.05: sig = "*"
        elif s["w_p"] < 0.1:  sig = "."
        print(f"{DISPLAY[col]:<14} {s['mean']:>+9.4f} {s['median']:>+9.4f} {s['wins']:>6} {s['losses']:>7} {s['ties']:>5} {s['cohen_d']:>+8.3f} {s['w_p']:>10.2e} {sig}")


def main():
    print("Sample500 NAVSIM PDMS — paired scene-level comparison")
    print(f"Significance: . p<0.1, * p<0.05, ** p<0.01, *** p<0.001 (Wilcoxon signed-rank, alternative='greater')")

    baseline = load(LOGS / "pdms_baseline_r1_result.csv")
    greedy_k3 = load(LOGS / "pdms_greedy_k3_nusc_result.csv")
    random_k3 = load(LOGS / "pdms_random_k3_seed0_result.csv")
    early_k4 = load(LOGS / "pdms_early_stop_k4_result.csv")
    greedy_k7 = load(LOGS / "pdms_greedy_k7_nusc_result.csv")

    # Critical comparison: greedy beats random at K=3
    report("greedy_k3_nusc", "random_k3_seed0", greedy_k3, random_k3)
    # K=3 greedy vs baseline (does method help or hurt vs no pruning?)
    report("greedy_k3_nusc", "baseline",        greedy_k3, baseline)
    # K=4 early-stop vs random k=3 (different K but interesting)
    report("early_stop_k4",  "random_k3_seed0", early_k4, random_k3)
    # K=7 greedy vs random k=3
    report("greedy_k7_nusc", "random_k3_seed0", greedy_k7, random_k3)


if __name__ == "__main__":
    main()
