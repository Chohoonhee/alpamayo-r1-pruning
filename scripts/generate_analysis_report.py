"""Generate a paper-grade analysis markdown from all experiment outputs.

Reads every JSON in scripts/logs/ that matches known patterns, builds
comparison tables, and writes ANALYSIS.md at repo root. Called at the
end of each queue phase to keep an always-up-to-date snapshot of the
findings.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parent.parent
LOGS = Path(__file__).resolve().parent / "logs"
OUT = REPO / "ANALYSIS.md"


def f(x, n=3):
    if x is None: return "—"
    try: return f"{float(x):.{n}f}"
    except Exception: return str(x)


def load_json(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def section_zero_shot_evals():
    out = ["## Zero-shot eval — nuScenes val (100 samples)\n"]
    out.append("| Backbone | Policy | Drop | L2 avg | L2 3s | Alignment |")
    out.append("|---|---|---:|---:|---:|---:|")
    rows = []
    for p in sorted(LOGS.glob("eval15_*.json")):
        if "_seed" in p.name: continue
        d = load_json(p)
        if not d: continue
        rows.append(("1.5", p.stem.replace("eval15_", ""),
                     d.get("n_dropped", 0), d.get("L2_avg"),
                     d.get("L2_3s"), d.get("alignment_match_rate")))
    for p in sorted(LOGS.glob("evalR1_*.json")):
        if "_seed" in p.name: continue
        d = load_json(p)
        if not d: continue
        if "navsim" in p.name: continue  # those go in another section
        rows.append(("R1", p.stem.replace("evalR1_", ""),
                     d.get("n_dropped", 0), d.get("L2_avg"),
                     d.get("L2_3s"), d.get("alignment_match_rate")))
    for bb, label, dn, l2, l2_3, al in rows:
        out.append(f"| {bb} | {label} | {dn} | {f(l2)} | {f(l2_3)} | {f(al)} |")
    out.append("")
    return "\n".join(out)


def section_cross_domain():
    out = ["## Cross-domain alignment matrix\n",
           "Calibration domain (rows) × Evaluation domain (cols).",
           "Higher = better.\n"]
    cells = {}
    def pull(path, key="alignment_match_rate"):
        d = load_json(path)
        return d.get(key) if d else None

    # nuScenes calibration policies
    cells[("nuS",   "nuS", "R1")] = pull(LOGS / "evalR1_greedy.json")
    cells[("nuS",   "nuS", "1.5")] = pull(LOGS / "eval15_greedy.json")
    cells[("nuS",   "NAVSIM", "R1")] = pull(LOGS / "evalR1_nuscgreedy_on_navsim.json")
    cells[("nuS",   "NAVSIM", "1.5")] = pull(LOGS / "eval15_nuscgreedy_on_navsim.json")
    # NAVSIM calibration policies
    cells[("NAVSIM","nuS", "R1")] = pull(LOGS / "evalR1_navsimgreedy_on_nuscenes.json")
    cells[("NAVSIM","nuS", "1.5")] = pull(LOGS / "eval15_navsimgreedy_on_nuscenes.json")
    cells[("NAVSIM","NAVSIM","R1")] = pull(LOGS / "evalR1_navsimgreedy_on_navsim.json")
    cells[("NAVSIM","NAVSIM","1.5")] = pull(LOGS / "eval15_navsimgreedy_on_navsim.json")
    # Baselines
    bR1_n = pull(LOGS / "evalR1_baseline.json")
    b15_n = pull(LOGS / "eval15_baseline.json")
    bR1_nv = pull(LOGS / "evalR1_baseline_on_navsim_holdout.json")
    b15_nv = pull(LOGS / "eval15_baseline_on_navsim_holdout.json")

    for bb in ["R1", "1.5"]:
        out.append(f"### {bb}\n")
        out.append("| | nuScenes val | NAVSIM holdout |")
        out.append("|---|---:|---:|")
        b_n  = bR1_n if bb == "R1" else b15_n
        b_nv = bR1_nv if bb == "R1" else b15_nv
        out.append(f"| baseline (no drop) | {f(b_n)} | {f(b_nv)} |")
        out.append(f"| greedy on nuScenes | {f(cells[('nuS','nuS',bb)])} | {f(cells[('nuS','NAVSIM',bb)])} |")
        out.append(f"| greedy on NAVSIM | {f(cells[('NAVSIM','nuS',bb)])} | {f(cells[('NAVSIM','NAVSIM',bb)])} |")
        out.append("")
    return "\n".join(out)


def section_greedy_progress():
    out = ["## Greedy progression\n"]
    for label, path in [
        ("R1 / nuScenes",   LOGS / "greedyR1.json"),
        ("R1 / NAVSIM",     LOGS / "greedyR1_navsim.json"),
        ("1.5 / nuScenes",  LOGS / "greedy15.json"),
        ("1.5 / NAVSIM",    LOGS / "greedy15_navsim.json"),
    ]:
        d = load_json(path)
        if not d: continue
        out.append(f"### {label} — final drop_set = `{d.get('final_drop_set')}`\n")
        out.append("| round | + layer | drop set | align |")
        out.append("|---:|---:|---|---:|")
        for h in d.get("history", []):
            if h["round"] == 0: continue
            out.append(f"| {h['round']} | {h['best_layer']} | "
                       f"`{h['drop_set']}` | {f(h['best_align'])} |")
        out.append("")
    return "\n".join(out)


def section_random_baseline():
    rows = []
    for p in sorted(LOGS.glob("random_baseline_*.json")):
        if ".eval." in p.name or ".meta." in p.name: continue
        d = load_json(p)
        if not d: continue
        rows.append((d.get("backbone"), d.get("n_drop"), d.get("n_seeds"),
                     d.get("L2_avg_mean"), d.get("L2_avg_std"),
                     d.get("alignment_mean"), d.get("alignment_std")))
    if not rows: return ""
    out = ["## Random baseline (control)\n"]
    out.append("| Backbone | Drop | Seeds | L2 avg (mean±std) | Alignment (mean±std) |")
    out.append("|---|---:|---:|---:|---:|")
    for bb, dn, ns, l2m, l2s, am, as_ in rows:
        out.append(f"| {bb} | {dn} | {ns} | {f(l2m)}±{f(l2s)} | "
                   f"{f(am)}±{f(as_)} |")
    out.append("")
    return "\n".join(out)


def section_sample_efficiency():
    out = []
    glob_patterns = ["greedyR1_navsim_N*.json", "greedy15_navsim_N*.json"]
    rows = []
    for pat in glob_patterns:
        for p in sorted(LOGS.glob(pat)):
            if "_meta" in p.name: continue
            d = load_json(p)
            if not d: continue
            n = d.get("n_calibration", 0)
            rows.append((d.get("backbone"), n,
                         d.get("baseline_align"),
                         d.get("history", [{}])[-1].get("best_align")))
    if not rows: return ""
    out.append("## Sample-efficiency curve\n")
    out.append("| Backbone | N (calibration) | baseline align | final align |")
    out.append("|---|---:|---:|---:|")
    for bb, n, bl, fl in rows:
        out.append(f"| {bb} | {n} | {f(bl)} | {f(fl)} |")
    out.append("")
    return "\n".join(out)


def render():
    md = []
    md.append("# Experiment analysis — alignment-grounded VLM pruning")
    md.append("")
    md.append("Auto-generated by `scripts/generate_analysis_report.py`. Pulls")
    md.append("the latest JSON outputs in `scripts/logs/` and renders a")
    md.append("paper-grade comparison so the user can read the findings without")
    md.append("re-running anything. Companion to `STATUS.md` (raw event log)")
    md.append("and `PAPER_DIRECTIONS.md` (interpretation & next steps).")
    md.append("")
    md.append("---")
    md.append("")
    md.append(section_zero_shot_evals())
    md.append("---\n")
    md.append(section_cross_domain())
    md.append("---\n")
    md.append(section_greedy_progress())
    md.append("---\n")
    md.append(section_random_baseline())
    md.append("---\n")
    md.append(section_sample_efficiency())
    md.append("---\n")
    md.append("## Key findings (auto-summary)\n")
    # Pull headline numbers if present
    r1_g = load_json(LOGS / "evalR1_greedy.json")
    if r1_g:
        md.append(f"- **R1 greedy on nuScenes**: L2 {f(r1_g.get('L2_avg'))} m, "
                  f"alignment {f(r1_g.get('alignment_match_rate'))} "
                  f"(at {r1_g.get('n_dropped')}-layer drop).")
    r1_b = load_json(LOGS / "evalR1_baseline.json")
    if r1_b:
        md.append(f"- R1 baseline: L2 {f(r1_b.get('L2_avg'))} m, "
                  f"alignment {f(r1_b.get('alignment_match_rate'))}.")
    md.append("")
    md.append("See `PAPER_DIRECTIONS.md` for the narrative around these numbers.")
    return "\n".join(md)


def main():
    md = render()
    OUT.write_text(md)
    print(f"Wrote {OUT} ({len(md)} bytes)")


if __name__ == "__main__":
    main()
