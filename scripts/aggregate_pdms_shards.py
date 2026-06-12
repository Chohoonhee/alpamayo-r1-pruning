"""Aggregate per-shard NAVSIM PDMS CSVs into a single result.

Each shard CSV has N scene rows + a final 'average_all_frames' row.
We concatenate all scene rows and recompute means for the standard PDMS columns.
"""
import argparse
import csv
import sys


METRIC_COLS = [
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "traffic_light_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "lane_keeping",
    "history_comfort",
    "two_frame_extended_comfort",
    "score",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = []
    for p in args.csvs:
        with open(p) as f:
            rows = list(csv.DictReader(f))
        # drop the average row from each shard
        scene_rows = [r for r in rows if r.get("token") != "average_all_frames"]
        all_rows.extend(scene_rows)
    n = len(all_rows)
    print(f"aggregated {n} scenes from {len(args.csvs)} shards")
    if n == 0:
        print("[err] no rows"); sys.exit(1)

    means = {}
    for col in METRIC_COLS:
        vals = []
        for r in all_rows:
            v = r.get(col, "")
            if v in ("", None): continue
            try: vals.append(float(v))
            except ValueError: continue
        if vals:
            means[col] = sum(vals) / len(vals)
        else:
            means[col] = float("nan")

    # write aggregated CSV
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "token", "valid"] + METRIC_COLS)
        for i, r in enumerate(all_rows):
            row = [i, r.get("token",""), r.get("valid","")] + [r.get(c,"") for c in METRIC_COLS]
            writer.writerow(row)
        avg_row = [n, "average_all_frames", "True"] + [means[c] for c in METRIC_COLS]
        writer.writerow(avg_row)
    print(f"\n=== aggregated full navtest (n={n}) ===")
    print(f"PDMS:                 {means['score']:.4f}")
    print(f"NCC (collision-free): {means['no_at_fault_collisions']:.3f}")
    print(f"DAC (drivable area):  {means['drivable_area_compliance']:.3f}")
    print(f"Direction:            {means['driving_direction_compliance']:.3f}")
    print(f"TLC (traffic light):  {means['traffic_light_compliance']:.3f}")
    print(f"ego_progress:         {means['ego_progress']:.3f}")
    print(f"TTC:                  {means['time_to_collision_within_bound']:.3f}")
    print(f"Lane keeping:         {means['lane_keeping']:.3f}")
    print(f"Comfort (history):    {means['history_comfort']:.3f}")
    print(f"Comfort (2-frame):    {means['two_frame_extended_comfort']:.3f}")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
