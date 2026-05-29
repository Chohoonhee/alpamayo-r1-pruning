"""Rule-based 4-class maneuver classifiers for CoT and trajectory.

Vendored from `vipe_test/compute_reliable_v2.py` (the ViPE project) so the
pruning side can score CoT-Action alignment without depending on the sister
repo. Both copies must stay in sync — if you change one, update the other
or factor into a shared package.

The 4 classes match how Alpamayo's action head behaves: nudge / merge / lane
change all collapse into `left_any` or `right_any` because the action head
treats them similarly, and 6-class splits had too few samples per bin.

Validation: on the ViPE 10,591-sample inference set, this rule-based CoT
classifier agreed with a Qwen3-VL judge at 88% — close enough that the
extra cost of a VLM judge isn't worth it for layer-scoring use.
"""
from __future__ import annotations

import re
import numpy as np


# ---------- CoT rule-based 4-class ----------
_LEFT_P = [
    r'\bturn(?:ing)? left\b', r'\bnudg(?:e|ing) (?:to )?(?:the )?left\b',
    r'\bmerg(?:e|ing) (?:to )?(?:the )?left\b', r'\bveer(?:ing)? left\b',
    r'\blane change to (?:the )?left\b', r'\bchange to (?:the )?left(?: lane)?\b',
    r'\bmove (?:to )?(?:the )?left\b', r'\bshift left\b', r'\bbear left\b',
    r'\b(?:left|left-hand) curve\b', r'\bcurves? left\b', r'\bbends? left\b',
    r'\bfollow(?:ing)? the left\b',
]
_RIGHT_P = [
    r'\bturn(?:ing)? right\b', r'\bnudg(?:e|ing) (?:to )?(?:the )?right\b',
    r'\bmerg(?:e|ing) (?:to )?(?:the )?right\b', r'\bveer(?:ing)? right\b',
    r'\blane change to (?:the )?right\b', r'\bchange to (?:the )?right(?: lane)?\b',
    r'\bmove (?:to )?(?:the )?right\b', r'\bshift right\b', r'\bbear right\b',
    r'\b(?:right|right-hand) curve\b', r'\bcurves? right\b', r'\bbends? right\b',
    r'\bfollow(?:ing)? the right\b',
]
_STOP_P = [
    r'\bi (?:will|should|need to|am going to) (?:come to a )?stop\b',
    r'\bcome to a (?:full )?stop\b', r'\bfull stop\b',
    r'\bstop (?:for|at|because|due to)\b',
    r'\bstop behind\b', r'\bstop (?:to (?:yield|allow|wait))\b',
    r'\bbrake (?:hard|firmly|to stop)\b',
    r'\bhalt\b', r'\bremain(?:ing)? stopped\b',
    r'\bstay(?:ing)? stopped\b', r'\bkeep(?:ing)? stopped\b',
    r'\bstop due to\b',
]


def classify_cot_rule(cot: str) -> str:
    """4-class CoT label: 'cruise' / 'left_any' / 'right_any' / 'stop'."""
    t = str(cot).lower()
    has_left = any(re.search(p, t) for p in _LEFT_P)
    has_right = any(re.search(p, t) for p in _RIGHT_P)
    if has_left and not has_right:
        return 'left_any'
    if has_right and not has_left:
        return 'right_any'
    if any(re.search(p, t) for p in _STOP_P):
        return 'stop'
    return 'cruise'


def classify_traj_4class(traj_xy, n: int = 30) -> str:
    """4-class trajectory label over the first `n` waypoints (3s at 10Hz).

    Conventions:
      x forward (+), y left (+).
      fwd  = mean(x[:n])    > 0.5 m → moving forward (else 'stop')
      lat  = mean(y[:n])    > 0.5 m → laterally biased
      final = y[-1]          > 1.0 m → ends offset → left/right
    """
    pts = np.array(traj_xy[:n])
    if len(pts) < 3:
        return 'unknown'
    fwd = pts[:, 0].mean()
    lat_mean = pts[:, 1].mean()
    final_lat = pts[-1, 1]
    if fwd < 0.5 and abs(final_lat) < 0.5:
        return 'stop'
    if abs(lat_mean) > 0.5 or abs(final_lat) > 1.0:
        return 'left_any' if lat_mean > 0 else 'right_any'
    return 'cruise'


def alignment_match(cot_label: str, action_label: str) -> int:
    """1 if both labels agree, else 0. Treat 'unknown' as non-matching."""
    if cot_label == 'unknown' or action_label == 'unknown':
        return 0
    return int(cot_label == action_label)
