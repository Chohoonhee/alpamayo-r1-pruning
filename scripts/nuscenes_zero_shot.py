"""
Alpamayo V1 zero-shot on nuScenes val (LAW-style L2 + collision metric).

Ported from Waymo pipeline:
- 6 surround cams → use 3 front (FRONT_LEFT, FRONT, FRONT_RIGHT) for Alpamayo
- Predict 6 waypoints at 2Hz (3s horizon) — standard nuScenes planning protocol
- GT trajectory: ego motion next 3s (6 steps)

Metrics (per LAW/UniAD/VAD convention):
- L2 @ 1s/2s/3s (average)
- Collision rate @ 1s/2s/3s (BEV box check against agents)
"""
import os, sys, json, math, glob
import numpy as np
import zmq
from PIL import Image
from pyquaternion import Quaternion

# alpamayo_client is vendored next to this file in scripts/
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from alpamayo_client import query_alpamayo
from paths import NUSC_ROOT as _NUSC_ROOT, NUSC_VERSION as _NUSC_VERSION

# Import nuScenes after download complete
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.data_classes import Box
except ImportError:
    print("nuscenes-devkit not installed yet. Install: pip install nuscenes-devkit")
    sys.exit(1)


NUSC_ROOT = str(_NUSC_ROOT)
VERSION   = _NUSC_VERSION
ALPAMAYO_PORT = 5556


# ── Metrics ─────────────────────────────────────────────────────────────────
class PlanningMetric:
    """Simplified L2 + collision metric (LAW/STP3 style)."""
    def __init__(self):
        self.W, self.H = 1.85, 4.084   # ego vehicle dims
        self.horizons = [2, 4, 6]      # steps = 1s, 2s, 3s at 2Hz

    def l2(self, pred_xy, gt_xy):
        """pred/gt: (6, 2). Return dict of {t_name: L2}."""
        l2 = np.sqrt(((pred_xy - gt_xy) ** 2).sum(-1))
        return {
            "L2_1s": float(l2[:2].mean()),
            "L2_2s": float(l2[:4].mean()),
            "L2_3s": float(l2.mean()),
        }

    def collision(self, pred_xy, agent_boxes_list):
        """
        pred_xy: (6, 2) in ego frame (+x forward, +y left)
        agent_boxes_list: list of 6 frames, each is list of (cx, cy, yaw, w, h) in ego
        Returns: {t_name: 0/1 per horizon}
        """
        cols = []
        for i, (px, py) in enumerate(pred_xy):
            col = 0
            if i < len(agent_boxes_list):
                for (cx, cy, yaw, w, h) in agent_boxes_list[i]:
                    if self._box_overlap(px, py, 0.0, self.W, self.H,
                                         cx, cy, yaw, w, h):
                        col = 1; break
            cols.append(col)
        return {
            "Col_1s": float(np.mean(cols[:2])),
            "Col_2s": float(np.mean(cols[:4])),
            "Col_3s": float(np.mean(cols)),
        }

    @staticmethod
    def _obb_corners(cx, cy, yaw, w, h):
        """Return 4 corners of OBB as (4,2) array. w=lateral, h=longitudinal."""
        c, s = math.cos(yaw), math.sin(yaw)
        # local half-extents: longitudinal (h/2 along heading) × lateral (w/2)
        dx = np.array([ h/2,  h/2, -h/2, -h/2])
        dy = np.array([ w/2, -w/2, -w/2,  w/2])
        xs = cx + c*dx - s*dy
        ys = cy + s*dx + c*dy
        return np.stack([xs, ys], axis=1)   # (4,2)

    @staticmethod
    def _sat_overlap(corners_a, corners_b):
        """Separating Axis Theorem for two convex polygons."""
        def axes(corners):
            n = len(corners)
            for i in range(n):
                edge = corners[(i+1) % n] - corners[i]
                yield np.array([-edge[1], edge[0]])

        def project(corners, axis):
            dots = corners @ axis
            return dots.min(), dots.max()

        for axis in list(axes(corners_a)) + list(axes(corners_b)):
            norm = np.dot(axis, axis)
            if norm < 1e-10:
                continue
            axis = axis / math.sqrt(norm)
            mn_a, mx_a = project(corners_a, axis)
            mn_b, mx_b = project(corners_b, axis)
            if mx_a < mn_b or mx_b < mn_a:
                return False   # separating axis found
        return True

    def _box_overlap(self, x1, y1, yaw1, w1, h1, x2, y2, yaw2, w2, h2):
        """Proper OBB-OBB overlap via Separating Axis Theorem."""
        # Fast circle reject
        r1 = math.hypot(w1, h1) / 2
        r2 = math.hypot(w2, h2) / 2
        if math.hypot(x1-x2, y1-y2) > r1 + r2:
            return False
        corners_a = self._obb_corners(x1, y1, yaw1, w1, h1)
        corners_b = self._obb_corners(x2, y2, yaw2, w2, h2)
        return self._sat_overlap(corners_a, corners_b)


# ── Data extraction ────────────────────────────────────────────────────────
def get_ego_in_world(nusc, sample_token):
    """Ego pose in world frame at sample's timestamp."""
    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.array(ep["translation"]), Quaternion(ep["rotation"])


def world_to_ego(pos_world, ego_trans, ego_rot):
    """Transform world xyz to ego frame."""
    rel = pos_world - ego_trans
    return ego_rot.inverse.rotation_matrix @ rel


def get_gt_future(nusc, sample_token, n_future=6, dt=0.5):
    """GT ego trajectory next 3s (6 steps at 2Hz) in ego frame at t0."""
    ego_t0_trans, ego_t0_rot = get_ego_in_world(nusc, sample_token)
    pts = []
    cur = nusc.get("sample", sample_token)
    for _ in range(n_future):
        if not cur["next"]:
            break
        cur = nusc.get("sample", cur["next"])
        tr, _ = get_ego_in_world(nusc, cur["token"])
        ego_xyz = world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
        pts.append([ego_xyz[0], ego_xyz[1]])
    while len(pts) < n_future:
        pts.append(pts[-1] if pts else [0, 0])
    return np.array(pts)


def get_past_history(nusc, sample_token, n_hist=16, target_hz=10, nusc_hz=2):
    """Past ego history + rotation in ego frame at t0, interpolated to target_hz.

    nuScenes samples are 2Hz; we interpolate positions and headings to 10Hz,
    then compute rotation matrices relative to t0 heading (Waymo convention).
    Returns: hist_xyz (16,3), hist_rot (16,3,3)
    """
    ego_t0_trans, ego_t0_rot = get_ego_in_world(nusc, sample_token)
    t0_yaw = ego_t0_rot.yaw_pitch_roll[0]  # absolute heading at t0

    # Collect available 2Hz samples (t=0, -0.5, -1.0, ...)
    sparse_xyz = []
    sparse_yaw = []
    sparse_t = []
    cur = nusc.get("sample", sample_token)
    dt_nusc = 1.0 / nusc_hz  # 0.5s
    n_back = int(math.ceil((n_hist / target_hz) / dt_nusc)) + 2
    for k in range(n_back):
        tr, q = get_ego_in_world(nusc, cur["token"])
        ego_xyz = world_to_ego(np.array(tr), ego_t0_trans, ego_t0_rot)
        abs_yaw = q.yaw_pitch_roll[0]
        rel_yaw = abs_yaw - t0_yaw  # heading relative to t0
        sparse_xyz.insert(0, [ego_xyz[0], ego_xyz[1], 0.0])
        sparse_yaw.insert(0, rel_yaw)
        sparse_t.insert(0, -k * dt_nusc)
        if not cur["prev"]:
            break
        cur = nusc.get("sample", cur["prev"])

    sparse_xyz = np.array(sparse_xyz, dtype=np.float64)
    sparse_yaw = np.array(sparse_yaw, dtype=np.float64)
    sparse_t = np.array(sparse_t, dtype=np.float64)

    # Interpolate to 10Hz grid
    dt_target = 1.0 / target_hz
    tgt_t = np.arange(-(n_hist - 1), 1) * dt_target  # -1.5 .. 0s

    out_xyz = np.zeros((n_hist, 3), dtype=np.float32)
    for dim in range(3):
        out_xyz[:, dim] = np.interp(tgt_t, sparse_t, sparse_xyz[:, dim],
                                    left=sparse_xyz[0, dim], right=sparse_xyz[-1, dim])
    out_xyz[-1] = [0, 0, 0]

    yaw_interp = np.interp(tgt_t, sparse_t, sparse_yaw,
                           left=sparse_yaw[0], right=sparse_yaw[-1])

    # Rotation matrices relative to t0 (same convention as Waymo build_ego_history)
    out_rot = np.zeros((n_hist, 3, 3), dtype=np.float32)
    for i in range(n_hist):
        y = float(yaw_interp[i])
        out_rot[i] = [[math.cos(y), -math.sin(y), 0],
                      [math.sin(y),  math.cos(y), 0],
                      [0, 0, 1]]
    out_rot[-1] = np.eye(3)

    return out_xyz, out_rot


def extract_front_cams(nusc, sample_token, resize=(512, 320), n_temporal=4):
    """Get 3 front cameras × n_temporal past sweeps → (12, 3, H, W).
    Uses actual past sweep data (oldest→newest order) instead of replicated frame.
    nuScenes sweeps are ~12Hz; 4 frames ≈ 0.33s history.
    """
    sample = nusc.get("sample", sample_token)
    cam_names = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    frames = []
    for cn in cam_names:
        cur_sd = nusc.get("sample_data", sample["data"][cn])
        cam_frames = []
        for _ in range(n_temporal):
            img_path = os.path.join(nusc.dataroot, cur_sd["filename"])
            arr = np.array(Image.open(img_path).resize(resize)).transpose(2, 0, 1).astype(np.uint8)
            cam_frames.insert(0, arr)   # newest at end
            if cur_sd["prev"]:
                cur_sd = nusc.get("sample_data", cur_sd["prev"])
        # Pad oldest frame if not enough history
        while len(cam_frames) < n_temporal:
            cam_frames.insert(0, cam_frames[0].copy())
        frames.extend(cam_frames)
    return np.stack(frames)   # (12, 3, H, W)


def get_agent_boxes_in_ego(nusc, sample_token, n_future=6):
    """Agent bounding boxes in ego frame for each future step.
    Boxes start from t=0.5s (first future step) to match pred_xy timing.
    """
    ego_t0_trans, ego_t0_rot = get_ego_in_world(nusc, sample_token)
    ego_t0_yaw = ego_t0_rot.yaw_pitch_roll[0]
    frames_boxes = []
    cur = nusc.get("sample", sample_token)
    # Advance to first future step so agent_boxes[i] aligns with pred_xy[i]
    if cur["next"]:
        cur = nusc.get("sample", cur["next"])
    for i in range(n_future):
        boxes = []
        for ann_tok in cur["anns"]:
            ann = nusc.get("sample_annotation", ann_tok)
            if "vehicle" not in ann["category_name"] and "human" not in ann["category_name"]:
                continue
            bp = np.array(ann["translation"])
            ego_xyz = world_to_ego(bp, ego_t0_trans, ego_t0_rot)
            w, l = ann["size"][0], ann["size"][1]
            # Yaw relative to ego frame at t0 (not world frame)
            q = Quaternion(ann["rotation"])
            yaw_world = q.yaw_pitch_roll[0]
            yaw_ego = yaw_world - ego_t0_yaw
            boxes.append((ego_xyz[0], ego_xyz[1], yaw_ego, w, l))
        frames_boxes.append(boxes)
        if not cur["next"]:
            break
        cur = nusc.get("sample", cur["next"])
    while len(frames_boxes) < n_future:
        frames_boxes.append([])
    return frames_boxes


def get_nav_text(nusc, sample_token):
    """Best-effort nav: detect turn from next 3s trajectory."""
    gt = get_gt_future(nusc, sample_token, n_future=6)
    delta_y = gt[-1, 1]
    if abs(delta_y) < 2.0:
        return "Continue straight ahead."
    return "Turn left at the next intersection." if delta_y > 0 else "Turn right at the next intersection."


# ── Main evaluation loop ───────────────────────────────────────────────────
def alpamayo_to_nuscenes_traj(waypoints_10hz_64):
    """Alpamayo 64 waypoints @10Hz → nuScenes 6 waypoints @2Hz (3s)."""
    src_t = np.arange(1, 65) * 0.1   # 0.1..6.4s
    tgt_t = np.arange(1, 7) * 0.5    # 0.5..3.0s
    x = np.interp(tgt_t, src_t, waypoints_10hz_64[:, 0])
    y = np.interp(tgt_t, src_t, waypoints_10hz_64[:, 1])
    return np.stack([x, y], axis=1)


def main(n_samples=100, port=5556, use_camera_indices=False):
    nusc = NuScenes(version=VERSION, dataroot=NUSC_ROOT, verbose=False)
    metric = PlanningMetric()

    val_splits = ["scene-0001", "scene-0002"]   # Will use scene-split
    # Use nuscenes splits
    from nuscenes.utils.splits import create_splits_scenes
    splits = create_splits_scenes()
    val_scene_names = set(splits.get("val", []))

    # Collect val samples
    samples = []
    for scene in nusc.scene:
        if scene["name"] in val_scene_names:
            sample_tok = scene["first_sample_token"]
            while sample_tok:
                samples.append(sample_tok)
                s = nusc.get("sample", sample_tok)
                sample_tok = s["next"]
    print(f"Val samples available: {len(samples)}")
    # Stride-sample for coverage across all scenes instead of just first n
    if n_samples < len(samples):
        stride = len(samples) // n_samples
        samples = samples[::stride][:n_samples]
    print(f"Using: {len(samples)}")

    results = []
    for i, tok in enumerate(samples):
        try:
            frames = extract_front_cams(nusc, tok)
            hist, hist_rot = get_past_history(nusc, tok)
            nav = get_nav_text(nusc, tok)

            # camera_indices: 3 cams × 4 temporal = [0,0,0,0, 1,1,1,1, 2,2,2,2]
            cam_idx = [0]*4 + [1]*4 + [2]*4 if use_camera_indices else None
            wps = query_alpamayo(frames, hist, hist_rot, nav, port=port,
                                 camera_indices=cam_idx)
            if wps is None:
                continue
            pred = alpamayo_to_nuscenes_traj(wps)

            gt = get_gt_future(nusc, tok)
            boxes = get_agent_boxes_in_ego(nusc, tok)

            l2 = metric.l2(pred, gt)
            col = metric.collision(pred, boxes)
            results.append({
                "sample_token": tok, "pred": pred.tolist(), "gt": gt.tolist(),
                **l2, **col,
            })

            if (i + 1) % 20 == 0:
                print(f"[{i+1}/{len(samples)}]  "
                      f"L2 1s/2s/3s mean: "
                      f"{np.mean([r['L2_1s'] for r in results]):.3f}/"
                      f"{np.mean([r['L2_2s'] for r in results]):.3f}/"
                      f"{np.mean([r['L2_3s'] for r in results]):.3f}m")
        except Exception as e:
            print(f"  [skip] {tok[:8]}: {e}")

    # Aggregate
    print(f"\n{'='*60}\nAlpamayo V1 Zero-Shot on nuScenes val ({len(results)})\n{'='*60}")
    for k in ["L2_1s","L2_2s","L2_3s","Col_1s","Col_2s","Col_3s"]:
        vals = [r[k] for r in results]
        if "L2" in k:
            print(f"  {k:8}: {np.mean(vals):.3f} m")
        else:
            print(f"  {k:8}: {100*np.mean(vals):.2f} %")
    avg_l2 = np.mean([[r["L2_1s"], r["L2_2s"], r["L2_3s"]] for r in results])
    avg_col = np.mean([[r["Col_1s"], r["Col_2s"], r["Col_3s"]] for r in results])
    print(f"\n  Avg L2:        {avg_l2:.3f} m")
    print(f"  Avg Collision: {100*avg_col:.2f} %")

    # Save
    from paths import OUTPUTS_DIR
    with open(str(OUTPUTS_DIR / "zero_shot_results.json"), "w") as f:
        json.dump(results, f)
    print(f"\nSaved results JSON.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--port", type=int, default=5556)
    parser.add_argument("--camera_indices", action="store_true")
    args = parser.parse_args()
    main(args.n_samples, args.port, args.camera_indices)
