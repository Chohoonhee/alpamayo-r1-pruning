"""Standalone Alpamayo inference client (no tensorflow dependency)."""
import json
import numpy as np
import zmq


def query_alpamayo(frames_np, hist_xyz, hist_rot, nav_text, port=5556, timeout=30000,
                   camera_indices=None, return_cot=False):
    """Send to Alpamayo infer server, return (64, 2) waypoints or None.

    camera_indices: list of int per image (e.g. [0,0,0,0,1,1,1,1,2,2,2,2]).
                    If None, server uses default ordering.
    return_cot: if True, returns (waypoints, cot_text) tuple (cot_text may be "").
    """
    ego_xyz = hist_xyz[None, None]   # (1, 1, 16, 3)
    ego_rot = hist_rot[None, None]   # (1, 1, 16, 3, 3)

    req = {
        "frames":  frames_np.tolist(),
        "ego_xyz": ego_xyz.tolist(),
        "ego_rot": ego_rot.tolist(),
        "nav":     nav_text,
    }
    if camera_indices is not None:
        req["camera_indices"] = camera_indices
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout)
    sock.connect(f"tcp://127.0.0.1:{port}")
    try:
        sock.send(json.dumps(req).encode())
        resp = json.loads(sock.recv())
        if not resp.get("ok"):
            return (None, "") if return_cot else None
        wps = np.array(resp["waypoints"])[:, :2]   # (64, 2)
        if return_cot:
            return wps, resp.get("cot", "")
        return wps
    finally:
        sock.close()
        ctx.term()
