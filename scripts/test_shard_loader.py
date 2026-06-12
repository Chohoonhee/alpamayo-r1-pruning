"""Diagnostic: does SceneLoader actually load scenes from our shard yaml?

Quick CPU-only check — no GPU, no Alpamayo. Just NAVSIM SceneLoader.
"""
import os, sys, yaml
from pathlib import Path

WS = "/home/irteam/ws/alpamayo_pruning/navsim_workspace"
os.environ.setdefault("NAVSIM_DEVKIT_ROOT", f"{WS}/navsim")
os.environ.setdefault("OPENSCENE_DATA_ROOT", f"{WS}/dataset")
os.environ.setdefault("NAVSIM_EXP_ROOT", f"{WS}/exp")
os.environ.setdefault("NUPLAN_MAPS_ROOT", f"{WS}/dataset/maps")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

sys.path.insert(0, f"{WS}/navsim")
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig


def load_filter_from_yaml(path: str) -> SceneFilter:
    with open(path) as f:
        data = yaml.safe_load(f)
    # strip Hydra/instantiate sigils
    for k in list(data.keys()):
        if k.startswith("_"):
            del data[k]
    return SceneFilter(**data)


def main():
    yaml_path = "/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim/planning/script/config/common/train_test_split/scene_filter/navtest_shard0of8.yaml"
    sf = load_filter_from_yaml(yaml_path)
    print(f"loaded filter — tokens={len(sf.tokens) if sf.tokens else 0}, log_names={sf.log_names}, max_scenes={sf.max_scenes}")
    print(f"first 3 tokens: {sf.tokens[:3] if sf.tokens else []}")

    # Try to load scenes
    logs_dir = Path(f"{WS}/dataset/navsim_logs/test")
    if not logs_dir.exists():
        logs_dir = Path(f"{WS}/dataset/navsim_logs/mini")
    print(f"using logs_dir={logs_dir} (exists={logs_dir.exists()})")

    sensor_dir = Path(f"{WS}/dataset/sensor_blobs/test")
    if not sensor_dir.exists():
        sensor_dir = Path(f"{WS}/dataset/sensor_blobs/mini")

    loader = SceneLoader(
        data_path=logs_dir,
        original_sensor_path=sensor_dir,
        scene_filter=sf,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    print(f"loader instantiated. n scenes = {len(loader.tokens)}")
    if len(loader.tokens) == 0:
        print("EMPTY! tokens filter is removing everything.")
        # Compare: try without tokens filter
        sf2 = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True)
        loader2 = SceneLoader(
            data_path=logs_dir, original_sensor_path=sensor_dir,
            scene_filter=sf2, sensor_config=SensorConfig.build_no_sensors(),
        )
        print(f"no-filter loader.n = {len(loader2.tokens)}")
        print(f"first 3 actual tokens: {loader2.tokens[:3]}")
        print(f"our shard tokens are present in NAVSIM tokens? {sf.tokens[0] in set(loader2.tokens) if sf.tokens else False}")
    else:
        print(f"first 3 loaded tokens: {loader.tokens[:3]}")


if __name__ == "__main__":
    main()
