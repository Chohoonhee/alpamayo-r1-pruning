# 다른 머신으로 이어가기 — Transfer Guide

`/home/irteam/ws/alpamayo_pruning_share/` 에서 시작해서, 새 머신에서 어디서부터 다시 시작하면 되는지 정리.

## 전체 사이즈 요약

| 항목 | 사이즈 | 필수 |
|---|---|---|
| `alpamayo_pruning_share/` (이 repo: 코드 + 결과 CSV) | 2.2 GB | yes |
| `alpamayo_pruning/weights/Alpamayo-1.5-10B/` (base model) | 21 GB | yes |
| `alpamayo_pruning/weights/sft_sweep_v9_noprune_safetoken_52k_lr1e4*/` (best 2개) | 42 GB | recommended |
| `alpamayo_pruning/navsim_workspace/dataset/sensor_blobs/trainval/` | **446 GB** | yes (학습용) |
| `alpamayo_pruning/navsim_workspace/dataset/sensor_blobs/test/` | 121 GB | yes (eval용) |
| `alpamayo_pruning/navsim_workspace/dataset/navsim_logs/{trainval,test}/` | 15 GB | yes |
| `alpamayo_pruning/navsim_workspace/dataset/maps/` | 1.4 GB | yes |
| `alpamayo_pruning/navsim_workspace/exp/metric_cache/` | 3.6 GB | yes (eval 빠름) |
| `alpamayo_pruning/navsim_workspace/navsim/` (devkit fork) | 6 GB | yes |
| `alpamayo_pruning/alpamayo1.5/` (1.5 source + `a1_5_venv`) | 8.4 GB | yes |
| **총** | **~666 GB** | |

다른 sweep checkpoint (54개 × 21GB = ~1.1TB)는 결과 CSV에만 의존하므로 옮기지 않아도 됨.

## 1) 전송 명령

기존 머신 (이쪽):
```bash
DEST_HOST=newmachine
DEST_BASE=/data/alpamayo

# 코드 + 결과
rsync -av --progress /home/irteam/ws/alpamayo_pruning_share/ ${DEST_HOST}:${DEST_BASE}/alpamayo_pruning_share/

# Base model + best checkpoints
rsync -av --progress \
    /home/irteam/ws/alpamayo_pruning/weights/Alpamayo-1.5-10B \
    /home/irteam/ws/alpamayo_pruning/weights/sft_sweep_v9_noprune_safetoken_52k_lr1e4 \
    /home/irteam/ws/alpamayo_pruning/weights/sft_sweep_v9_noprune_safetoken_52k_lr1e4_2ep \
    ${DEST_HOST}:${DEST_BASE}/weights/

# NAVSIM dataset (가장 큼, 시간 걸림)
rsync -av --progress \
    /home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset \
    /home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim \
    /home/irteam/ws/alpamayo_pruning/navsim_workspace/exp \
    ${DEST_HOST}:${DEST_BASE}/navsim_workspace/

# Alpamayo 1.5 source (a1_5_venv 포함)
rsync -av --progress \
    /home/irteam/ws/alpamayo_pruning/alpamayo1.5 \
    ${DEST_HOST}:${DEST_BASE}/alpamayo1.5/
```

## 2) 새 머신에서 환경변수 설정

`~/.bashrc` 또는 `${DEST_BASE}/env.sh`:
```bash
export DEST_BASE=/data/alpamayo
export ALPAMAYO_15_SRC=${DEST_BASE}/alpamayo1.5
export ALPAMAYO_WEIGHTS_DIR=${DEST_BASE}/weights
export NAVSIM_WORKSPACE=${DEST_BASE}/navsim_workspace
export OUTPUTS_DIR=${DEST_BASE}/alpamayo_pruning_share/scripts

# NAVSIM env vars (run_ft_sweep_*.sh이 export)
export HF_HUB_OFFLINE=1
export NAVSIM_DEVKIT_ROOT=${NAVSIM_WORKSPACE}/navsim
export OPENSCENE_DATA_ROOT=${NAVSIM_WORKSPACE}/dataset
export NAVSIM_EXP_ROOT=${NAVSIM_WORKSPACE}/exp
export NUPLAN_MAPS_ROOT=${OPENSCENE_DATA_ROOT}/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

이미 sweep scripts 안에 export 다 있으니까 사실 위 5개만 필요.

## 3) Python 환경 두 개 재구성

### a. `alpamayo_b2d` (conda — 학습/일반 작업용)
```bash
conda create -n alpamayo_b2d python=3.12 -y
conda activate alpamayo_b2d
pip install torch==2.8.0 numpy==1.26.4 \
    transformers peft einops accelerate \
    safetensors pillow pyquaternion nuscenes-devkit \
    flash-attn==2.8.3 --no-build-isolation
```

### b. `a1_5_venv` (uv — Alpamayo 1.5 inference server용)
```bash
cd ${ALPAMAYO_15_SRC}
# 만약 source 안에 ".venv" 이미 있으면 그대로 사용 가능
# 새로 만들려면:
uv venv a1_5_venv --python 3.10
uv pip install -e . --python a1_5_venv/bin/python
uv pip install flash-attn==2.8.3 --python a1_5_venv/bin/python --no-build-isolation
```

### c. `navsim_venv` (NAVSIM eval — Python 3.9)
```bash
cd ${NAVSIM_WORKSPACE}/navsim
python3.9 -m venv navsim_venv
source navsim_venv/bin/activate
pip install -e .
```

## 4) 검증

```bash
cd ${DEST_BASE}/alpamayo_pruning_share/scripts
conda activate alpamayo_b2d
python paths.py              # 모든 경로 print + 존재 확인
python -c "from navsim_sft_dataset import NavsimSFTDataset; ds = NavsimSFTDataset(n_samples=5); print('OK:', len(ds))"
```

## 5) 학습 이어서 시작

기존 머신의 진행 상황: **v9 SAFE+token combo lr=1e-4 2ep = PDMS 0.8214 (best)**.
v11 r3-r5 (lr=5e-4 / 260k stride1 / 5 epoch) 미실행 — 재개:

```bash
cd ${DEST_BASE}/alpamayo_pruning_share/scripts
nohup bash run_ft_sweep_v11_scaleup.sh > logs/ft_sweep_v11_stdout.log 2>&1 &
```

v11 script 안의 `[skip-train] $RECIPE exists` 로직이 자동으로:
- r1 (이미 완료, CSV에 결과) skip
- r2 (weights 있지만 eval 안 됨) eval만 retry
- r3, r4, r5 처음부터 학습

## 6) 결과 비교

새 머신 학습 끝나면:
```bash
cat logs/ft_sweep_results.csv | grep "^v11" | sort -t',' -k8 -gr
```

Best (현재 0.8214) 대비 새 recipe 결과 확인.

## 자료 출처

- Base model: HuggingFace `nvidia/Alpamayo-1.5-10B` (gated)
- NAVSIM dataset: navsim.org (registration required)
- nuScenes: nuscenes.org (registration required, optional — NAVSIM trainval만 있어도 됨)

다시 다운로드해야 하면 [REPRODUCE.md](docs/REPRODUCE.md) 참고.
