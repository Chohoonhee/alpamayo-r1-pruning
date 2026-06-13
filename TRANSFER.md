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

## 자료 처음부터 받기 (rsync 없이 새로 다운로드)

이쪽 머신에서 rsync 불가능한 경우 — 인터넷에서 직접 받기. 시간 더 오래 걸리지만 가능.

### A. Alpamayo 1.5 base model (21GB) — HuggingFace gated

```bash
# 1. https://huggingface.co/nvidia/Alpamayo-1.5-10B 에서 "Request access" 클릭
# 2. https://huggingface.co/nvidia/Cosmos-Reason2-8B 에서도 "Request access" 클릭
#    (Alpamayo 1.5의 processor가 Cosmos-Reason2를 참조함)
# 3. 둘 다 승인되면 (보통 same-day, 회사 도메인 이메일 권장):

huggingface-cli login    # HF token (read scope) 입력
huggingface-cli download nvidia/Alpamayo-1.5-10B \
    --local-dir ${DEST_BASE}/weights/Alpamayo-1.5-10B
```

**중요:** 이후 모든 1.5 명령에 `export HF_HUB_OFFLINE=1` 필수 (Cosmos-Reason2를 매번 다시 가져오려 해서 403 뜸).

### B. NAVSIM dataset (~700GB) — HuggingFace

NAVSIM은 OpenDriveLab/AutoVision이 공개한 데이터셋이고 HF에서 직접 받을 수 있음. nuPlan 기반.

```bash
mkdir -p ${DEST_BASE}/navsim_workspace/dataset
cd ${DEST_BASE}/navsim_workspace/dataset

# 메타데이터 (logs, .pkl 파일들)
huggingface-cli download OpenDriveLab/OpenScene-V1.1 \
    --repo-type dataset \
    --include "openscene-v1.1/meta_datas/trainval/*" \
    --include "openscene-v1.1/meta_datas/test/*" \
    --local-dir ./hf_cache

# 위치 정리 — navsim_logs/trainval, navsim_logs/test로 옮김
mv hf_cache/openscene-v1.1/meta_datas/trainval navsim_logs/trainval
mv hf_cache/openscene-v1.1/meta_datas/test navsim_logs/test

# Sensor blobs (카메라 이미지) — 가장 큼
# NAVSIM 공식 가이드: https://github.com/autonomousvision/navsim/blob/main/docs/install.md
# split 별로 받기 (trainval 446GB, test 121GB)
huggingface-cli download OpenDriveLab/OpenScene-V1.1 \
    --repo-type dataset \
    --include "openscene-v1.1/sensor_blobs/trainval/*" \
    --include "openscene-v1.1/sensor_blobs/test/*" \
    --local-dir ./hf_cache

mv hf_cache/openscene-v1.1/sensor_blobs/trainval sensor_blobs/trainval
mv hf_cache/openscene-v1.1/sensor_blobs/test sensor_blobs/test

# Maps (nuPlan, ~1.4GB)
huggingface-cli download autonomousvision/navsim \
    --repo-type dataset \
    --include "maps/*" \
    --local-dir .

# Metric cache (eval 빠르게, 3.6GB) — 직접 생성하거나 NAVSIM 공식 cache 다운
cd ${DEST_BASE}/navsim_workspace/navsim
# 생성:
python navsim/planning/script/run_metric_caching.py train_test_split=navtest
```

대안: NAVSIM 공식 GitHub repo의 `download_data.sh` 가 모든 split 자동 다운로드.
- https://github.com/autonomousvision/navsim
- 단점: AWS S3 의존, 일부 시간대에 throttling

### C. NAVSIM devkit fork (6GB)

```bash
# 우리는 fork된 버전 사용 (alpamayo_agent.py 추가됨)
# 단순한 방법: 우리 fork 통째 옮기기 (rsync)
# OR: 공식 NAVSIM clone + scripts/alpamayo_navsim_agent.py copy
mkdir -p ${DEST_BASE}/navsim_workspace
cd ${DEST_BASE}/navsim_workspace
git clone https://github.com/autonomousvision/navsim navsim
# 우리 alpamayo agent 추가:
cp ${DEST_BASE}/alpamayo_pruning_share/scripts/alpamayo_navsim_agent.py \
   navsim/navsim/agents/alpamayo_agent.py
```

### D. Alpamayo 1.5 source (8GB) — NVIDIA gated repo

`alpamayo1.5/src/alpamayo1_5/` 패키지가 필요. NVIDIA의 private repo이므로 접근 권한 필요. NVIDIA contact를 통해 source clone URL 받아야 함.

```bash
# 권한 있으면:
git clone <nvidia-private-url> ${DEST_BASE}/alpamayo1.5
cd ${DEST_BASE}/alpamayo1.5
uv venv a1_5_venv --python 3.10
uv pip install -e . --python a1_5_venv/bin/python
uv pip install flash-attn==2.8.3 --python a1_5_venv/bin/python --no-build-isolation
```

권한 없으면 NVIDIA 담당자에게 요청 (Alpamayo 1.5는 외부 공개 안 됨).

### E. nuScenes (선택 — NAVSIM만 있어도 됨)

Pruning importance scoring (angular distance) 재현하려면 필요. FT만 하려면 skip.

```bash
# 1. https://www.nuscenes.org 가입
# 2. Download → US/Asia → "v1.0-trainval" full
# 3. .tgz 10개 받아서 풀기:
mkdir -p ${DEST_BASE}/nuscenes/raw_extracted
cd ${DEST_BASE}/nuscenes/raw_extracted
for f in v1.0-trainval01_blobs.tgz v1.0-trainval02_blobs.tgz ...; do
    tar xzf $f
done
```

**총 ~340GB**. NAVSIM만 있어도 FT 학습/eval 모두 가능하므로 보통 생략.

### F. 우리 sweep checkpoint (선택)

GitHub에 코드 + CSV는 있지만 학습된 weight는 없음. 새 머신에서 처음부터 학습하면 v8/v9/v11 자동 재생산. 만약 best 모델 (v9 SAFE+token 2ep) 만 받고 싶으면 NVIDIA NGC / Google Drive 백업 위치 확인:

- 백업: `gdrive:alpamayo-pruning-artifacts/sft_sweep_v9_noprune_safetoken_*` (없으면 학습으로 재생성)

---

다시 다운로드해야 하면 [REPRODUCE.md](docs/REPRODUCE.md) 참고 (Alpamayo 모델 설치 + Python env 설정 더 자세).
