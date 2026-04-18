#!/bin/bash
set -u
export NAVSIM_DEVKIT_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim
export OPENSCENE_DATA_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/dataset
export NAVSIM_EXP_ROOT=/home/irteam/ws/alpamayo_pruning/navsim_workspace/exp
export NUPLAN_MAPS_ROOT=$OPENSCENE_DATA_ROOT/maps
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export HYDRA_FULL_ERROR=1

PY=/home/irteam/ws/alpamayo_pruning/navsim_workspace/navsim/navsim_venv/bin/python
CACHE=$NAVSIM_EXP_ROOT/metric_cache
LOG_DIR=$NAVSIM_EXP_ROOT/logs
mkdir -p $LOG_DIR

log() { echo "[$(date +%H:%M:%S)] $*"; }

log "서버 ready 대기 중..."
for port in 5557 5558 5559 5560; do
  until grep -q "ready on" /home/irteam/ws/alpamayo_pruning/scripts/server_${port}.log 2>/dev/null; do
    sleep 5
  done
  log "server $port ready"
done

log "sample500 R1 eval 시작..."
cd $NAVSIM_DEVKIT_ROOT
$PY navsim/planning/script/run_pdm_score_one_stage.py \
    agent=alpamayo_agent \
    train_test_split=navtest \
    train_test_split.scene_filter.max_scenes=500 \
    metric_cache_path=$CACHE \
    experiment_name=alpamayo_sample500_r1 \
    worker=single_machine_thread_pool \
    worker.max_workers=8 \
    2>&1 | tee $LOG_DIR/pdm_sample500_r1.log

log "DONE"
