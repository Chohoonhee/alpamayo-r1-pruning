#!/bin/bash
# Priority 3: Early-stopping greedy — patches run_iterative_greedy_navsim.py
# to stop when align doesn't improve over best-so-far by ε for N rounds.
# Re-runs on R1 and 1.5 NAVSIM to validate "auto-found K" matches the
# peak-K we manually identified.

set -uo pipefail
source /home/irteam/miniconda/etc/profile.d/conda.sh

REPO=/home/irteam/ws/alpamayo_pruning_share
SCRIPTS=$REPO/scripts
LOGS=$SCRIPTS/logs
COMMIT=$SCRIPTS/auto_commit.sh
STATUS=$SCRIPTS/update_status.py
ANALYSIS=$SCRIPTS/generate_analysis_report.py
TRANSCRIPT=$SCRIPTS/extract_conversation.py
QLOG=$LOGS/earlystop_queue.log

# Wait for stability done
echo "[$(date +%H:%M:%S)] earlystop: waiting for stability ..." | tee -a $QLOG
while ! grep -q "STABILITY DONE" $LOGS/stability_queue.log 2>/dev/null; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] stability done, building early-stop greedy" | tee -a $QLOG

# Write the early-stop variant inline (small patch on top of existing greedy)
cat > $SCRIPTS/run_iterative_greedy_navsim_earlystop.py <<'PY'
"""Early-stopping iterative-greedy: stop when align hasn't improved by ε
for patience rounds. Reports the auto-determined K and uses peak drop_set.
Same input format and outputs as run_iterative_greedy_navsim.py.
"""
from __future__ import annotations

from paths import (
    ALPAMAYO_15_WEIGHTS, ALPAMAYO_R1_WEIGHTS,
    add_alpamayo_to_syspath,
)
import argparse, json, pickle, time
from contextlib import contextmanager
import torch
from maneuver_classifiers import (
    classify_cot_rule, classify_traj_4class, alignment_match,
)


@contextmanager
def bypass_layers(model, layer_indices):
    layers = model.vlm.language_model.layers
    originals = {i: layers[i].forward for i in layer_indices}
    def _identity(hidden_states, *a, **kw): return hidden_states
    for i in layer_indices: layers[i].forward = _identity
    try: yield
    finally:
        for i, orig in originals.items(): layers[i].forward = orig


INTEGER_KEYS = {"input_ids","attention_mask","token_type_ids","labels","position_ids"}


def run_inference(model, processor, sample, device, backbone):
    if backbone == "15":
        from alpamayo1_5 import helper as h15
        frames = torch.from_numpy(sample["image_frames"]).to(device)
        n_cam, n_frames = frames.shape[0], frames.shape[1]
        flat = frames.reshape(n_cam * n_frames, *frames.shape[2:])
        cam_idx = torch.tensor(sample["camera_indices"], dtype=torch.long).to(device)
        messages = h15.create_message(frames=flat, camera_indices=cam_idx,
                                       nav_text=(sample["nav_text"] or None))
        tok = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
    else:
        from alpamayo_r1.helper import create_message as cmR1
        frames = torch.from_numpy(sample["image_frames"]).to(device)
        n_cam, n_frames = frames.shape[0], frames.shape[1]
        flat = frames.reshape(n_cam * n_frames, *frames.shape[2:])
        messages = cmR1(flat)
        tok = model.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt",
        )
    td = {}
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            td[k] = v.long().to(device) if (k in INTEGER_KEYS or not v.is_floating_point()) else v.to(device=device, dtype=torch.bfloat16)
        else:
            td[k] = v
    ego_xyz = torch.from_numpy(sample["ego_history_xyz"]).float().to(device)
    ego_rot = torch.from_numpy(sample["ego_history_rot"]).float().to(device)
    data = {"tokenized_data": td, "ego_history_xyz": ego_xyz, "ego_history_rot": ego_rot}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if backbone == "15":
            pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data=data, top_p=0.98, temperature=0.6,
                num_traj_samples=1, max_generation_length=256, return_extra=True)
        else:
            pred_xyz, _, extra = model.sample_trajectories_from_data_with_vlm_rollout(
                data, num_traj_samples=1, num_traj_sets=1,
                return_extra=True, top_p=0.98, temperature=0.6, max_new_tokens=256)
    action64 = pred_xyz[0, 0, 0, :, :2].detach().float().cpu().numpy()
    cot_text = str(extra["cot"][0, 0, 0])
    return cot_text, action64


def measure_alignment(model, processor, samples, device, backbone, drop_set):
    matches = 0
    n = len(samples)
    with bypass_layers(model, drop_set):
        for s in samples:
            try:
                cot, act = run_inference(model, processor, s, device, backbone)
                cot_l = classify_cot_rule(cot)
                act_l = classify_traj_4class(act.tolist())
                matches += alignment_match(cot_l, act_l)
            except Exception:
                pass
    return matches / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", choices=["15","r1"], required=True)
    ap.add_argument("--orig_weights", default=None)
    ap.add_argument("--samples_pkl", required=True)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--max_rounds", type=int, default=15)
    ap.add_argument("--patience", type=int, default=2,
                    help="stop after N rounds without improvement over best-so-far")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="alignment improvement threshold")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    if args.backbone == "15":
        add_alpamayo_to_syspath(v15=True)
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
        from alpamayo1_5 import helper as h15
        weights = args.orig_weights or str(ALPAMAYO_15_WEIGHTS)
        print(f"[load] 1.5: {weights}", flush=True)
        model = Alpamayo1_5.from_pretrained(weights, dtype=torch.bfloat16).to(args.device)
        processor = h15.get_processor(model.tokenizer)
    else:
        add_alpamayo_to_syspath(r1=True)
        from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
        weights = args.orig_weights or str(ALPAMAYO_R1_WEIGHTS)
        print(f"[load] R1: {weights}", flush=True)
        model = AlpamayoR1.from_pretrained(weights, dtype=torch.bfloat16).to(args.device)
        processor = None
    model.eval()
    n_layers = len(model.vlm.language_model.layers)
    print(f"[load] {n_layers} VLM layers detected", flush=True)

    with open(args.samples_pkl, "rb") as f: samples = pickle.load(f)
    if args.n_samples > 0: samples = samples[:args.n_samples]
    print(f"[cal] {len(samples)} calibration samples", flush=True)

    base_align = measure_alignment(model, processor, samples, args.device, args.backbone, set())
    print(f"[baseline] align={base_align:.4f}", flush=True)

    drop_set = []
    best_align = base_align
    best_drop = []
    rounds_no_improve = 0
    history = [{"round": 0, "drop_set": [], "best_layer": None, "best_align": base_align}]

    for r in range(1, args.max_rounds + 1):
        t0 = time.time()
        candidates = sorted(set(range(n_layers)) - set(drop_set))
        scores = {}
        for ci, c in enumerate(candidates):
            trial = drop_set + [c]
            a = measure_alignment(model, processor, samples, args.device, args.backbone, set(trial))
            scores[c] = a
            print(f"  [r{r} {ci+1}/{len(candidates)} ℓ={c}] align={a:.3f}", flush=True)

        best_layer_this = max(scores, key=scores.get)
        best_align_this = scores[best_layer_this]
        drop_set.append(best_layer_this)
        history.append({
            "round": r, "drop_set": list(drop_set),
            "best_layer": best_layer_this, "best_align": best_align_this,
            "round_elapsed_s": time.time() - t0,
        })
        improvement = best_align_this - best_align
        print(f"\n[round {r}] +ℓ={best_layer_this} → drop={drop_set}  "
              f"align={best_align_this:.4f}  Δ={improvement:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

        if improvement > args.epsilon:
            best_align = best_align_this
            best_drop = list(drop_set)
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1
            print(f"  [no improvement] {rounds_no_improve}/{args.patience}", flush=True)
            if rounds_no_improve >= args.patience:
                print(f"\n[EARLY STOP] no improvement for {args.patience} rounds. "
                      f"Best K={len(best_drop)} drop={best_drop} align={best_align:.4f}",
                      flush=True)
                break

    final_drop = best_drop if best_drop else drop_set
    with open(args.out_json, "w") as f:
        json.dump({
            "backbone": args.backbone, "weights": weights,
            "samples_pkl": args.samples_pkl,
            "n_calibration": len(samples),
            "baseline_align": base_align,
            "best_align": best_align, "best_K": len(best_drop),
            "final_drop_set": final_drop,
            "history": history,
            "early_stopped": rounds_no_improve >= args.patience,
        }, f, indent=2)
    print(f"\nSaved → {args.out_json}")

    meta_path = args.out_json.replace(".json", "_meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "dropped_layers": sorted(final_drop),
            "policy": "iterative_greedy_navsim_earlystop",
            "backbone": args.backbone,
            "source": args.out_json,
            "K": len(final_drop),
        }, f, indent=2)
    print(f"Saved meta → {meta_path}")


if __name__ == "__main__":
    main()
PY
echo "wrote run_iterative_greedy_navsim_earlystop.py" | tee -a $QLOG

conda activate alpamayo_b2d
export HF_HUB_OFFLINE=1
cd $SCRIPTS

PICK_GPU() {
    for i in 0 1 2 3 4 5 6 7; do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $i)
        if [ "$used" -lt 5000 ]; then echo $i; return; fi
    done
    echo 0
}

G1=$(PICK_GPU); sleep 2; G2=$(PICK_GPU)
[ "$G2" = "$G1" ] && G2=$((G1 + 1))

# R1 + 1.5 with early stop on NAVSIM samples 0:50
nohup bash -c "
export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$G1
python run_iterative_greedy_navsim_earlystop.py --backbone r1 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --n_samples 50 \
    --max_rounds 15 --patience 2 --epsilon 0.01 --device cuda:0 \
    --out_json $LOGS/greedyR1_navsim_earlystop.json
" > $LOGS/greedyR1_navsim_earlystop.log 2>&1 &
P1=$!
nohup bash -c "
export HF_HUB_OFFLINE=1; export CUDA_VISIBLE_DEVICES=$G2
python run_iterative_greedy_navsim_earlystop.py --backbone 15 \
    --samples_pkl $LOGS/navsim_samples_100.pkl --n_samples 50 \
    --max_rounds 15 --patience 2 --epsilon 0.01 --device cuda:0 \
    --out_json $LOGS/greedy15_navsim_earlystop.json
" > $LOGS/greedy15_navsim_earlystop.log 2>&1 &
P2=$!

echo "early-stop R1 pid=$P1 gpu=$G1" | tee -a $QLOG
echo "early-stop 1.5 pid=$P2 gpu=$G2" | tee -a $QLOG

wait $P1 $P2
echo "[$(date +%H:%M:%S)] early-stop greedy done" | tee -a $QLOG

# Eval early-stop policies on holdout + nuScenes val
for bb in r1 15; do
    META=$LOGS/greedy${bb^^}_navsim_earlystop_meta.json
    [ "$bb" = "15" ] && META=$LOGS/greedy15_navsim_earlystop_meta.json
    G1=$(PICK_GPU); sleep 2; G2=$(PICK_GPU)
    [ "$G2" = "$G1" ] && G2=$((G1 + 1))
    nohup bash /tmp/eval_peak_one.sh $G1 $bb $META nusc   evalES_${bb}_navsim_on_nusc   > $LOGS/evalES_${bb}_navsim_on_nusc.log 2>&1 &
    EP1=$!
    nohup bash /tmp/eval_peak_one.sh $G2 $bb $META navsim evalES_${bb}_navsim_on_navsim > $LOGS/evalES_${bb}_navsim_on_navsim.log 2>&1 &
    EP2=$!
    wait $EP1 $EP2
done

python $ANALYSIS 2>&1 | tail -1 | tee -a $QLOG
python $STATUS 2>&1 | tail -1 | tee -a $QLOG
python $TRANSCRIPT 2>&1 | tail -1 | tee -a $QLOG
bash $COMMIT "early-stopping greedy: auto-found K + holdout evals" 2>&1 | tail -2 | tee -a $QLOG

echo "" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === EARLY-STOP DONE ===" | tee -a $QLOG
echo "[$(date +%H:%M:%S)] === ALL PRIORITIES DONE ===" | tee -a $QLOG
