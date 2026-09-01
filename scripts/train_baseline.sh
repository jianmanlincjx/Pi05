#!/usr/bin/env bash
# pi05 baseline: PaliGemma VLM + randomly initialised action expert, trained on one dataset.
#
# This is the reference the goal-prior method is measured against, and it is plain upstream
# lerobot -- nothing in src/pi05_goal_prior is loaded. Edit the block below for your robot.
set -euo pipefail

# ---- robot / dataset ----------------------------------------------------------------
REPO_ID="${REPO_ID:?e.g. yourname/yam_pick_place}"
DATA_ROOT="${DATA_ROOT:?path to the LeRobotDataset}"
INIT="${INIT:?path built by tools/build_init.py}"
OUT="${OUT:-./outputs/pi05_baseline}"

# chunk_size is how many actions the model predicts; n_action_steps is how many are executed
# before it replans. Executing the whole chunk (equal values) matches how we train; a shorter
# n_action_steps replans more often at the cost of more forward passes.
CHUNK="${CHUNK:-10}"
NAS="${NAS:-10}"

# pi05 always runs three image slots. Set this to 3 minus the number of cameras you have, so
# the unused slots are filled with a constant -1 frame and masked out.
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

# ---- optimisation -------------------------------------------------------------------
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-16}"          # per process
NPROC="${NPROC:-8}"
LR="${LR:-1e-4}"

accelerate launch --num_processes="$NPROC" -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$REPO_ID" \
  --dataset.root="$DATA_ROOT" \
  --policy.type=pi05 \
  --policy.pretrained_path="$INIT" \
  --policy.chunk_size="$CHUNK" \
  --policy.n_action_steps="$NAS" \
  --policy.empty_cameras="$EMPTY_CAMERAS" \
  --policy.optimizer_lr="$LR" \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.normalization_mapping='{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}' \
  --policy.gradient_checkpointing=true \
  --policy.dtype=bfloat16 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" \
  --job_name="$(basename "$OUT")" \
  --batch_size="$BATCH" \
  --num_workers=8 \
  --steps="$STEPS" \
  --save_freq=5000 \
  --log_freq=50 \
  --seed=1000 \
  --wandb.enable=false
