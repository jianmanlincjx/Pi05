#!/usr/bin/env bash
# Stage 1: vision-free, oracle-goal-conditioned action prior.
#
# The VLM is frozen and the images are dropped entirely -- not merely frozen, since the point
# is that appearance is unavailable, not un-trained. The action expert learns "given where the
# gripper must end up, produce the trajectory that gets there". Stage 2 then replaces the
# oracle with latents inferred from vision.
#
# The goal pose is the state at t + chunk_size, read from the dataset. It is injected as extra
# key/value columns visible only to the action rows, so the frozen VLM never sees it. The goal
# horizon follows CHUNK -- target_pose_delta_index is left unset and the config resolves it to
# chunk_size -- so changing CHUNK moves how far ahead the oracle points, not just how many
# actions are predicted. Stage 2 inherits the resolved value from this checkpoint's config.
set -euo pipefail

REPO_ID="${REPO_ID:?}"
DATA_ROOT="${DATA_ROOT:?}"
INIT="${INIT:?path built by tools/build_init.py}"
OUT="${OUT:-./outputs/stage1}"
CHUNK="${CHUNK:-10}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"
STEPS="${STEPS:-20000}"
BATCH="${BATCH:-64}"      # larger than Stage 2: no image towers, so the step is much cheaper
NPROC="${NPROC:-8}"

accelerate launch --num_processes="$NPROC" -m pi05_goal_prior.cli \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DATA_ROOT" \
  --policy.type=pi05_goal_prior \
  --policy.pretrained_path="$INIT" \
  --policy.goal_prior_stage=stage1 \
  --policy.train_expert_only=true \
  --policy.chunk_size="$CHUNK" --policy.n_action_steps="$CHUNK" \
  --policy.empty_cameras="$EMPTY_CAMERAS" \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=4000 --policy.scheduler_decay_steps="$STEPS" \
  --policy.normalization_mapping='{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}' \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name="$(basename "$OUT")" \
  --batch_size="$BATCH" --num_workers=8 --steps="$STEPS" \
  --save_freq=5000 --log_freq=50 --seed=1000 --wandb.enable=false
