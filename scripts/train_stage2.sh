#!/usr/bin/env bash
# Stage 2: the oracle retires and learnable latents take its slot.
#
# This is the configuration that produced the reported results -- every flag below is copied
# from that run's train_config.json. See "Training the goal-prior version" in the README.
#
# Four of these flags are part of the method, not tuning knobs:
#
#   mask_image_from_action_expert=true
#   mask_language_from_action_expert=true
#   normalize_latent_keys=true
#   use_syn_gate=false
#
# A run with any of them changed still trains and still converges -- to a lower loss, in fact
# -- but it is a different and weaker model, and nothing in the loss curve says so. Two earlier
# configurations that differed only in these landed within 0.7 points of each other on
# LIBERO-Plus, then jumped 10 points once they were all set as above. Ablate them as labelled
# experiments if you want to; do not quietly edit them here.
set -euo pipefail

REPO_ID="${REPO_ID:?}"
DATA_ROOT="${DATA_ROOT:?}"
STAGE1="${STAGE1:?path to the stage 1 pretrained_model directory}"
OUT="${OUT:-./outputs/stage2}"
CHUNK="${CHUNK:-10}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"
STEPS="${STEPS:-30000}"
# The reported run was 18 x 7 = 126: seven processes only because one GPU on that node was
# faulty. 16 x 8 = 128 is the closest effective batch on a healthy eight-GPU node.
BATCH="${BATCH:-16}"
NPROC="${NPROC:-8}"

accelerate launch --num_processes="$NPROC" -m pi05_goal_prior.cli \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DATA_ROOT" \
  --policy.type=pi05_goal_prior \
  --policy.pretrained_path="$STAGE1" \
  --policy.goal_prior_stage=stage2 \
  --policy.chunk_size="$CHUNK" \
  --policy.n_action_steps="$CHUNK" \
  --policy.mask_image_from_action_expert=true \
  --policy.mask_language_from_action_expert=true \
  --policy.normalize_latent_keys=true \
  --policy.use_syn_gate=false \
  --policy.context_token_dropout=0.0 \
  --policy.context_blackout_prob=0.0 \
  --policy.empty_cameras="$EMPTY_CAMERAS" \
  --policy.optimizer_lr=1e-4 \
  --policy.aggregator_optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=5000 \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.normalization_mapping='{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}' \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name="$(basename "$OUT")" \
  --batch_size="$BATCH" --num_workers=8 --steps="$STEPS" \
  --save_freq=5000 --log_freq=50 --seed=1000 --wandb.enable=false
