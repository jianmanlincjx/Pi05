#!/usr/bin/env bash
# Stage 2: the oracle retires, learnable latents take its slot.
#
# 100 latents are refreshed at every backbone layer by cross-attending to that layer's hidden
# states, then injected into the action rows as extra key/value columns. The first 8 carry an
# SE(3) pose-reconstruction loss; the rest are free context.
#
# The three masking switches are what decide whether the mechanism actually runs, and they are
# the hard-won part of this recipe:
#
#   mask_image_from_action_expert     hides the image columns from the action rows
#   mask_language_from_action_expert  hides the language/state columns as well
#   normalize_latent_keys             rescales the injected keys to the backbone key norm
#
# pi05 runs the backbone and the action expert in ONE shared attention, and writes the robot
# state into the language prompt as 256-way discretised text. Masking images alone therefore
# leaves the action rows a complete fallback -- task text plus proprioception -- and they take
# it: measured 66% of their attention on language and 0.5% on the latents, whether or not the
# keys were rescaled. Masking language too removes the fallback, and the latents then draw
# ~9%, against 12.7% for Stage 1's oracle channel.
#
# If you keep the language columns, expect the latents to be ignored and the aggregator to act
# as an auxiliary loss rather than an information channel.
set -euo pipefail

REPO_ID="${REPO_ID:?}"
DATA_ROOT="${DATA_ROOT:?}"
STAGE1="${STAGE1:?path to the stage 1 checkpoint's pretrained_model}"
OUT="${OUT:-./outputs/stage2}"
CHUNK="${CHUNK:-10}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-16}"
NPROC="${NPROC:-8}"

accelerate launch --num_processes="$NPROC" -m pi05_goal_prior.cli \
  --dataset.repo_id="$REPO_ID" --dataset.root="$DATA_ROOT" \
  --policy.type=pi05_goal_prior \
  --policy.pretrained_path="$STAGE1" \
  --policy.goal_prior_stage=stage2 \
  --policy.chunk_size="$CHUNK" --policy.n_action_steps="$CHUNK" \
  --policy.mask_image_from_action_expert=true \
  --policy.mask_language_from_action_expert=true \
  --policy.normalize_latent_keys=true \
  --policy.use_syn_gate=false \
  --policy.context_token_dropout=0.0 --policy.context_blackout_prob=0.0 \
  --policy.empty_cameras="$EMPTY_CAMERAS" \
  --policy.optimizer_lr=1e-4 --policy.aggregator_optimizer_lr=1e-4 \
  --policy.scheduler_warmup_steps=5000 --policy.scheduler_decay_steps="$STEPS" \
  --policy.normalization_mapping='{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}' \
  --policy.gradient_checkpointing=true --policy.dtype=bfloat16 --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$OUT" --job_name="$(basename "$OUT")" \
  --batch_size="$BATCH" --num_workers=8 --steps="$STEPS" \
  --save_freq=5000 --log_freq=50 --seed=1000 --wandb.enable=false
