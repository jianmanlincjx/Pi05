#!/usr/bin/env bash
# Stage 2: the oracle retires and learnable latents take its slot.
#
# THIS IS THE CONFIGURATION THAT PRODUCED THE REPORTED RESULTS. Every flag below is copied
# from the train_config.json of that run. Three of them decide whether the mechanism runs at
# all, and two earlier configurations that differed only in those three landed within 0.7
# points of each other on LIBERO-Plus -- because in both of them the action expert ignored
# the latents entirely and was reading the language prompt instead.
#
#   mask_image_from_action_expert     hide the image columns from the action rows
#   mask_language_from_action_expert  hide the language/state columns as well
#   normalize_latent_keys             rescale injected keys to the backbone key norm
#
# pi05 runs the backbone and the action expert in ONE shared attention, and writes the robot
# state into the language prompt as 256-way discretised text. Hiding images alone therefore
# leaves the action rows a complete fallback -- task text plus proprioception -- and they
# take it: 66% of their attention on language, 0.5% on the 100 latents. Hiding language too
# removes the fallback, and the latents then draw ~9%, against 12.7% for the Stage 1 oracle
# channel. Measured effect of that one switch: latent attention 0.09% -> 8.9%, clean LIBERO
# 83.10 -> 91.80, LIBERO-Plus on libero_spatial 72.0 -> 82.91.
#
# Turning any of the three off reproduces a version whose numbers look plausible and whose
# mechanism does nothing. Do not "simplify" them away.
set -euo pipefail

REPO_ID="${REPO_ID:?}"
DATA_ROOT="${DATA_ROOT:?}"
STAGE1="${STAGE1:?path to the stage 1 checkpoint's pretrained_model directory}"
OUT="${OUT:-./outputs/stage2}"
CHUNK="${CHUNK:-10}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"
STEPS="${STEPS:-30000}"
BATCH="${BATCH:-18}"
NPROC="${NPROC:-7}"

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
