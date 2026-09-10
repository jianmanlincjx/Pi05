# Reproducing the LIBERO / LIBERO-Plus results

`README.md` covers installation, weights, and the three training scripts. This page adds the
evaluation protocols and says which checkpoint produced which reported number.

## Training

Two things, not interchangeable (see `README.md` for the full walkthrough):

```bash
scripts/train_baseline.sh                        # stock pi0.5, unmodified
scripts/train_stage1.sh && scripts/train_stage2.sh   # LIT, both stages in order
```

- **Stage 1** — language + robot state + each chunk's terminal SE(3) end-effector pose,
  no image observations. 20,000 steps, batch 64, lr 1e-4, warmup 4,000.
- **Stage 2** — vision restored, but only through 100 learnable latents that aggregate the
  backbone; raw visual tokens are masked out of the action expert and 8 latents reconstruct
  the same pose (`lambda_pose = 0.3`). 30,000 steps, batch 16, lr 1e-4, warmup 5,000.

Stage 1's SE(3) encoder is training-time scaffolding: Stage 2 drops it and the latents
predict the pose from vision, so no privileged input is needed at inference.

**pi0.5-specific.** The other backbones inject the latents as cross-attention KV. pi0.5 is a
MoT with shared self-attention, so the action expert can read language and state directly and
bypass the interface entirely — measured at 66% of its attention on language and 0.5% on the
latents. `mask_language_from_action_expert` hides the language and state columns from the
action rows, making the latents the only route in. It never masks the action rows' own
columns, which would cut the flow-matching self-attention. This is part of the method on
pi0.5, not tuning.

## Evaluation

### LIBERO (in-distribution)

Official protocol: 50 episodes per task, official horizons, 4 suites x 10 tasks = 2,000 episodes.

```bash
pi05gp-eval --policy.path=<run>/checkpoints/030000/pretrained_model \
            --env.type=libero --eval.n_episodes=50 --seed=1000
```

### LIBERO-Plus (out of distribution)

[LIBERO-Plus](https://github.com/sylvestf/LIBERO-plus) is 10,030 perturbed tasks over seven
axes, one episode each.

```bash
LIBERO_PLUS_FIX_LANG=1 \
pi05gp-eval --policy.path=<run>/checkpoints/030000/pretrained_model \
            --env.type=libero_plus --eval.n_episodes=1 --eval.batch_size=1 --seed=1000
```

**`LIBERO_PLUS_FIX_LANG=1` is required.** Upstream LIBERO-Plus derives the instruction from
the perturbed file name, so on every non-language axis the policy is otherwise fed strings
like `... view 0 0 100 2 352 initstate 0`. See
[`docs/libero_plus_language_bug.md`](docs/libero_plus_language_bug.md); numbers produced
without the fix are not comparable.

## Verified on a fresh machine

2026-09-10. Base: **upstream LeRobot 0.6.2** with this package on `PYTHONPATH` (or `pip install -e .`).
Both released checkpoints load directly with `lerobot_eval` and roll out (2/2 on `libero_spatial` task 0
each) — `config.json` carries `"type": "pi05"` for the baseline and `"type": "pi05_goal_prior"` for LIT,
and importing `pi05_goal_prior` is what registers the latter. Do not also have a LeRobot tree that already
bundles `lerobot/policies/pi05_goal_prior` on the path: the type would be registered twice.

```bash
python -m lerobot.scripts.lerobot_eval --policy.path=<ckpt_dir> --policy.device=cuda   --env.type=libero --env.task=libero_spatial --env.task_ids="[0]" --eval.n_episodes=2 --eval.batch_size=1 --seed=1000
```

(`lerobot_eval` in LeRobot 0.6.2 has no `--eval.max_episodes_rendered`; videos are written by default.)

## Checking the latents are actually used

```bash
python tools/probe_latent_attention.py --policy.path=<stage2>/pretrained_model
```

On pi0.5 this is worth running: without the language/state mask the action expert routes
almost nothing through the latents, and the method degrades to the baseline.

## Checkpoints

`paper_ckpt/pi05/` — `baseline_030000`, `lit_stage1_020000`, `lit_stage2_030000`. Each is a
self-contained LeRobot `pretrained_model` directory (`model.safetensors`, `config.json`,
`train_config.json`, tokenizer, normaliser tensors).

## The same method on other backbones

MolmoAct2 `jianmanlincjx/Molmoact2` · FAST-WAM `jianmanlincjx/fastwam` · ImageWAM `jianmanlincjx/ImageWAM`
