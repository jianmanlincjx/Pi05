# LIT on π0.5

The **Latent Interface Training (LIT)** instantiation of *Breaking the Vision–Action Shortcut: Latent
Interface Training for Generalizable Robot Foundation Models* on π0.5.
Hub, project page, checkpoints: https://github.com/jianmanlincjx/LIT · https://jianmanlincjx.github.io/LIT/ ·
https://huggingface.co/linjianman/LIT

A small package on top of **upstream [LeRobot](https://github.com/huggingface/lerobot) 0.6.2**. Nothing under
`lerobot/policies/pi05/` is patched — the baseline is stock π0.5 — this repository only adds the
`pi05_goal_prior` policy type. The earlier long-form README is kept as [`README_upstream.md`](./README_upstream.md)
(PaliGemma download, `build_init.py`, per-script details).

```bash
pip install lerobot==0.6.2
git clone https://github.com/jianmanlincjx/Pi05.git && pip install -e Pi05    # registers --policy.type=pi05_goal_prior
```

Every LIBERO-Plus number was produced with **`LIBERO_PLUS_FIX_LANG=1`** (see
[`docs/libero_plus_language_bug.md`](docs/libero_plus_language_bug.md)); Overall is the mean over the seven
perturbation axes.

---

## 1. Evaluate the released checkpoint

```bash
hf download linjianman/LIT --include "pi05/*" --local-dir ./LIT_ckpt
CK=./LIT_ckpt/pi05/lit_stage2         # LeRobot policy dir; config.json says "type": "pi05_goal_prior"
```

`lerobot_eval` loads it directly (importing `pi05_goal_prior` — which `pi05gp-eval` does for you — is what
registers the type):

```bash
# LIBERO, official protocol: 50 episodes per task, 4 suites x 10 tasks = 2,000 episodes
pi05gp-eval --policy.path="$CK" --env.type=libero --eval.n_episodes=50 --seed=1000

# LIBERO-Plus: 10,030 perturbed tasks, one episode each
LIBERO_PLUS_FIX_LANG=1 \
pi05gp-eval --policy.path="$CK" --env.type=libero_plus --eval.n_episodes=1 --eval.batch_size=1 --seed=1000
```

Quick check (one task, two episodes, ~20 s):

```bash
python -m lerobot.scripts.lerobot_eval --policy.path="$CK" --policy.device=cuda \
  --env.type=libero --env.task=libero_spatial --env.task_ids="[0]" --eval.n_episodes=2 --eval.batch_size=1 --seed=1000
```

Verified on a fresh machine on 2026-09-10: the baseline (`"type": "pi05"`) and LIT checkpoints both load
and roll out 2/2. Do not put a LeRobot tree that already bundles `lerobot/policies/pi05_goal_prior` on the
path at the same time — the type would be registered twice.

---

## 2. Train, then evaluate

**What you need** — the VLM the policy starts from, and the initial checkpoint built from it:

```bash
hf download google/paligemma-3b-pt-224 --local-dir ./checkpoints/paligemma-3b-pt-224   # gated: accept the licence first
python tools/build_init.py  --paligemma ./checkpoints/paligemma-3b-pt-224 --out ./checkpoints/pi05_init \
                            --state-dim 14 --action-dim 14 --cameras 2
python tools/verify_init.py --init ./checkpoints/pi05_init --paligemma ./checkpoints/paligemma-3b-pt-224
```

`pi05_base` (the fully post-trained VLA) is deliberately **not** used: it would hand the action expert exactly
the prior Stage 1 exists to build. Baseline and both stages start from PaliGemma with a random action
expert, as on the other backbones. Data: LIBERO in LeRobot format, all four suites, `no_noops`.

```bash
bash scripts/train_baseline.sh    # stock pi05, 30K steps
bash scripts/train_stage1.sh      # Stage 1: language + state + chunk-end SE(3) -> action prior, no images (20K steps)
bash scripts/train_stage2.sh      # Stage 2: latent interface, from the Stage-1 checkpoint (30K steps)
```

Then evaluate `<run>/checkpoints/030000/pretrained_model` exactly as in §1. Reported checkpoints:
baseline 030000, Stage 1 020000, LIT 030000.

`tools/probe_latent_attention.py` checks that the action expert actually reads the latents (attention mass
on latent tokens vs. everything else) — the sanity check we ran before trusting any Stage-2 run.

---

## 3. How LIT is integrated in π0.5

π0.5 is a mixture-of-transformers: the PaliGemma VLM and the action expert share one self-attention over a
joint sequence, so "conditioning" is attention masking rather than a separate cross-attention path. That is
exactly where LIT lives, in `src/pi05_goal_prior/`:

| Piece | Where | What it does |
| --- | --- | --- |
| Policy type | `configuration_pi05_goal_prior.py`, `modeling_pi05_goal_prior.py` | `PI05GoalPriorConfig/Policy`, registered as `pi05_goal_prior`; wraps stock π0.5 and adds the pieces below |
| Firewall | attention mask in `modeling_pi05_goal_prior.py` | action-expert tokens cannot attend to image tokens (and, in Stage 1, to nothing visual at all); `mask_language_from_action_expert` additionally keeps language/state out of the action expert so vision reaches it only through the latents |
| Latent interface | `goal_prior.py` (`_SelfAttentionBlock` latent stack) | 100 learnable latent tokens appended to the joint sequence; they attend to the VLM's image and text tokens, and the action expert attends to them |
| Spatial supervision | `GoalPoseDecoder` in `goal_prior.py`, `lambda_pose = 0.3` | 8 of the latents are decoded to the chunk-end SE(3) target used in Stage 1 |
| Stage-1 conditioning | `SE3Encoder` in `goal_prior.py` | encodes the terminal pose into tokens the action expert attends to while images are absent |
| Trainer guard (optional) | `patches/` | skip an optimiser step on a non-finite or exploding gradient norm; never fired in any reported run |

Interface settings match the other three backbones and were not tuned per model: `num_latents=100`,
`num_pose_tokens=8`, `latent_dim=768`, `inner_dim=512`, `lambda_pose=0.3`.
