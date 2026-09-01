# Pi05 — π0.5 baseline and goal-prior on LeRobot

Training code for a π0.5 baseline and a two-stage goal-prior variant, built as a small
package on top of upstream [LeRobot](https://github.com/huggingface/lerobot). Nothing under
`lerobot/policies/pi05/` is patched, so **the baseline is stock π0.5** — this repo only adds a
new policy type and one optional guard in the trainer.

Running the baseline needs none of the model code here. What it does need is the starting
checkpoint, which this repo builds, and the dataset conventions, which it documents.

```
1.  install lerobot                       see Install
2.  download PaliGemma                    see Weights
3.  tools/build_init.py                   PaliGemma VLM + randomly initialised action expert
4.  convert your data to LeRobotDataset   see docs/adapting_to_a_new_robot.md
5.  scripts/train_baseline.sh             train
6.  scripts/infer_realrobot.py            run it on the robot
```

---

## Install

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[pi]"          # tested against 0.6.2 / main
cd .. && git clone https://github.com/jianmanlincjx/Pi05.git
cd Pi05 && pip install -e .                   # registers policy.type=pi05_goal_prior
```

The goal-prior package imports lerobot's `pi05` and reuses its backbone, so the two must
match — pin lerobot to the commit you trained with.

lerobot does not import third-party policy packages on its own, so run training and
evaluation through `pi05gp-train` / `pi05gp-eval` (installed by `pip install -e .`), or
`import pi05_goal_prior` before calling lerobot's own entry points. The baseline is plain
`pi05` and needs neither.

---

## Weights

### 1. PaliGemma (required)

The VLM the policy is initialised from.

| | |
| --- | --- |
| repo | [`google/paligemma-3b-pt-224`](https://huggingface.co/google/paligemma-3b-pt-224) |
| access | **gated** — accept the licence on the model page while signed in |
| size | 11 GB, three safetensors shards plus an index |
| files needed | `model-0000{1,2,3}-of-00003.safetensors`, `model.safetensors.index.json`, `config.json` |

```bash
huggingface-cli login
huggingface-cli download google/paligemma-3b-pt-224 \
    --local-dir ./checkpoints/paligemma-3b-pt-224
```

Behind the Great Firewall, `HF_ENDPOINT=https://hf-mirror.com` works for public repos but
returns 403 for gated ones. Community re-uploads exist and are byte-size identical to Google's
listing, but the mirror redacts the official sha256 for gated repos, so **their provenance
cannot be checked from the mirror alone**. Accepting the licence on Hugging Face unmasks the
official hashes, and a downloaded copy can be verified against them afterwards without
re-downloading. Prefer the official repo.

`pi05_base` is deliberately **not** used. It is a fully post-trained VLA, and loading it would
hand the action expert exactly the prior Stage 1 exists to build — the comparison would then
measure the donated prior rather than the method. Both the baseline and the two stages start
from PaliGemma with a random action expert, which is also what the MolmoAct2 and FastWAM
variants of this experiment do.

### 2. The starting checkpoint (built locally)

```bash
python tools/build_init.py \
    --paligemma  ./checkpoints/paligemma-3b-pt-224 \
    --out        ./checkpoints/pi05_init \
    --state-dim  14 \
    --action-dim 14 \
    --cameras    2

python tools/verify_init.py \
    --init       ./checkpoints/pi05_init \
    --paligemma  ./checkpoints/paligemma-3b-pt-224
```

`build_init.py` fills only the `paligemma` submodule and leaves the action expert, the action
projections and the time MLP at random initialisation. It remaps keys from the transformers
4.x layout on disk to the 5.x layout lerobot builds, and shape-checks every tensor before
writing. `verify_init.py` then compares the result against the source value for value, rather
than merely checking that something changed.

The result is fp32 and about **16 GB**. Training reads it once and runs in bf16.

Layout written:

```
checkpoints/pi05_init/
├── config.json                    input/output feature shapes, chunk size, empty_cameras
├── model.safetensors              16 GB, fp32
├── policy_preprocessor.json       normalisation + tokenizer pipeline
├── policy_postprocessor.json
├── tokenizer/
└── train_config.json
```

### 3. Trained checkpoints

Not distributed here. A finished run writes to `--output_dir`, one directory per
`--save_freq` steps:

```
outputs/<job>/checkpoints/030000/pretrained_model/     ~9.4 GB in bf16
```

Point `--policy.path` at that `pretrained_model` directory for evaluation and for
`scripts/infer_realrobot.py`.

---

## Adapting to your robot

Full detail in [`docs/adapting_to_a_new_robot.md`](docs/adapting_to_a_new_robot.md). The
short version — π0.5 does not care about the joint count, the control mode, the camera
resolution or the control rate. State and action are padded to 32 dimensions internally and
images are resized with padding to 224×224.

It does care about four things, and each fails silently when wrong:

| | requirement | what goes wrong otherwise |
| --- | --- | --- |
| state range | normalised into `[-1, 1]` | the state is discretised into 256 bins over `[-1, 1]` and written into the language prompt; out-of-range values saturate to bin 0 or 255 and the model is handed a constant |
| image range | raw `[0, 1]`, `VISUAL: IDENTITY` | the policy rescales to `[-1, 1]` itself; pre-normalising applies the transform twice |
| camera count | `empty_cameras = 3 − cameras` | π0.5 always runs three image slots; a wrong count feeds the model a constant frame it treats as real |
| instruction | the same phrasing as training | the task string goes into the prompt verbatim; a model trained on a fixed set of instructions does not generalise past them |

Dataset keys, per frame:

```
observation.images.<name>    one per camera, uint8 HWC or float CHW in [0, 1]
observation.state            proprioception, dimension ≤ 32
action                       commanded action, dimension ≤ 32
task                         the language instruction, as a string
```

Statistics must include `q01` and `q99`, since normalisation is `QUANTILES`.

---

## Training

```bash
REPO_ID=yourname/yam_pick_place \
DATA_ROOT=/data/yam_pick_place \
INIT=./checkpoints/pi05_init \
OUT=./outputs/pi05_baseline \
CHUNK=10 NAS=10 EMPTY_CAMERAS=1 NPROC=8 BATCH=16 \
bash scripts/train_baseline.sh
```

Roughly 26 h for 30k steps on 8×A800 at an effective batch of 128, about 50 GB per GPU with
gradient checkpointing on.

`scripts/train_stage1.sh` and `scripts/train_stage2.sh` run the two-stage method. Stage 2's
three masking switches are the load-bearing part of that recipe and are documented in the
script itself.

---

## Inference on the robot

`scripts/infer_realrobot.py` is the loop; plug your driver into `connect_to_robot()`.

Two things that bite:

- **Call `policy.reset()` once per episode.** The policy holds an action queue and refills it
  every `n_action_steps`. Without a reset, a new episode begins by replaying the tail of the
  previous one.
- **The instruction string must match training.** It reaches the prompt verbatim after
  `strip()` and `_`→space.

---

## Repository layout

```
src/pi05_goal_prior/    the goal-prior policy (not needed for the baseline)
scripts/                training and inference entry points
tools/                  build and verify the starting checkpoint
patches/                optional trainer guard
docs/                   notes worth reading before trusting a result
```

- [`docs/adapting_to_a_new_robot.md`](docs/adapting_to_a_new_robot.md) — what has to change
  for a different embodiment, and the sanity checks to run before a long job.
- [`docs/libero_plus_language_bug.md`](docs/libero_plus_language_bug.md) — LIBERO-Plus derives
  the language instruction from the task file name, so the perturbation parameters reach the
  policy as words. Only relevant if you evaluate on that benchmark, where it moves the numbers
  a long way.

`patches/lerobot_train.diff` adds a gradient-spike guard that skips an update when the
pre-clip norm is non-finite or exceeds `GP_SKIP_GRAD_ABOVE` (default `1e5`). It never fired in
any run reported here; apply it only if you hit NaN losses.
