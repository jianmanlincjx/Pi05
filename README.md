# Pi05 — π0.5 baseline and goal-prior on LeRobot

Training code for a π0.5 baseline and a two-stage goal-prior variant, built as a small
package on top of upstream [LeRobot](https://github.com/huggingface/lerobot). Nothing in
`lerobot/policies/pi05/` is patched, so **the baseline is stock π0.5** — this repo only adds
a new policy type and one optional guard in the trainer.

---

## 快速开始（真机 baseline）

只跑 baseline 的话不需要本仓库的模型代码，装好上游 lerobot 就够了；这里提供的是**初始化
构造**、**启动脚本**和**接入自己数据集的适配步骤**。四步：

1. 装 lerobot（见 Install）
2. `tools/build_init.py` 造起点权重：PaliGemma VLM + 随机初始化的动作专家
3. 把自己的数据转成 LeRobotDataset（见 Adapting to your robot）
4. `scripts/train_baseline.sh` 起训练，`scripts/infer_realrobot.py` 上机

关键一点：**起点不是 `pi05_base`**。`pi05_base` 是完整后训练过的 VLA，用它等于直接把动作
先验送给模型；本文的两阶段方法要论证的正是这个先验能不能自己学出来，所以两边都从
PaliGemma + 随机动作专家开始。

---

## Install

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && pip install -e ".[pi]"      # 0.6.2 / main
cd .. && git clone https://github.com/jianmanlincjx/Pi05.git
cd Pi05 && pip install -e .               # registers policy.type=pi05_goal_prior
```

The goal-prior package imports lerobot's `pi05` and reuses its backbone, so the two versions
must match. Pin lerobot to the commit you trained with.

Optional: `patches/lerobot_train.diff` adds a gradient-spike guard that skips an update when
the pre-clip norm is non-finite or exceeds `GP_SKIP_GRAD_ABOVE` (default `1e5`). It never
fired in any of our runs; apply it only if you see NaN losses.

---

## Building the starting checkpoint

`google/paligemma-3b-pt-224` is a gated repo — accept the licence on Hugging Face first.

```bash
python tools/build_init.py \
    --paligemma  /path/to/paligemma-3b-pt-224 \
    --out        /path/to/pi05_init \
    --state-dim  14 \
    --action-dim 14 \
    --cameras    2
python tools/verify_init.py --init /path/to/pi05_init --paligemma /path/to/paligemma-3b-pt-224
```

`build_init.py` fills only the `paligemma` submodule and leaves the action expert, the action
projections and the time MLP at random init. It remaps keys from the transformers 4.x layout
on disk to the 5.x layout lerobot builds, and shape-checks every tensor before writing.

The result is fp32 and about 16.5 GB. Training reads it once and runs in bf16.

---

## Adapting to your robot

### Dataset

A standard `LeRobotDataset` with, per frame:

| key | content |
| --- | --- |
| `observation.images.<name>` | one entry per camera, uint8 HWC or float CHW in `[0, 1]` |
| `observation.state` | proprioception, any dimension ≤ 32 |
| `action` | the commanded action, any dimension ≤ 32 |
| `task` | the language instruction, as a string |

π0.5 pads state and action to 32 internally, so a bimanual 14-DoF arm needs no code change —
only the right `shape` in `input_features` / `output_features`, which `build_init.py` writes
into `config.json` for you.

### Cameras

π0.5 always runs three image slots. Set `EMPTY_CAMERAS = 3 - (number of cameras)`; the unused
slots are filled with a constant `-1` frame and masked out of attention. Two cameras (one
scene, one wrist) is what we used.

Images are resized with padding to 224×224 and rescaled from `[0, 1]` to `[-1, 1]` inside the
policy. **Do not pre-normalise them** and do not set `VISUAL` to anything but `IDENTITY`.

### Normalisation

Keep `{"ACTION": "QUANTILES", "STATE": "QUANTILES", "VISUAL": "IDENTITY"}`. π0.5 uses
quantile normalisation for state and action, and it matters more than usual here: the policy
**writes the state into the language prompt**, discretised into 256 bins over `[-1, 1]`:

```
Task: pick up the red block, State: 89 109 235 236 135 151 245 8;\nAction:
```

so the state must actually land in `[-1, 1]`. Your dataset statistics need `q01` and `q99`;
`lerobot-dataset-stats` computes them.

### Chunk size and control rate

`chunk_size` is how many actions the model predicts, `n_action_steps` how many it executes
before replanning. We train and run with both at 10. If your control rate is much higher than
the demonstrations', predict a longer chunk and execute a prefix of it, but note that
training then spends most of its loss on actions that are never executed.

---

## Training

```bash
REPO_ID=yourname/yam_pick_place \
DATA_ROOT=/data/yam_pick_place \
INIT=/path/to/pi05_init \
OUT=./outputs/pi05_baseline \
CHUNK=10 NAS=10 EMPTY_CAMERAS=1 NPROC=8 BATCH=16 \
bash scripts/train_baseline.sh
```

Roughly 26 h for 30k steps on 8×A800 at an effective batch of 128, ~50 GB per GPU with
gradient checkpointing on.

---

## Inference on the robot

`scripts/infer_realrobot.py` is the loop; plug your driver into `connect_to_robot()`.

Two things that bite:

- **`policy.reset()` once per episode.** The policy holds an action queue and refills it every
  `n_action_steps`. Without a reset the new episode starts by replaying the tail of the old one.
- **The instruction string must match training.** The task text goes into the prompt verbatim
  after `strip()` and `_`→space; a model trained on a fixed set of instructions has no
  robustness to rephrasing unless the dataset contained variety.

---

## Repository layout

```
src/pi05_goal_prior/    the goal-prior policy (not needed for the baseline)
scripts/                training and inference entry points
tools/                  build and verify the starting checkpoint
patches/                optional trainer guard
docs/                   notes worth reading before trusting an evaluation
```

## Notes

- `docs/adapting_to_a_new_robot.md` — what actually has to change for a different embodiment,
  and the four things that fail silently if they are wrong.
- `docs/libero_plus_language_bug.md` — LIBERO-Plus derives the language instruction from the
  task file name, so the perturbation parameters reach the policy as words. Only relevant if
  you evaluate on that benchmark, but it changes the numbers a lot when it applies.
