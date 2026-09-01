# Adapting to a new robot

Everything below is what actually has to change to train π0.5 on a different embodiment and
dataset. Written against a bimanual YAM setup, but nothing is specific to it.

## 1. What π0.5 expects, and what it does not care about

**Does not care:** the number of joints, the control mode, the camera resolution, the control
rate. State and action are padded to 32 dimensions internally, images are resized with padding
to 224×224, and the action chunk is whatever you configure.

**Does care, and will fail silently if wrong:**

| thing | requirement | what goes wrong otherwise |
| --- | --- | --- |
| state range | normalised into `[-1, 1]` | the state is discretised into 256 bins over `[-1, 1]` and written into the prompt; out-of-range values saturate to bin 0 or 255 |
| image range | raw `[0, 1]`, `VISUAL: IDENTITY` | the policy rescales to `[-1, 1]` itself; pre-normalising doubles the transform |
| camera count | `empty_cameras = 3 − cameras` | a missing slot is not detected; the model reads a garbage frame |
| instruction | the same phrasing as training | the prompt is the task string verbatim; a model trained on N fixed strings does not generalise past them |

## 2. Dataset

A standard `LeRobotDataset`. Per frame:

```
observation.images.image     uint8 HWC, or float CHW in [0, 1]
observation.images.image2    second camera, same convention
observation.state            float, dimension D_s ≤ 32
action                       float, dimension D_a ≤ 32
task                         str, the language instruction
```

For a bimanual arm, concatenate both arms into one vector rather than adding a third feature:
`observation.state = [left joints, left gripper, right joints, right gripper]`, and the same
ordering for `action`. Keep that ordering fixed forever — it is baked into the normalisation
statistics and into every checkpoint.

Statistics must include `q01` and `q99`, since normalisation is `QUANTILES`.

## 3. Config

`tools/build_init.py` writes these into `config.json`:

```
--state-dim   D_s
--action-dim  D_a
--cameras     number of real cameras
```

Check afterwards that `config.json` has the right `input_features` / `output_features` shapes
and `empty_cameras`. A mismatch between the checkpoint's declared shapes and the dataset's
actual shapes surfaces late and confusingly.

## 4. Chunk size and control rate

`chunk_size` actions are predicted, `n_action_steps` of them executed before replanning. We
use 10 and 10 at 10 Hz demonstrations.

Two failure modes worth knowing:

- **Predicting far more than you execute** wastes the loss. With `chunk_size=50,
  n_action_steps=10`, four fifths of the action loss sits on steps that never run. Measured on
  LIBERO, moving to 10/10 improved every suite except the long-horizon one.
- **Executing the whole chunk** means the last action of every chunk is the one the model
  predicts worst, because nothing follows it to smooth it. Measured at the end of Stage 1: the
  interior steps beat a chunk-50 model by 9%, and the boundary step lost to it by 57%. If that
  matters for your task, predict a slightly longer chunk than you execute.

## 5. Bimanual specifics

Nothing in the architecture is single-arm. The things to get right:

- One flat state/action vector, fixed ordering, both arms concatenated.
- If the two arms have different gripper conventions (position vs. binary), normalise them
  consistently — quantile normalisation on a near-binary channel gives a two-valued output,
  which is fine, but mixing a `[0, 1]` gripper with a `[-1, 1]` one is not.
- The goal pose used by Stage 1 is `observation.state` at `t + chunk_size`, i.e. the whole
  state vector, not an end-effector pose. For a bimanual robot that means both arms' targets,
  which is what you want.

## 6. Sanity checks before a long run

```bash
# the prompt the policy will actually receive, including the discretised state
python - <<'PY'
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import OBS_LANGUAGE_TOKENS
from transformers import AutoTokenizer
import torch
CK = "/path/to/init"
cfg = PreTrainedConfig.from_pretrained(CK); cfg.pretrained_path = CK
pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=CK)
ds = LeRobotDataset("yourname/yourdata", root="/data/yourdata")
row = ds[0]
batch = {k: v[None] for k, v in row.items() if isinstance(v, torch.Tensor)}
batch["task"] = [row["task"]]
out = pre(batch)
tok = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
ids = out[OBS_LANGUAGE_TOKENS][0]
print(tok.decode(ids[ids != 0]))
PY
```

Read the printed `State: ...` numbers. If they are pinned at `0` or `255`, the normalisation
range is wrong and the model is being handed a constant.

Then run 200 steps and confirm the loss falls below the value it starts at. π0.5 from this
initialisation starts around 1.5 and should be under 0.5 within a few hundred steps at
`lr=1e-4` — if it is flat, the data pipeline is wrong, not the model.
