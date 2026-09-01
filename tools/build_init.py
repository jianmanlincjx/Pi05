#!/usr/bin/env python
"""Build the training starting point: PaliGemma VLM weights + a randomly initialised action expert.

This is the initialisation both the baseline and the two-stage method start from. It is NOT
pi05_base: that checkpoint is a fully post-trained VLA, and loading it would hand the action
expert exactly the prior Stage 1 exists to build, so the comparison would measure the
donated prior rather than the method.

Only the `paligemma` submodule is filled. The action expert (`gemma_expert`), the action
projections and the time MLP stay at their random initialisation.

The key remap exists because the released PaliGemma checkpoint uses the transformers 4.x
layout while lerobot builds the 5.x one; the two differ by a prefix rename only, and every
tensor is shape-checked before it is written.

    python tools/build_init.py \
        --paligemma /path/to/paligemma-3b-pt-224 \
        --out       /path/to/pi05_paligemma_init \
        --state-dim 14 --action-dim 14 --cameras 2
"""
import argparse, json, pathlib
import torch
from safetensors import safe_open

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def ckpt_name(k: str) -> str:
    """policy key -> checkpoint key (transformers 5.x layout -> the 4.x layout on disk)."""
    if k.startswith("model.language_model."):
        return "language_model.model." + k[len("model.language_model."):]
    if k.startswith(("model.vision_tower.", "model.multi_modal_projector.")):
        return k[len("model."):]
    return k


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--paligemma", required=True, help="google/paligemma-3b-pt-224 checkout")
    p.add_argument("--out", required=True)
    p.add_argument("--state-dim", type=int, default=8)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--cameras", type=int, default=2, help="how many real cameras the robot has")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cpu",
                   help="where to build the fp32 model; cpu keeps a GPU free and needs ~20 GB RAM")
    a = p.parse_args()

    feats = {
        f"observation.images.image{'' if i == 0 else i + 1}":
            PolicyFeature(type=FeatureType.VISUAL, shape=(3, a.image_size, a.image_size))
        for i in range(a.cameras)
    }
    feats["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(a.state_dim,))
    cfg = PI05Config(
        input_features=feats,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(a.action_dim,))},
        # pi05 always runs three image slots; the unused ones are fed a constant -1 frame.
        empty_cameras=max(0, 3 - a.cameras),
        device=a.device,
    )
    policy = PI05Policy(cfg)
    vlm = policy.model.paligemma_with_expert.paligemma
    target = vlm.state_dict()

    base = pathlib.Path(a.paligemma)
    weight_map = json.load(open(base / "model.safetensors.index.json"))["weight_map"]
    handles: dict[str, object] = {}

    def source(name: str) -> torch.Tensor:
        f = weight_map[name]
        handles.setdefault(f, safe_open(base / f, framework="pt"))
        return handles[f].get_tensor(name)

    loaded, skipped, missing, truncated = 0, [], [], []
    new_state = {}
    for k, t in target.items():
        c = ckpt_name(k)
        if c not in weight_map:
            # lm_head is tied to the embedding in this checkpoint and is not used here.
            (skipped if c == "lm_head.weight" else missing).append(k)
            continue
        s = source(c)
        if tuple(s.shape) != tuple(t.shape):
            # The token embedding is the one legitimate size difference, and it goes the way
            # round that is easy to get backwards: the released checkpoint pads the vocabulary
            # to a multiple of 64 (257216 rows) while lerobot builds 257152, so the extra rows
            # are padding and the first 257152 line up one for one. Truncate, never pad.
            same_tail = s.ndim == t.ndim and s.shape[1:] == t.shape[1:]
            if same_tail and s.shape[0] >= t.shape[0]:
                new_state[k] = s[: t.shape[0]].to(t.dtype)
                truncated.append((k, s.shape[0], t.shape[0]))
                loaded += 1
                continue
            raise SystemExit(f"shape mismatch {k}: policy {tuple(t.shape)} vs ckpt {tuple(s.shape)}")
        new_state[k] = s.to(t.dtype)
        loaded += 1

    if missing:
        raise SystemExit(f"{len(missing)} policy tensors have no source, e.g. {missing[:3]}")

    # lm_head is tied to the token embedding in the released checkpoint and so has no entry of
    # its own. pi05 never runs it -- actions come out of action_out_proj -- but leaving it at
    # random init writes half a gigabyte of noise into the file and makes a checkpoint that
    # cannot be compared against the source. Tie it here as the source does.
    emb_key = "model.language_model.embed_tokens.weight"
    if "lm_head.weight" in target and emb_key in new_state:
        new_state["lm_head.weight"] = new_state[emb_key].clone()
        loaded += 1
        skipped.remove("lm_head.weight") if "lm_head.weight" in skipped else None

    vlm.load_state_dict(new_state, strict=False)
    print(f"VLM tensors loaded {loaded}, intentionally skipped {len(skipped)} ({skipped})")
    for k, had, want in truncated:
        print(f"  truncated {k}: {had} -> {want} rows (released vocab is padded to a multiple of 64)")

    ae = sum(p.numel() for n, p in policy.named_parameters() if ".paligemma." not in n)
    print(f"left at random init: {ae/1e6:.1f}M parameters (action expert, projections, time MLP)")

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(out)
    print(f"written to {out}")
    print("verify with:  python tools/verify_init.py --init <out> --paligemma <paligemma>")


if __name__ == "__main__":
    main()
