#!/usr/bin/env python
"""Minimal real-robot inference loop for a trained pi05 checkpoint.

The policy owns an action queue: select_action returns one action per call and refills the
queue by running the flow-matching sampler every n_action_steps. Call reset() at the start of
every episode -- without it the queue carries actions from the previous episode into the new
one, which looks like a policy that begins each episode by replaying the end of the last.

Observation keys must match the ones the checkpoint was trained with; read them off
config.json rather than guessing. Images are uint8 HWC or float CHW in [0, 1]; pi05 resizes
with padding to 224 and rescales to [-1, 1] itself, so do not pre-normalise them.
"""
import argparse, time
import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--task", required=True, help="language instruction, exactly as in training")
    p.add_argument("--hz", type=float, default=10.0)
    p.add_argument("--max-steps", type=int, default=600)
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    cfg.pretrained_path = a.ckpt
    cfg.device = device
    policy = PI05Policy.from_pretrained(a.ckpt, config=cfg).to(device).eval()
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=a.ckpt,
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    print("image keys:", [k for k in cfg.input_features if k.startswith("observation.images")])
    print("state dim :", cfg.input_features["observation.state"].shape)
    print("action dim:", cfg.output_features["action"].shape)
    print("chunk", cfg.chunk_size, "| executed per replan", cfg.n_action_steps)

    robot = connect_to_robot()          # <-- your driver
    policy.reset()                      # once per episode, not once per process

    period = 1.0 / a.hz
    for _ in range(a.max_steps):
        t0 = time.perf_counter()
        obs = robot.get_observation()   # dict of np arrays, keys as printed above
        batch = {k: torch.as_tensor(np.asarray(v))[None] for k, v in obs.items()}
        batch["task"] = [a.task]
        with torch.inference_mode():
            action = policy.select_action(pre(batch))
        action = post(action)[0].cpu().numpy()
        robot.send_action(action)
        if (rest := period - (time.perf_counter() - t0)) > 0:
            time.sleep(rest)


def connect_to_robot():
    raise NotImplementedError("plug in your YAM driver here")


if __name__ == "__main__":
    main()
