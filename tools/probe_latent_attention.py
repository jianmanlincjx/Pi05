"""How much attention the action rows actually give the latents.

Run this after Stage 2. It is the one check that separates a trained model that uses the
latent interface from one that quietly ignored it -- the loss curves do not, and the second
kind converges to a *lower* action loss because reading the language prompt directly is an
easier problem than reading a summary of the scene.

Measure on real observations from the training distribution. The same probe on random noise
proves nothing: SigLIP on noise produces features the aggregator cannot summarise, so an
action expert ignoring the latents there would be behaving correctly.

    python tools/probe_latent_attention.py --ckpt OUT/checkpoints/030000/pretrained_model \
                                           --repo-id yourname/yam_pick_place \
                                           --data /data/yam_pick_place

Healthy, on the reference LIBERO run: latents ~9%, images 0.000%, language 0.000%. Latents
near 0.1% with language in the tens of percent means a mask switch is off -- see the Stage 2
section of the README. Images or language nonzero at all means the mask is not being applied.
"""
import argparse
import os

import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import pi05_goal_prior  # noqa: F401  -- registers the policy type
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE
from pi05_goal_prior.modeling_pi05_goal_prior import PI05GoalPriorPolicy
from transformers.models.gemma import modeling_gemma


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="a Stage 2 pretrained_model directory")
    ap.add_argument("--repo-id", default="lerobot/libero")
    ap.add_argument("--data", required=True, help="dataset root")
    ap.add_argument("--samples", type=int, default=16)
    args = ap.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.ckpt)
    cfg.pretrained_path, cfg.device = args.ckpt, "cuda"
    policy = PI05GoalPriorPolicy.from_pretrained(args.ckpt, config=cfg).to("cuda").eval()

    n_latent = int(cfg.num_semantic_visual_tokens)
    chunk = int(cfg.chunk_size)
    print(
        f"stage={policy.model.stage}  latents={n_latent}  chunk={chunk}\n"
        f"mask_image={cfg.mask_image_from_action_expert}  "
        f"mask_language={getattr(cfg, 'mask_language_from_action_expert', 'ABSENT')}  "
        f"normalize_keys={getattr(cfg, 'normalize_latent_keys', 'ABSENT')}  "
        f"gate={cfg.use_syn_gate}"
    )
    if getattr(cfg, "mask_language_from_action_expert", "ABSENT") == "ABSENT":
        print("\n  !! This checkpoint predates the language mask. It is not the reported method.")

    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=args.ckpt)
    dt = {
        OBS_STATE: [i / 10.0 for i in cfg.state_observation_delta_indices],
        ACTION: [i / 10.0 for i in cfg.action_delta_indices],
    }
    ds = LeRobotDataset(args.repo_id, root=args.data, delta_timestamps=dt)
    rows = [ds[i * (len(ds) // args.samples)] for i in range(args.samples)]

    # The image block is the head of the prefix; language and state follow it; the action
    # rows are the tail of the whole sequence, and the latents are appended past its end.
    n_img = 768
    stats = []
    original = modeling_gemma.eager_attention_forward

    def probe(module, query, key, value, attention_mask, scaling, **kw):
        with torch.no_grad():
            k32, q32 = key.float(), query.float()
            if k32.shape[2] > n_latent:
                n_bb = k32.shape[2] - n_latent
                kk = k32.repeat_interleave(q32.shape[1] // k32.shape[1], dim=1)
                logits = torch.matmul(q32, kk.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    logits = logits + attention_mask[:, :, :, : logits.shape[-1]].float()
                p = logits.softmax(-1)
                act = slice(n_bb - chunk, n_bb)
                stats.append(
                    (
                        p[:, :, act, n_bb:].sum(-1).mean().item(),
                        p[:, :, act, :n_img].sum(-1).mean().item(),
                        p[:, :, act, n_img : n_bb - chunk].sum(-1).mean().item(),
                        p[:, :, act, n_bb - chunk : n_bb].sum(-1).mean().item(),
                    )
                )
        return original(module, query, key, value, attention_mask, scaling, **kw)

    modeling_gemma.eager_attention_forward = probe
    try:
        batch = {}
        for r in rows:
            for k, v in r.items():
                if isinstance(v, torch.Tensor):
                    batch.setdefault(k, []).append(v)
        batch = {k: torch.stack(v).to("cuda") for k, v in batch.items()}
        batch["task"] = [r["task"] for r in rows]
        policy.reset()
        with torch.no_grad():
            policy.select_action(pre(batch))
    finally:
        modeling_gemma.eager_attention_forward = original

    if not stats:
        raise SystemExit("no attention captured -- is this a Stage 2 checkpoint?")
    lat, img, lng, own = (sum(s[i] for s in stats) / len(stats) for i in range(4))
    print(f"\n{args.samples} real observations, {len(stats)} (layer, denoise step) pairs")
    print("  where the action rows look:")
    print(f"    {n_latent} latent columns      {100 * lat:7.3f}%")
    print(f"    {n_img} image columns        {100 * img:7.3f}%")
    print(f"    language + state columns  {100 * lng:7.3f}%")
    print(f"    {chunk} own action columns     {100 * own:7.3f}%")
    print(f"    total                     {100 * (lat + img + lng + own):7.3f}%")
    per = [s[0] for s in stats]
    print(f"\n  latent share across layers: min {100 * min(per):.4f}%  max {100 * max(per):.4f}%")
    if lat < 0.01:
        print("\n  !! Latents are being ignored. The method is not active in this checkpoint.")


if __name__ == "__main__":
    main()
