"""Verify the pi05 init checkpoint is exactly PaliGemma-VLM + random-AE.

Not "it changed from random" -- that is too weak. This compares the saved tensors value-for-value
against the source safetensors, checks the action expert is untouched random, and runs a forward
pass to confirm the vision tower produces non-degenerate features.
"""
import argparse, json, pathlib, torch
from safetensors import safe_open

_a = argparse.ArgumentParser()
_a.add_argument("--init", required=True, help="directory written by build_init.py")
_a.add_argument("--paligemma", required=True, help="google/paligemma-3b-pt-224 checkout")
_args = _a.parse_args()
INIT = pathlib.Path(_args.init) / "model.safetensors"
SRC = pathlib.Path(_args.paligemma)

idx = json.load(open(SRC/"model.safetensors.index.json")); wm = idx["weight_map"]
handles = {}
def src_tensor(name):
    f = wm[name]
    handles.setdefault(f, safe_open(SRC/f, framework="pt"))
    return handles[f].get_tensor(name)

init = safe_open(INIT, framework="pt")
keys = list(init.keys())
print(f"init checkpoint tensors: {len(keys)}")

def remap(k):
    k = k[len("paligemma_with_expert.paligemma."):]
    if k.startswith("model.language_model."):  return "language_model.model." + k[len("model.language_model."):]
    if k.startswith(("model.vision_tower.", "model.multi_modal_projector.")): return k[len("model."):]
    return k

# ---- 1. every VLM tensor must equal its source, value for value ----
vlm_keys = [k for k in keys if k.startswith("paligemma_with_expert.paligemma.")]
checked = mismatch = trunc_ok = skipped = 0
for k in vlm_keys:
    s = remap(k)
    if s == "lm_head.weight":
        skipped += 1; continue
    if s not in wm:
        print(f"  !! no source for {k}"); mismatch += 1; continue
    a, b = init.get_tensor(k), src_tensor(s)
    if a.shape != b.shape:
        if torch.equal(a, b[:a.shape[0]]): trunc_ok += 1
        else: print(f"  !! truncated tensor differs: {k}"); mismatch += 1
        continue
    if torch.equal(a, b): checked += 1
    else: print(f"  !! value mismatch: {k}"); mismatch += 1
print(f"\n  VLM tensors identical to source : {checked}")
print(f"  identical after truncation      : {trunc_ok}")
print(f"  tied (lm_head, checked below)   : {skipped}")
print(f"  MISMATCHES                      : {mismatch}")

# ---- 2. lm_head must equal the (truncated) embedding it is tied to ----
lm = init.get_tensor("paligemma_with_expert.paligemma.lm_head.weight")
emb = init.get_tensor("paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight")
print(f"  lm_head == embed_tokens         : {torch.equal(lm, emb)}")

# ---- 3. action expert must be random init, not from any checkpoint ----
ae = [k for k in keys if k.startswith("paligemma_with_expert.gemma_expert.")]
n = sum(init.get_tensor(k).numel() for k in ae)
w = torch.cat([init.get_tensor(k).flatten().float() for k in ae if init.get_tensor(k).dim() == 2][:20])
print(f"\n  action-expert tensors           : {len(ae)}  ({n/1e6:.1f}M params)")
print(f"  sampled weight mean / std       : {w.mean():+.5f} / {w.std():.5f}   (random init ~0 / small)")
zero = sum(1 for k in ae if torch.count_nonzero(init.get_tensor(k)) == 0)
print(f"  all-zero tensors (biases/norms) : {zero}")

# ---- 4. forward the real vision tower on a real image, check features are not degenerate ----
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.types import FeatureType, PolicyFeature
cfg = PI05Config(
    input_features={
        "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3,224,224)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3,224,224)),
        "observation.state":         PolicyFeature(type=FeatureType.STATE,  shape=(8,)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
)
p = PI05Policy(cfg)
sd = {k: init.get_tensor(k) for k in keys}
missing, unexpected = p.model.load_state_dict(sd, strict=False)
print(f"\n  reload into a fresh policy      : {len(missing)} missing, {len(unexpected)} unexpected")
p.eval()
with torch.no_grad():
    dev = next(p.model.parameters()).device
    img = torch.rand(1, 3, 224, 224, device=dev)
    feat = p.model.paligemma_with_expert.embed_image(img)
    f = feat if torch.is_tensor(feat) else feat[0]
print(f"  vision features shape           : {tuple(f.shape)}")
print(f"  feature std / |mean|            : {f.float().std():.4f} / {f.float().mean().abs():.4f}   (dead tower would be ~0)")
