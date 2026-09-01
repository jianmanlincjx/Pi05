# The 604 vs 603 gap is a transformers v4 -> v5 layout rename, not an architecture change.
# Derive the remap, then prove it: every policy tensor must be covered by a checkpoint tensor
# of identical shape, or be explicitly accounted for.
import json, pathlib, collections, re, sys
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.configs.types import FeatureType, PolicyFeature
from safetensors import safe_open

cfg = PI05Config(
    input_features={
        "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3,224,224)),
        "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3,224,224)),
        "observation.state":         PolicyFeature(type=FeatureType.STATE,  shape=(8,)),
    },
    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
)
vlm = PI05Policy(cfg).model.paligemma_with_expert.paligemma
sd = vlm.state_dict()

BASE = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "./paligemma-3b-pt-224")
idx = json.load(open(BASE/"model.safetensors.index.json"))
wm = idx["weight_map"]

def ckpt_name(k: str) -> str:
    """policy key -> checkpoint key (transformers 5.x layout -> the 4.x layout on disk)"""
    if k.startswith("model.language_model."):
        return "language_model.model." + k[len("model.language_model."):]
    if k.startswith("model.vision_tower."):
        return k[len("model."):]
    if k.startswith("model.multi_modal_projector."):
        return k[len("model."):]
    return k

handles = {}
def shape_of(name):
    f = wm[name]
    handles.setdefault(f, safe_open(BASE/f, framework="pt"))
    return tuple(handles[f].get_slice(name).get_shape())

hit, miss, mismatch = [], [], []
for k, t in sd.items():
    c = ckpt_name(k)
    if c not in wm: miss.append((k, c)); continue
    (hit if shape_of(c) == tuple(t.shape) else mismatch).append((k, c))

print(f"  policy tensors      : {len(sd)}")
print(f"  mapped + shape OK   : {len(hit)}")
print(f"  mapped, shape wrong : {len(mismatch)}")
print(f"  unmapped            : {len(miss)}")
if mismatch:
    print("\n  shape mismatches:")
    for k, c in mismatch[:8]: print(f"    {k}\n      -> {c}   ckpt {shape_of(c)}  policy {tuple(sd[k].shape)}")
if miss:
    print("\n  unmapped policy tensors:")
    for k, c in miss[:8]: print(f"    {k}   (tried {c})")

used = {ckpt_name(k) for k in sd}
unused = set(wm) - used
print(f"\n  checkpoint tensors left unused: {len(unused)}")
for k in sorted(unused)[:8]: print(f"    {k}  {shape_of(k)}")
