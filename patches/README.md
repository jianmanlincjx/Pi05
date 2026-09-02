# lerobot_train.py — gradient-spike guard (optional)

Adds one guard around the optimizer step: if the pre-clip gradient norm is non-finite, or
exceeds `GP_SKIP_GRAD_ABOVE` (default `1e5`), the update is skipped and the gradients are
zeroed. `GP_GRAD_PROBE="lo-hi"` logs per-module norms over that window of sync steps, which
is how a spike gets attributed — the seed is fixed, so it lands on the same step every run.

The norm is read **before** clipping on purpose: the clipper rescales every gradient (by zero
when the norm is `inf`) and destroys the evidence.

**It never fired in any run reported here** — baseline or method, on LIBERO. Apply it only if
you hit NaN losses on your own data; the results do not depend on it.

This is written as an edit rather than a `.diff` because the surrounding lines move between
lerobot revisions, and a patch that fails to apply is worse than one you paste in by hand.
Both edits go in `src/lerobot/scripts/lerobot_train.py`; it needs `import math` and
`import os`, which that file already has.

## 1. Module level, above `update_policy`

```python
# --- gradient-spike guard ---------------------------------------------------------------
# GP_SKIP_GRAD_ABOVE : skip the update when the pre-clip norm exceeds this (non-finite is
#                      always skipped). GP_GRAD_PROBE="lo-hi" logs per-module norms for
#                      that window of sync steps -- the seed is fixed, so the spike lands
#                      on the same step every run and can be attributed exactly.
_GP_SKIP_ABOVE = float(os.environ.get("GP_SKIP_GRAD_ABOVE", "1e5"))
_GP_PROBE = os.environ.get("GP_GRAD_PROBE", "")
_gp_sync_steps = 0
_gp_skipped = 0


def _gp_in_probe(step: int) -> bool:
    if "-" not in _GP_PROBE:
        return False
    lo, _, hi = _GP_PROBE.partition("-")
    try:
        return int(lo) <= step <= int(hi)
    except ValueError:
        return False


def _gp_module_grad_norms(policy, step: int, why: str) -> None:
    """Per-submodule gradient norms, largest first. Walks 4.3B parameters, so this runs
    only inside the probe window or on an actual spike."""
    acc: dict[str, float] = {}
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        g = p.grad.detach().float()
        n = float("inf") if not torch.isfinite(g).all() else float(torch.linalg.vector_norm(g))
        key = ".".join(name.split(".")[:4])
        acc[key] = max(acc.get(key, 0.0), n)
    top = sorted(acc.items(), key=lambda kv: -kv[1])[:12]
    logging.warning(
        "[grad-probe step=%d %s] %s", step, why,
        "  ".join(f"{k}={v:.3e}" for k, v in top),
    )
```

## 2. Inside `update_policy`, around the clip and the optimizer step

Find the block that clips and steps. Stock lerobot reads roughly:

```python
        grad_norm = None
        if accelerator.sync_gradients and grad_clip_norm > 0:
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)

        with lock if lock is not None else nullcontext():
            optimizer.step()
        optimizer.zero_grad()

        if lr_scheduler is not None:
            lr_scheduler.step()
```

Replace it with:

```python
        grad_norm = None
        gp_skip = False
        if accelerator.sync_gradients and grad_clip_norm > 0:
            global _gp_sync_steps, _gp_skipped
            _gp_sync_steps += 1
            # Attribute the norm BEFORE clipping: the clipper rescales every gradient (by
            # zero when the norm is inf) and destroys the evidence.
            if _gp_in_probe(_gp_sync_steps):
                _gp_module_grad_norms(policy, _gp_sync_steps, "probe")
            grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            gn = float(grad_norm)
            if not math.isfinite(gn) or gn > _GP_SKIP_ABOVE:
                gp_skip = True
                _gp_skipped += 1
                logging.warning(
                    "grad spike at sync step %d: norm=%s -> update SKIPPED (%d skipped so far)",
                    _gp_sync_steps, gn, _gp_skipped,
                )
                if not _gp_in_probe(_gp_sync_steps):
                    _gp_module_grad_norms(policy, _gp_sync_steps, "spike")
                optimizer.zero_grad(set_to_none=True)

        with lock if lock is not None else nullcontext():
            if not gp_skip:
                optimizer.step()
        optimizer.zero_grad()

        # The scheduler steps on a skipped update too: the LR trajectory has to stay
        # identical to the baseline's, or a step-matched comparison stops being one.
        if lr_scheduler is not None:
            lr_scheduler.step()
```
