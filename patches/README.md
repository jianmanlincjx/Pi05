# lerobot_train.py — gradient-spike guard (optional)

Adds one guard around the optimizer step: if the pre-clip gradient norm is non-finite, or
exceeds `GP_SKIP_GRAD_ABOVE` (default `1e5`), the update is skipped and the gradients are
zeroed. `GP_GRAD_PROBE="lo-hi"` logs per-module norms over that window of sync steps, which
is how a spike gets attributed — the seed is fixed, so it lands on the same step every run.

The norm is read **before** clipping on purpose: the clipper rescales every gradient (by zero
when the norm is `inf`) and destroys the evidence.

It never fired in any run reported here — baseline or method, on LIBERO. Apply it only if you
hit NaN losses on your own data; the results do not depend on it.

Apply with:

    cd /path/to/lerobot && git apply /path/to/Pi05/patches/lerobot_train.diff
