# LIBERO-Plus feeds the perturbation parameters to the policy as language

Only relevant if you evaluate on LIBERO-Plus. It changes the numbers substantially.

## What happens

`libero/libero/benchmark/__init__.py` derives a task's language instruction from its file
name:

```python
def grab_language_from_filename(suite_name, x):
    if "_language_" not in x:
        language = " ".join(x.split("_"))          # the file name IS the instruction
        return language[:language.find(".bddl")]
    else:
        ...read language_instruction from the BDDL   # only this axis is correct
```

In the original LIBERO the file name *is* the task description, so this is fine. LIBERO-Plus
encodes each perturbation into the file name, and that line was not updated. The policy then
receives:

```
training:  pick up the black bowl between the plate and the ramekin and place it on the plate
camera:    ... and place it on the plate view 0 0 100 2 352 initstate 0
noise:     ... and place it on the plate view 0 0 100 0 0 initstate 0 noise 27
```

The Language Instructions axis is the only one that reads the BDDL, and it is the only one
that is correct.

## Why it stays hidden

The axes whose suffix is longest are the ones that collapse, which reads as "this perturbation
is harder" — exactly what the benchmark is meant to show. Clean LIBERO is unaffected, so the
usual sanity check passes. Models with broad language pretraining tolerate the junk tokens;
only a policy trained on a small fixed instruction set is destroyed by it. Ours was: on three
axes it scored 0/20 with the bug and 16/20, 17/20 and 4/4 with the instruction repaired,
nothing else changed.

## The fix

Resolve a Plus file name back to the original task it was derived from, then let the untouched
upstream logic run on that. A hand-written list of perturbation markers does not work —
`_table_` also occurs inside the real task `pick_up_the_black_bowl_from_table_center_...`, and
the actual markers (`_tb_N`, `_level<N>_sample<M>`, `_moved_`) are not guessable. Matching the
longest `.bddl` file that prefixes the name does not work either, because the texture, light
and layout perturbations ship their own `.bddl`.

What does work: a suite's original tasks are the `.bddl` basenames that do not extend another
`.bddl` basename — every perturbed file is an original plus a suffix, so the originals are the
minimal elements under the prefix order. Resolve to the longest original the name extends.

Audited over all four suites and all seven axes, 10,030 tasks: every repaired instruction
appears verbatim in the training instruction set, and the Language axis still comes from the
BDDL. Gate it behind an environment variable so the pre-fix numbers stay reproducible.
