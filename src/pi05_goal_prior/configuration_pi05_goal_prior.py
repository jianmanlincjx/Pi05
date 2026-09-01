#!/usr/bin/env python
"""Configuration for the two-stage goal-pose prior on pi05.

One policy class covers both stages; `goal_prior_stage` selects which one. Keeping them in a
single class means the Stage 1 -> Stage 2 handoff is a checkpoint load plus a config change
rather than a second code path, and the handoff is where silent mistakes hide.
"""

from dataclasses import dataclass

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config


@PreTrainedConfig.register_subclass("pi05_goal_prior")
@dataclass
class PI05GoalPriorConfig(PI05Config):
    # ---- which stage ------------------------------------------------------------------
    # "off"    : behaves exactly like stock pi05 (used to prove the forward copy is faithful)
    # "stage1" : VLM frozen and vision dropped; the oracle goal pose is injected as 8 tokens
    # "stage2" : vision restored, everything trains; the 8 pose tokens are inferred by the
    #            aggregator instead of read from the oracle
    goal_prior_stage: str = "off"

    # ---- the target ------------------------------------------------------------------
    # The goal is ONE pose: the end of the action chunk. None means "= chunk_size", which is
    # the rule the earlier backbones followed (MolmoAct2 ran chunk 10 with delta index 10).
    target_pose_delta_index: int | None = None

    # ---- Stage 1 ---------------------------------------------------------------------
    num_goal_tokens: int = 8
    goal_hidden_dim: int = 512
    # Stage 1 is vision-free: the image towers are skipped, not merely frozen.
    stage1_drop_images: bool = True

    # ---- Stage 2 aggregator (values measured on MolmoAct2 v3) --------------------------
    num_semantic_visual_tokens: int = 100
    num_semantic_visual_pose_tokens: int = 8
    semantic_visual_hidden_dim: int = 768
    semantic_visual_num_heads: int = 8
    semantic_visual_ffn_ratio: float = 4.0
    semantic_visual_dropout: float = 0.0
    semantic_visual_enable_self_attention: bool = True
    semantic_visual_num_layer_groups: int = 6
    pose_recon_loss_weight: float = 0.3

    # ---- Stage 2 learning rates ------------------------------------------------------
    # The VLM is frozen through Stage 1 and then fully trained in Stage 2 with a freshly
    # initialised 157M aggregator hanging off it. At one shared LR the backbone dominates
    # the gradient and the run dies with grdn:inf. None keeps the single-LR behaviour.
    vlm_optimizer_lr: float | None = None

    # The aggregator is 157M parameters at random init hanging off a pretrained backbone.
    # MolmoAct2 gives it its own group an order of magnitude above everything else
    # (optimizer_lr 1e-5 for backbone AND action expert, optimizer_semantic_visual_lr 1e-4).
    # Matching that ratio here rather than the absolute value keeps the backbone at the
    # baseline's LR -- which is the comparison the baseline exists for -- while preserving
    # what the ratio actually buys: a new module that moves faster than the weights it
    # hangs off. Covers the aggregator, the pose norm and the pose decoder, exactly as
    # MolmoAct2's "semantic_visual_" name filter does. None keeps the single-LR behaviour.
    aggregator_optimizer_lr: float | None = None

    # ---- Stage 2 gate ----------------------------------------------------------------
    # Stored as init + learnable delta and summed in fp32. dtype defaults to bfloat16 here,
    # where the grid spacing at -2.0 is 0.0078 while AdamW moves ~1e-4 per step: a plain
    # bf16 parameter rounds every update straight back and the gate freezes for the whole
    # run behind a healthy-looking loss curve. This cost us a FastWAM run.
    syn_gate_bias_init: float = -2.0
    # The gate only makes sense when the action expert also has a direct visual
    # path to trade against. With mask_image_from_action_expert it does not, and the
    # bias just handicaps the only channel carrying vision. MolmoAct2 has no gate.
    use_syn_gate: bool = True
    # Rescale the injected latent keys to the backbone key RMS of the layer they enter.
    # Without it the aggregator's to_key is unconstrained relative to the backbone's k_proj,
    # and the cheapest way for the optimizer to ignore the latents is to inflate their norm:
    # measured at 38x by layer 17, with 0.09% of the action rows' attention reaching them.
    # MolmoAct2 has no such freedom -- its latent K/V share context_k_proj with the backbone.
    normalize_latent_keys: bool = False
    gate_pose_tokens: bool = False

    # ---- Stage 2 channel regime (effective ratios, not nominal) ------------------------
    # both 0.50 / ref_only 0.20 / syn_only 0.30. Sampling uses fixed counts per batch, never
    # independent coins: ranks that disagree about which metric keys exist deadlock the
    # all-gather. Write effective values -- ImageWAM's nominal 0.55/0.15/0.30 lands here at
    # its batch of 10, and copying nominal numbers to another batch size changes the regime.
    # Stage 2, MolmoAct2 parity: hide the image columns from the action rows for
    # every sample so visual information reaches them only through the latents.
    # When true the sampled channel regime is skipped entirely.
    mask_image_from_action_expert: bool = False
    p_ref_only: float = 0.20
    p_syn_only: float = 0.30
    context_token_dropout: float = 0.05
    context_blackout_prob: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        if self.goal_prior_stage not in {"off", "stage1", "stage2"}:
            raise ValueError(
                f"goal_prior_stage must be one of off/stage1/stage2, got {self.goal_prior_stage}"
            )
        if self.target_pose_delta_index is None:
            object.__setattr__(self, "target_pose_delta_index", self.chunk_size)
        if not 1 <= self.target_pose_delta_index:
            raise ValueError(
                f"target_pose_delta_index must be >= 1, got {self.target_pose_delta_index}"
            )
        n_layers = self.action_expert_depth if hasattr(self, "action_expert_depth") else None
        if self.goal_prior_stage == "stage2" and self.num_semantic_visual_pose_tokens > self.num_semantic_visual_tokens:
            raise ValueError("pose tokens cannot exceed the total number of latents")

    @property
    def state_observation_delta_indices(self) -> list[int] | None:
        """Fetch the current state AND the goal pose in one sample.

        Out-of-range indices are clamped to the episode's last frame and flagged in
        `observation.state_is_pad`, which is the right semantics here: near the end of an
        episode the goal genuinely is the final pose.
        """
        if self.goal_prior_stage == "off":
            return super().state_observation_delta_indices
        # Goal FIRST, current LAST. The stock Pi05PrepareStateTokenizerProcessorStep writes
        # `state[:, -1]` into the prompt; with [current, goal] that would put the GOAL in the
        # prompt and hand the model the answer, with a loss curve that only looks excellent.
        # Ordering the indices this way makes the stock step correct as written, so there is
        # no processor to override and no way for the two to drift apart.
        return [int(self.target_pose_delta_index), 0]
