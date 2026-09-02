#!/usr/bin/env python
"""Two-stage goal-pose prior on pi05.

The layer loop from pi05 is reproduced here so extra key/value entries can be appended per
layer -- the same injection point MolmoAct2 and FastWAM use, except that pi05 has no separate
cross-attention, so the latents ride the joint attention's K/V and are hidden from the prefix
rows by the mask.

Two things are easy to get wrong and neither raises:
  * extra K/V must be appended AFTER apply_rotary_pos_emb. The goal tokens do not sit at real
    sequence positions; giving them a rotary phase silently corrupts them.
  * the added mask columns must be visible to the suffix (action) rows only. If the prefix
    could see them, the VLM would read latents computed from its own hidden states.

With `goal_prior_stage="off"` this file is numerically indistinguishable from stock pi05
(verified step-for-step against two stock runs), which is what makes the copied loop safe.
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.models.gemma import modeling_gemma

from lerobot.policies.pi05.modeling_pi05 import (
    PI05Policy,
    PI05Pytorch,
    PaliGemmaWithExpertModel,
    _build_flow_matching_inputs,
    _gated_residual,
    _sample_training_rtc_prefix_mask,
    layernorm_forward,
    make_att_2d_masks,
    prepare_attention_masks_4d,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.import_utils import require_package

from .configuration_pi05_goal_prior import PI05GoalPriorConfig
from .goal_prior import (
    GoalPoseDecoder,
    SE3Encoder,
    SemanticVisualAggregator,
    sample_channel_regime,
)

NEG = -2.3819763e38  # the blocked value prepare_attention_masks_4d uses


def compute_layer_with_extra_kv(
    inputs_embeds, attention_mask, position_ids, adarms_cond, latents,
    *, layers, rotary_emb, kv_fn=None,
):
    """One joint attention layer, optionally attending to extra key/value entries.

    Mirrors `pi05.modeling_pi05.compute_layer_complete`. `kv_fn` produces the extra entries
    from this layer's prefix hidden state and the incoming latents, and must run here rather
    than outside so the layer stays a single checkpointable unit.

    Returns (outputs_embeds, latents_out).
    """
    query_states, key_states, value_states, gates = [], [], [], []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        query_states.append(layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        key_states.append(layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2))
        value_states.append(layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2))

    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0], query_states.shape[2], query_states.shape[-1],
        device=query_states.device, dtype=query_states.dtype,
    )
    cos, sin = rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    # --- injection: after RoPE, before attention ---------------------------------------
    # The hook runs here rather than at the top of the layer so it can see this layer's
    # backbone keys; the latents it consumes come from inputs_embeds, unchanged either way.
    extra_k = extra_v = extra_mask = None
    if kv_fn is not None:
        extra_k, extra_v, extra_mask, latents = kv_fn(inputs_embeds[0], latents, key_states)

    if extra_k is not None:
        key_states = torch.cat([key_states, extra_k.to(key_states.dtype)], dim=2)
        value_states = torch.cat([value_states, extra_v.to(value_states.dtype)], dim=2)
        attention_mask = torch.cat([attention_mask, extra_mask.to(attention_mask.dtype)], dim=-1)

    batch_size = query_states.shape[0]
    paligemma_layer = layers[0]
    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma_layer.self_attn, query_states, key_states, value_states,
        attention_mask, paligemma_layer.self_attn.scaling,
    )
    head_dim = paligemma_layer.self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = layers[i]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds, latents


class PaliGemmaWithExpertGoalPrior(PaliGemmaWithExpertModel):
    """Adds a per-layer hook to the joint forward. Holds no parameters and no state.

    `extra_kv_fn(layer_idx, prefix_hidden, num_layers) -> (k, v, mask) | None` is owned by the
    model, which is where the recurrent latent state lives; keeping this class stateless is
    what makes installing it by class reassignment safe.
    """

    extra_kv_fn = None
    initial_latents_fn = None
    final_latents = None

    def forward(self, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, adarms_cond=None):
        # Prefill / decode-only paths are untouched: no action rows are present, so there is
        # nothing for the goal tokens to condition.
        if inputs_embeds[1] is None or inputs_embeds[0] is None or self.extra_kv_fn is None:
            return super().forward(
                attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, adarms_cond=adarms_cond,
            )

        if adarms_cond is None:
            adarms_cond = [None, None]
        paligemma_layers = self.paligemma.model.language_model.layers
        gemma_expert_layers = self.gemma_expert.model.layers
        rotary_emb = self.paligemma.model.language_model.rotary_emb
        use_gradient_checkpointing = (
            hasattr(self.gemma_expert.model, "gradient_checkpointing")
            and self.gemma_expert.model.gradient_checkpointing and self.training
        ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

        num_layers = len(paligemma_layers)
        latents = self.initial_latents_fn() if self.initial_latents_fn is not None else None
        for layer_idx, layers in enumerate(zip(paligemma_layers, gemma_expert_layers, strict=True)):
            kv_fn = None if self.extra_kv_fn is None else (
                lambda h, lat, bk, _i=layer_idx: self.extra_kv_fn(_i, h, lat, num_layers, bk)
            )
            if use_gradient_checkpointing:
                inputs_embeds, latents = torch.utils.checkpoint.checkpoint(
                    compute_layer_with_extra_kv, inputs_embeds, attention_mask, position_ids,
                    adarms_cond, latents, use_reentrant=False, preserve_rng_state=False,
                    layers=layers, rotary_emb=rotary_emb, kv_fn=kv_fn,
                )
            else:
                inputs_embeds, latents = compute_layer_with_extra_kv(
                    inputs_embeds, attention_mask, position_ids, adarms_cond, latents,
                    layers=layers, rotary_emb=rotary_emb, kv_fn=kv_fn,
                )
        self.final_latents = latents

        final_norms = (self.paligemma.model.language_model.norm, self.gemma_expert.model.norm)

        def compute_final_norms(inputs_embeds, adarms_cond):
            return [
                layernorm_forward(final_norms[i], h, adarms_cond[i])[0]
                for i, h in enumerate(inputs_embeds)
            ]

        if use_gradient_checkpointing:
            outputs_embeds = torch.utils.checkpoint.checkpoint(
                compute_final_norms, inputs_embeds, adarms_cond,
                use_reentrant=False, preserve_rng_state=False,
            )
        else:
            outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)
        return [outputs_embeds[0], outputs_embeds[1]], None


class PI05GoalPriorPytorch(PI05Pytorch):
    """pi05 core model plus the goal-pose channel."""

    def __init__(self, config: PI05GoalPriorConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        # The wrapper subclass adds no state, so installing it by class reassignment avoids
        # duplicating PaliGemmaWithExpertModel's constructor and leaves pi05/ untouched.
        self.paligemma_with_expert.__class__ = PaliGemmaWithExpertGoalPrior
        # Always installed, even at stage "off": otherwise the copied layer loop is never
        # exercised and the equivalence test would pass without testing anything.
        self.paligemma_with_expert.extra_kv_fn = self._extra_kv
        self.paligemma_with_expert.initial_latents_fn = self._initial_latents

        self.stage = config.goal_prior_stage
        self.pose_dim = int(config.input_features[OBS_STATE].shape[0])
        latent_dim = int(config.semantic_visual_hidden_dim)
        attn = self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn
        self.head_dim = int(attn.head_dim)
        # Read the KV-head count off the projection itself. The previous form,
        # `int(getattr(attn, "num_key_value_groups", 1) and 1)`, is a boolean `and`
        # that evaluates to 1 for any truthy value -- right for PaliGemma (MQA, one
        # KV head) purely by coincidence, and silently wrong for any other backbone.
        self.kv_heads = int(attn.k_proj.out_features // self.head_dim)
        kv_dim = self.kv_heads * self.head_dim

        if self.stage == "stage1":
            self.se3_encoder = SE3Encoder(
                pose_dim=self.pose_dim, hidden_dim=int(config.goal_hidden_dim),
                num_tokens=int(config.num_goal_tokens), latent_dim=latent_dim,
            )
            self.goal_to_key = nn.Linear(latent_dim, kv_dim, bias=False)
            self.goal_to_value = nn.Linear(latent_dim, kv_dim, bias=False)
        elif self.stage == "stage2":
            self.aggregator = SemanticVisualAggregator(
                num_tokens=int(config.num_semantic_visual_tokens),
                latent_dim=latent_dim,
                context_dim=int(self.paligemma_with_expert.paligemma.config.text_config.hidden_size),
                kv_dim=kv_dim,
                num_heads=int(config.semantic_visual_num_heads),
                ffn_ratio=float(config.semantic_visual_ffn_ratio),
                dropout=float(config.semantic_visual_dropout),
                enable_self_attention=bool(config.semantic_visual_enable_self_attention),
                num_layer_groups=int(config.semantic_visual_num_layer_groups),
                num_pose_tokens=int(config.num_semantic_visual_pose_tokens),
                gate_bias_init=float(config.syn_gate_bias_init),
                gate_pose_tokens=bool(config.gate_pose_tokens),
                context_token_dropout=float(config.context_token_dropout),
                context_blackout_prob=float(config.context_blackout_prob),
            )
            self.pose_norm = nn.LayerNorm(latent_dim)
            self.pose_decoder = GoalPoseDecoder(
                num_tokens=int(config.num_semantic_visual_pose_tokens),
                hidden_size=latent_dim, pose_dim=self.pose_dim,
            )
        self._gp = None  # per-forward scratch, set by forward()

    # ---- Stage 1 is vision-free -------------------------------------------------------
    def embed_prefix(self, images, img_masks, tokens, masks, states=None, state_masks=None):
        if self.stage == "stage1" and bool(self.config.stage1_drop_images):
            # Skipped, not frozen: two towers at 256 tokens each dominate the step cost, and
            # the point of Stage 1 is that appearance is unavailable, not merely un-trained.
            images, img_masks = [], []
        return super().embed_prefix(images, img_masks, tokens, masks, states, state_masks)

    # ---- the per-layer hook -----------------------------------------------------------
    def _initial_latents(self):
        gp = self._gp
        return None if gp is None else gp.get("latents")

    def _shape_kv(self, t):
        b, n, _ = t.shape
        return t.view(b, n, self.kv_heads, self.head_dim).transpose(1, 2)

    def _rescale_keys(self, k, backbone_k):
        """Put the latent keys on the backbone's scale for this layer.

        Competing softmax logits are q.k, so a channel whose keys are free to grow can be
        silenced by norm alone. Matching the backbone RMS leaves only direction.
        """
        if backbone_k is None or not bool(getattr(self.config, "normalize_latent_keys", False)):
            return k
        tgt = backbone_k.detach().float().pow(2).mean().sqrt()
        cur = k.float().pow(2).mean().sqrt().clamp_min(1e-6)
        return (k.float() * (tgt / cur)).to(k.dtype)

    def _extra_kv(self, layer_idx, prefix_hidden, latents, num_layers, backbone_k=None):
        """Extra K/V for one layer. Runs inside the checkpointed unit, so it takes the
        latents in and hands them back rather than mutating shared state."""
        gp = self._gp
        if gp is None:
            return None, None, None, latents
        if self.stage == "stage1":
            return self._rescale_keys(gp["k"], backbone_k), gp["v"], gp["mask"], latents
        if self.stage != "stage2":
            return None, None, None, latents

        g = self.aggregator.layer_group_index(layer_idx, num_layers)
        latents = self.aggregator(
            latents, prefix_hidden.to(latents.dtype),
            semantic_mask=gp["semantic_mask"], image_mask=gp["image_mask"], group_idx=g,
        )
        k, v = self.aggregator.project_kv(latents, group_idx=g)

        # Gate: one learnable scalar per group, added to the float mask. The pose-supervised
        # span stays ungated so that at step 0 the action rows still see exactly Stage 1's
        # interface rather than a pose-conditioned prior with the pose removed.
        if bool(getattr(self.config, "use_syn_gate", True)):
            mask = gp["mask"].clone()
            lo, hi = self.aggregator.gated_span()
            allowed = mask[..., lo:hi] > (NEG / 2)
            mask[..., lo:hi] = torch.where(
                allowed, mask[..., lo:hi] + self.aggregator.gate_bias(g).to(mask.dtype), mask[..., lo:hi]
            )
        else:
            # no gate: every latent column competes on its own logits, as in MolmoAct2
            mask = gp["mask"]
        return self._rescale_keys(self._shape_kv(k), backbone_k), self._shape_kv(v), mask, latents

    # ---- forward ----------------------------------------------------------------------
    def forward(self, images, img_masks, tokens, masks, actions, noise, time,
                prefix_mask=None, states=None, state_masks=None, goal_pose=None):
        self._gp = None
        if self.stage == "off" or goal_pose is None:
            return super().forward(images, img_masks, tokens, masks, actions, noise, time,
                                   prefix_mask=prefix_mask, states=states, state_masks=state_masks)

        x_t, model_time = _build_flow_matching_inputs(actions, noise, time, prefix_mask)
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, model_time)
        dtype = self.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        if dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_4d = prepare_attention_masks_4d(att_2d)

        bsize = prefix_embs.shape[0]
        device = prefix_embs.device
        n_prefix = prefix_embs.shape[1]
        self._gp = self._build_goal_state(bsize, device, n_prefix, att_4d, goal_pose, masks,
                                          img_masks=img_masks)

        def forward_func(prefix_embs, suffix_embs, att_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs], use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, self._gp["att_4d"], position_ids, adarms_cond
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        v_t = self._apply_checkpoint(self.action_out_proj, suffix_out)

        pose_pred = None
        if self.stage == "stage2":
            final = self.paligemma_with_expert.final_latents
            pose_hidden = self.pose_norm(final[:, : self.aggregator.num_pose_tokens, :])
            pose_pred = self.pose_decoder(pose_hidden.to(torch.float32))
        # `_gp` is deliberately NOT cleared here. Backward runs after forward returns, and
        # gradient checkpointing re-executes each layer -- including the hook that reads
        # `_gp`. Clearing it made the recomputation inject no latents at all, so forward and
        # backward built different graphs ("a different number of tensors was saved"). It is
        # reset at the start of the next forward instead.
        return F.mse_loss(u_t, v_t, reduction="none"), pose_pred

    def sample_actions(self, images, img_masks, tokens, masks, states=None, state_masks=None,
                       noise=None, num_steps=None, **kwargs):
        """Joint-mode denoising, so inference builds the same graph training did.

        See module history: the cached sampler silently drops the goal latents because the
        goal-prior forward requires both halves of ``inputs_embeds`` to be present.
        """
        from lerobot.policies.common.flow_matching import euler_integrate

        if self.stage == "off":
            return super().sample_actions(images, img_masks, tokens, masks, states=states,
                                          state_masks=state_masks, noise=noise,
                                          num_steps=num_steps, **kwargs)
        if self.stage == "stage1":
            raise RuntimeError(
                "stage1 conditions on a ground-truth goal pose and has no inference path; "
                "evaluate a stage2 checkpoint."
            )
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize, device = tokens.shape[0], tokens.device
        if noise is None:
            noise = self.sample_noise(
                (bsize, self.config.chunk_size, self.config.max_action_dim), device
            )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks, states, state_masks
        )
        dtype = self.paligemma_with_expert.paligemma.model.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype
        if dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
        n_prefix = prefix_embs.shape[1]

        def step(x_t, timestep):
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
                x_t, timestep
            )
            if dtype == torch.bfloat16:
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
            att_2d = make_att_2d_masks(pad_masks, att_masks)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_4d = prepare_attention_masks_4d(att_2d)
            # goal_pose is unused by the stage2 branch -- the pose is decoded from vision.
            self._gp = self._build_goal_state(
                bsize, device, n_prefix, att_4d, None, masks, img_masks=img_masks
            )
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=self._gp["att_4d"],
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
            return self.action_out_proj(suffix_out)

        return euler_integrate(step, noise, num_steps)

    def _build_goal_state(self, bsize, device, n_prefix, att_4d, goal_pose, lang_masks,
                          img_masks=None):
        """Everything the per-layer hook needs, computed once per forward.

        The prefix layout is [images... | language], so the language block is the tail of
        length `lang_masks.shape[1]` and the images are everything before it. Deriving the
        split from the tensor rather than from the config keeps it correct if a camera is
        added or `empty_cameras` changes.
        """
        n_lang = int(lang_masks.shape[1])
        n_img = n_prefix - n_lang
        n_query = att_4d.shape[-2]
        n_suffix = n_query - n_prefix
        latent_dtype = torch.float32

        gp = {"att_4d": att_4d}

        if self.stage == "stage1":
            n_extra = int(self.config.num_goal_tokens)
            tokens = self.se3_encoder(goal_pose.to(latent_dtype))
            k = self._shape_kv(self.goal_to_key(tokens))
            v = self._shape_kv(self.goal_to_value(tokens))
            keep = None
        else:
            n_extra = int(self.aggregator.num_tokens)
            gp["latents"] = self.aggregator.initial_queries(bsize, device=device, dtype=latent_dtype)
            # Both cross-attention blocks read the same prefix hidden state; only the mask
            # distinguishes semantics from vision.
            sem = torch.zeros(bsize, n_prefix, dtype=torch.bool, device=device)
            img = torch.zeros(bsize, n_prefix, dtype=torch.bool, device=device)
            sem[:, n_img:] = lang_masks.to(torch.bool)
            if n_img > 0:
                img[:, :n_img] = True
                # empty_cameras pads the missing camera with an all -1 image and img_mask
                # False. The prefix attention mask drops it via pad_masks; the aggregator
                # builds its own mask and has to drop it too, or a third of the visual
                # context it reads is a constant dummy frame.
                if img_masks:
                    per = n_img // len(img_masks)
                    if per * len(img_masks) != n_img:
                        raise ValueError(
                            f"image span {n_img} does not divide evenly over "
                            f"{len(img_masks)} cameras; the prefix layout changed."
                        )
                    for ci, m in enumerate(img_masks):
                        m = m[:, -1] if m.ndim == 2 else m
                        img[:, ci * per : (ci + 1) * per] = m.to(torch.bool)[:, None]
            gp["semantic_mask"], gp["image_mask"] = sem, img
            k = v = None

            if bool(getattr(self.config, "mask_language_from_action_expert", False)):
                # The language block is the tail of the prefix; the action rows start at
                # n_prefix. Their own columns are left alone.
                att_4d = att_4d.clone()
                if att_4d.shape[0] == 1:
                    att_4d = att_4d.expand(bsize, -1, -1, -1).clone()
                att_4d[:, :, n_prefix:, n_img:n_prefix] = NEG
                gp["att_4d"] = att_4d

            if bool(getattr(self.config, "mask_image_from_action_expert", False)):
                # Every sample, not a 30% quota: with the images permanently gone the
                # latents are the only visual path, so there is nothing for them to lose
                # a softmax against. ref_only would now mean "no vision at all", so the
                # regime is skipped rather than re-weighted -- and note that setting
                # p_ref_only=0 would not have skipped it.
                ref_keep = syn_keep = None
                if n_img > 0:
                    att_4d = gp["att_4d"].clone()
                    if att_4d.shape[0] == 1:
                        att_4d = att_4d.expand(bsize, -1, -1, -1).clone()
                    att_4d[:, :, n_prefix:, :n_img] = NEG
                    gp["att_4d"] = att_4d
            else:
                ref_keep, syn_keep = sample_channel_regime(
                    bsize, device, self.config.p_ref_only, self.config.p_syn_only, self.training
                )
            keep = self.aggregator.context_keep_mask(bsize, device, self.training)
            if syn_keep is not None:
                keep = keep & syn_keep.unsqueeze(1)          # ref_only: drop every latent
            if ref_keep is not None and not bool(ref_keep.all()) and n_img > 0:
                # syn_only: the ACTION rows lose the image columns. The VLM keeps them --
                # the aggregator has to read vision from somewhere.
                att_4d = att_4d.clone()
                if att_4d.shape[0] == 1:
                    att_4d = att_4d.expand(bsize, -1, -1, -1).clone()
                blocked = ~ref_keep
                att_4d[blocked, :, n_prefix:, :n_img] = NEG
                gp["att_4d"] = att_4d

        # Extra mask columns: suffix rows only. A prefix row that could see these would be
        # reading latents derived from its own hidden state.
        mask = torch.full((bsize, 1, n_query, n_extra), NEG, dtype=att_4d.dtype, device=device)
        mask[:, :, n_prefix:, :] = 0.0
        if keep is not None:
            drop = (~keep).view(bsize, 1, 1, n_extra).expand(bsize, 1, n_query, n_extra)
            mask = mask.masked_fill(drop, NEG)
        gp["mask"] = mask
        if self.stage == "stage1":
            gp["k"], gp["v"] = k, v
        return gp


class PI05GoalPriorPolicy(PI05Policy):
    """pi05 with the two-stage goal-pose prior."""

    config_class = PI05GoalPriorConfig
    name = "pi05_goal_prior"

    def __init__(self, config: PI05GoalPriorConfig, **kwargs):
        # PI05Policy.__init__ is reproduced rather than called so the core model is built
        # exactly once. Calling super() and then replacing self.model builds it twice, and the
        # extra construction advances the global RNG -- which is what sample_noise and
        # sample_time draw from, so every step would see different noise from stock while the
        # weights, loaded from the checkpoint, still looked perfectly correct.
        require_package("transformers", extra="pi")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self.init_rtc_processor()
        self.model = PI05GoalPriorPytorch(config, rtc_processor=self.rtc_processor)
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        # PI05Policy.__init__ ends with these two; reproducing the constructor by hand had
        # dropped them. lerobot_eval calls policy.reset() before every rollout so evaluation
        # was unaffected, but select_action on a freshly built policy raised AttributeError
        # on _action_queue.
        self.model.to(config.device)
        self.reset()

    # Modules that do not exist in the pi05 checkpoint. Everything else must still be present:
    # pi05's from_pretrained loads with strict=True and swallows the exception in a bare
    # `except`, so one missing key silently leaves the ENTIRE model at random init while the
    # loss curve looks normal. Seeding only these prefixes keeps that guard working for the
    # weights that matter.
    _NEW_MODULE_PREFIXES = (
        "model.se3_encoder.",
        "model.goal_to_key.",
        "model.goal_to_value.",
        "model.aggregator.",
        "model.pose_norm.",
        "model.pose_decoder.",
    )

    # Stage 1 trains these; Stage 2 does not create them at all. Carrying them forward is the
    # whole point of the handoff -- the oracle channel retires and an inferred one takes its
    # slot -- but they still sit in the Stage 1 checkpoint and would be "unexpected" keys.
    _STAGE1_ONLY_PREFIXES = (
        "model.se3_encoder.",
        "model.goal_to_key.",
        "model.goal_to_value.",
    )

    def _prepare_pretrained_state_dict(self, state_dict):
        state_dict = super()._prepare_pretrained_state_dict(state_dict)
        current = self.state_dict()

        # Seed modules this stage owns but the checkpoint predates, then drop modules the
        # checkpoint owns but this stage retired. pi05's loader is strict=True inside a bare
        # `except`, so either mismatch would leave the ENTIRE model at random init behind a
        # perfectly normal loss curve. Only these two known prefix sets are touched: a genuinely
        # missing backbone key must still raise.
        for key, value in current.items():
            if key not in state_dict and key.startswith(self._NEW_MODULE_PREFIXES):
                state_dict[key] = value

        retired = [k for k in state_dict if k not in current and k.startswith(self._STAGE1_ONLY_PREFIXES)]
        for key in retired:
            del state_dict[key]
        if retired:
            import logging
            logging.info(
                f"goal_prior: retired {len(retired)} Stage-1-only tensors on load "
                f"(e.g. {retired[0]}); the oracle channel does not carry into Stage 2."
            )
        return state_dict

    # The VLM's own parameter group. `paligemma` covers the language tower AND the vision
    # tower, which is what MolmoAct2 does too (VLM 1e-5, ViT 1e-5). The action expert lives
    # under `gemma_expert` and stays at the top-level LR alongside the aggregator, the pose
    # head and the action/state projections.
    _VLM_PREFIX = "model.paligemma_with_expert.paligemma."

    # MolmoAct2 selects its aggregator group by the "semantic_visual_" name prefix, which
    # covers the aggregator, the pose norm and the pose decoder. These are the same three.
    _AGG_PREFIXES = ("model.aggregator.", "model.pose_norm.", "model.pose_decoder.")

    def get_optim_params(self):
        vlm_lr = getattr(self.config, "vlm_optimizer_lr", None)
        agg_lr = getattr(self.config, "aggregator_optimizer_lr", None)
        if self.config.goal_prior_stage != "stage2" or (vlm_lr is None and agg_lr is None):
            return self.parameters()
        vlm, agg, rest = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if agg_lr is not None and name.startswith(self._AGG_PREFIXES):
                agg.append(p)
            elif vlm_lr is not None and name.startswith(self._VLM_PREFIX):
                vlm.append(p)
            else:
                rest.append(p)
        if agg_lr is not None and not agg:
            raise ValueError(
                f"aggregator_optimizer_lr is set but no parameter starts with {self._AGG_PREFIXES!r}; "
                "the split would silently train the aggregator at the top-level LR."
            )
        if vlm_lr is not None and not vlm:
            raise ValueError(
                f"vlm_optimizer_lr is set but no parameter starts with {self._VLM_PREFIX!r}; "
                "the split would silently train everything at the top-level LR."
            )
        groups = [{"params": rest}]
        if agg:
            groups.append({"params": agg, "lr": float(agg_lr)})
        if vlm:
            groups.append({"params": vlm, "lr": float(vlm_lr)})
        logging.info(
            "goal_prior: param groups -- rest %d @ top-level lr | aggregator %d @ %s | VLM %d @ %s",
            len(rest), len(agg), agg_lr, len(vlm), vlm_lr,
        )
        return groups

    def _split_state(self, batch):
        """(current, goal) from the [B, 2, D] state the delta indices deliver.

        Order is [goal, current] -- see state_observation_delta_indices for why.
        """
        state = batch.get(OBS_STATE)
        if state is None or state.ndim != 3 or state.shape[1] < 2:
            return None, None
        d = self.model.pose_dim
        return state[:, -1, :d], state[:, 0, :d]

    def forward(self, batch, reduction: str = "mean"):
        if self.config.goal_prior_stage == "off":
            return super().forward(batch, reduction=reduction)

        images, img_masks = self._preprocess_images(batch)
        states, state_masks = self._prepare_memory_states(batch)
        tokens, masks = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        _, goal_pose = self._split_state(batch)
        if goal_pose is None:
            raise ValueError(
                "goal_prior needs observation.state with 2 timesteps (current, goal); got "
                f"{None if batch.get(OBS_STATE) is None else tuple(batch[OBS_STATE].shape)}. "
                "state_observation_delta_indices should be [0, target_pose_delta_index]."
            )

        actions = self.prepare_action(batch)
        noise = self.model.sample_noise(actions.shape, actions.device)
        time = self.model.sample_time(actions.shape[0], actions.device)
        prefix_mask = _sample_training_rtc_prefix_mask(
            actions.shape[0], actions.shape[1], self.config.rtc_training_max_delay, actions.device
        )

        losses, pose_pred = self.model.forward(
            images, img_masks, tokens, masks, actions, noise, time,
            prefix_mask=prefix_mask, states=states, state_masks=state_masks, goal_pose=goal_pose,
        )
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses = losses[:, :, :original_action_dim]
        action_loss = losses.mean()

        out = {"loss_action": action_loss.item()}
        loss = action_loss
        if pose_pred is not None:
            # Supervised in the normalised space the model already sees, so no single
            # dimension's scale dominates the term.
            pose_loss = F.mse_loss(pose_pred, goal_pose.to(pose_pred.dtype))
            loss = loss + float(self.config.pose_recon_loss_weight) * pose_loss
            out["loss_pose"] = pose_loss.item()
            out["gate"] = self.model.aggregator.gate_delta.detach().float().mean().item()
        return loss, out
