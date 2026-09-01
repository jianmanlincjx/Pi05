#!/usr/bin/env python
"""Goal-pose prior modules for pi05, ported from MolmoAct2 v3 and FastWAM.

Shapes and hyper-parameters follow the v3 checkpoint config rather than being re-derived:
100 latents, 8 pose-supervised, hidden 768, 8 heads, ffn 4.0, self-attention on, 6 layer
groups, pose loss weight 0.3.
"""

from __future__ import annotations

import torch
from torch import nn


# --------------------------------------------------------------------------------------
# Stage 1: the oracle channel
# --------------------------------------------------------------------------------------
class SE3Encoder(nn.Module):
    """One target pose -> `num_tokens` goal tokens.

    The target is a single point (the end of the action chunk); the tokens are a
    representational width for that one target, not several targets.
    """

    def __init__(self, pose_dim: int, hidden_dim: int, num_tokens: int, latent_dim: int):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.latent_dim = int(latent_dim)
        self.net = nn.Sequential(
            nn.Linear(int(pose_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), self.num_tokens * self.latent_dim),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return self.net(pose).view(pose.shape[0], self.num_tokens, self.latent_dim)


class GoalPoseDecoder(nn.Module):
    """Stage 2: reconstruct the target pose from the K pose-supervised latents."""

    def __init__(self, num_tokens: int, hidden_size: int, pose_dim: int, inner_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(num_tokens) * int(hidden_size), inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, int(pose_dim)),
        )

    def forward(self, goal_hidden: torch.Tensor) -> torch.Tensor:
        return self.net(goal_hidden.reshape(goal_hidden.shape[0], -1))


# --------------------------------------------------------------------------------------
# Stage 2: the aggregator
# --------------------------------------------------------------------------------------
class _SelfAttentionBlock(nn.Module):
    """Lets the latents talk to each other before they read the streams."""

    def __init__(self, latent_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(latent_dim, num_heads, dropout=dropout, batch_first=True)
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, latent_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(tokens)
        tokens = tokens + self.dropout(self.attn(h, h, h, need_weights=False)[0])
        return tokens + self.dropout(self.ffn(self.ffn_norm(tokens)))


class _CrossAttentionBlock(nn.Module):
    """Reads one stream, selected by mask rather than by a separate tensor."""

    def __init__(self, latent_dim: int, context_dim: int, num_heads: int, ffn_ratio: float, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(latent_dim)
        self.context_norm = nn.LayerNorm(context_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim, num_heads=num_heads, dropout=dropout,
            kdim=context_dim, vdim=context_dim, batch_first=True,
        )
        inner = max(1, int(round(latent_dim * ffn_ratio)))
        self.ffn_norm = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, latent_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, context: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if tuple(valid_mask.shape) != tuple(context.shape[:2]):
            raise ValueError(
                f"cross-attention mask must be {tuple(context.shape[:2])}, got {tuple(valid_mask.shape)}"
            )
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("cross-attention received an empty context row")
        ctx = self.context_norm(context)
        update, _ = self.cross_attn(
            self.query_norm(queries), ctx, ctx,
            key_padding_mask=~valid_mask.to(torch.bool), need_weights=False,
        )
        queries = queries + self.dropout(update)
        return queries + self.dropout(self.ffn(self.ffn_norm(queries)))


class _AggregatorGroup(nn.Module):
    """One depth group: self-attend, then read semantics, then read vision."""

    def __init__(self, *, latent_dim, context_dim, kv_dim, num_heads, ffn_ratio, dropout,
                 enable_self_attention):
        super().__init__()
        self.self_block = (
            _SelfAttentionBlock(latent_dim, num_heads, ffn_ratio, dropout)
            if enable_self_attention else None
        )
        kw = dict(latent_dim=latent_dim, context_dim=context_dim, num_heads=num_heads,
                  ffn_ratio=ffn_ratio, dropout=dropout)
        self.semantic_block = _CrossAttentionBlock(**kw)
        self.visual_block = _CrossAttentionBlock(**kw)
        self.to_key = nn.Linear(latent_dim, kv_dim, bias=False)
        self.to_value = nn.Linear(latent_dim, kv_dim, bias=False)

    def forward(self, queries, layer_hidden, *, semantic_mask, image_mask):
        if self.self_block is not None:
            queries = self.self_block(queries)
        queries = self.semantic_block(queries, layer_hidden, semantic_mask)
        return self.visual_block(queries, layer_hidden, image_mask)

    def project_kv(self, tokens):
        return self.to_key(tokens), self.to_value(tokens)


class SemanticVisualAggregator(nn.Module):
    """Grouped recurrent latent aggregator with one shared learnable query bank.

    Refreshed at *every* layer using that layer's group weights, matching v3: 18 layers over
    6 groups means each group's weights are applied 3 times.
    """

    def __init__(self, *, num_tokens, latent_dim, context_dim, kv_dim, num_heads, ffn_ratio,
                 dropout, enable_self_attention, num_layer_groups,
                 num_pose_tokens, gate_bias_init, gate_pose_tokens,
                 context_token_dropout, context_blackout_prob):
        super().__init__()
        if int(num_layer_groups) < 1:
            raise ValueError(f"num_layer_groups must be >= 1, got {num_layer_groups}")
        self.num_layer_groups = int(num_layer_groups)
        self.num_tokens = int(num_tokens)
        self.num_pose_tokens = int(num_pose_tokens)
        self.context_token_dropout = float(context_token_dropout)
        self.context_blackout_prob = float(context_blackout_prob)

        q = torch.empty(self.num_tokens, int(latent_dim))
        nn.init.trunc_normal_(q, std=0.02)
        self.queries = nn.Parameter(q)

        self.groups = nn.ModuleList([
            _AggregatorGroup(
                latent_dim=int(latent_dim), context_dim=int(context_dim), kv_dim=int(kv_dim),
                num_heads=int(num_heads), ffn_ratio=float(ffn_ratio), dropout=float(dropout),
                enable_self_attention=bool(enable_self_attention),
            )
            for _ in range(self.num_layer_groups)
        ])

        # The gate is stored as a constant init plus a learnable delta, summed in fp32.
        # dtype here is bfloat16: at -2.0 the bf16 grid spacing is 0.0078 while AdamW moves
        # ~1e-4 per step, so a plain bf16 parameter rounds every update straight back and the
        # gate stays frozen for the whole run behind a healthy loss curve.
        self.register_buffer("gate_init", torch.full((self.num_layer_groups,), float(gate_bias_init)))
        self.gate_delta = nn.Parameter(torch.zeros(self.num_layer_groups))
        self.gate_pose_tokens = bool(gate_pose_tokens)

    def initial_queries(self, batch_size, *, device, dtype):
        return self.queries.to(device=device, dtype=dtype).unsqueeze(0).expand(int(batch_size), -1, -1)

    def validate_num_layers(self, num_layers: int) -> None:
        if int(num_layers) < 1 or int(num_layers) % self.num_layer_groups != 0:
            raise ValueError(
                f"layer count must be divisible by num_layer_groups, got {num_layers} and "
                f"{self.num_layer_groups}"
            )

    def layer_group_index(self, layer_idx: int, num_layers: int) -> int:
        self.validate_num_layers(num_layers)
        return int(layer_idx) // (int(num_layers) // self.num_layer_groups)

    def forward(self, queries, layer_hidden, *, semantic_mask, image_mask, group_idx=0):
        return self.groups[int(group_idx)](
            queries, layer_hidden, semantic_mask=semantic_mask, image_mask=image_mask
        )

    def project_kv(self, tokens, *, group_idx=0):
        return self.groups[int(group_idx)].project_kv(tokens)

    def gate_bias(self, group_idx: int) -> torch.Tensor:
        return self.gate_init[int(group_idx)].float() + self.gate_delta[int(group_idx)].float()

    def gated_span(self) -> tuple[int, int]:
        """Columns the gate applies to. The pose-supervised span stays ungated so that at
        step 0 the action expert still sees exactly Stage 1's interface."""
        return (0, self.num_tokens) if self.gate_pose_tokens else (self.num_pose_tokens, self.num_tokens)

    def context_keep_mask(self, batch_size, device, training):
        """Per-sample keep mask over the latents. **Pose columns are never dropped.**

        Column dropout stops the action expert leaning on a single aggregate; blackout removes
        the context block outright for a fraction of samples. Both touch the free context
        columns only, because the fallback state has to be Stage 1's interface
        -- [language | state | pose | action] -- not [language | state | action]. Dropping the
        pose columns here would sever the very prior Stage 2 exists to carry forward, and it
        would do so silently: the loss curve looks the same either way.

        The channel regime is different and is applied by the caller: `ref_only` drops all
        100 columns including pose, since its whole purpose is to break dependence on the
        synthetic channel entirely.
        """
        keep = torch.ones(int(batch_size), self.num_tokens, dtype=torch.bool, device=device)
        if not training:
            return keep
        n_ctx = self.num_tokens - self.num_pose_tokens
        if n_ctx > 0 and self.context_token_dropout > 0:
            drop = torch.rand(int(batch_size), n_ctx, device=device) < self.context_token_dropout
            keep[:, self.num_pose_tokens :] &= ~drop
        if n_ctx > 0 and self.context_blackout_prob > 0:
            black = torch.rand(int(batch_size), device=device) < self.context_blackout_prob
            keep[black, self.num_pose_tokens :] = False
        return keep


# --------------------------------------------------------------------------------------
# Channel regime
# --------------------------------------------------------------------------------------
def sample_channel_regime(batch_size, device, p_ref_only, p_syn_only, training):
    """Per-sample (ref_keep, syn_keep), or (None, None) outside training.

    Fixed counts rather than independent coins: every rank then sees every regime in every
    batch, so the per-regime metric keys are always defined. Ranks that disagree about which
    keys exist deadlock the trainer's per-key all-gather.

    The counts round per batch, so the *effective* ratio depends on batch size -- which is why
    the config states effective values (both 0.50 / ref_only 0.20 / syn_only 0.30) rather than
    the nominal ones ImageWAM was configured with.
    """
    if not training:
        return None, None
    n = int(batch_size)
    if n <= 0:
        return None, None
    n_ref_only = int(round(float(p_ref_only) * n))
    n_syn_only = int(round(float(p_syn_only) * n))
    if n >= 3:
        n_ref_only = max(1, min(n_ref_only, n - 2))
        n_syn_only = max(1, min(n_syn_only, n - 1 - n_ref_only))
    else:
        n_ref_only = min(n_ref_only, n)
        n_syn_only = min(n_syn_only, n - n_ref_only)
    order = torch.randperm(n, device=device)
    ref_keep = torch.ones(n, dtype=torch.bool, device=device)
    syn_keep = torch.ones(n, dtype=torch.bool, device=device)
    ref_keep[order[:n_syn_only]] = False                              # syn-only samples
    syn_keep[order[n_syn_only : n_syn_only + n_ref_only]] = False     # ref-only samples
    return ref_keep, syn_keep
