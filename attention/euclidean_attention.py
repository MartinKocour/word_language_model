import torch
from torch import nn

from fast_transformers.attention_registry import AttentionRegistry, Optional, Int, Choice, Float, EventDispatcherInstance
from fast_transformers.events import EventDispatcher, AttentionEvent

from .additive_attention import BaseAttention


class EuclideanAttention(BaseAttention):
    def forward_score(self, queries, keys):
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape

        # Compute energy
        Q = queries.reshape(B, L, 1, H, E)
        K = keys.reshape(B, 1, S, H, E)
        # broadcasted sum
        E = Q - K  # [B, L, S, H, E]
        E = -1 * torch.sum(E**2, dim=-1)
        return E  # [B, L, S, H]


# Register the attention implementation so that it becomes available in
# FastTransformers builder
AttentionRegistry.register(
    "euclidean", EuclideanAttention,
    [
        ("query_dimensions", Int),
        ("n_heads", Int),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)


class FastEuclideanAttention(nn.Module):
    def __init__(
        self,
        query_dimensions,
        n_heads,
        attention_dropout=0.1,
        event_dispatcher="",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.query_dimensions = query_dimensions
        self.dropout = nn.Dropout(attention_dropout)
        self.w = nn.Parameter(torch.ones(n_heads, query_dimensions, device=device, dtype=dtype))

    def forward(self, queries, keys, values, attn_mask, q_lengths, k_lengths):
        if attn_mask is not None and attn_mask.lower_triangular:
            return self._causal_fast_euclidean(queries, keys, values, q_lengths, k_lengths)
        else:
            return self._fast_euclidean(queries, keys, values, q_lengths, k_lengths)

    def _compute_summary(self, queries, keys, values, q_mask=None, k_mask=None):
        scores = torch.einsum("blhe,he->blh", queries, self.w)
        if q_mask is not None:
            scores = scores * q_mask[:, :, None]
        scores = torch.softmax(scores, dim=1)  # [B, L, H]
        scores = scores.unsqueeze(-1)  # [B, L, H, 1]
        qhat = torch.sum(scores * queries, dim=1).unsqueeze(1)  # [B, 1, H, E]

        scores = qhat - keys  # [B, T, H, E]
        scores = -1 * torch.sum(scores**2, dim=-1)  # [B, T, H]
        if k_mask is not None:
            scores = scores * k_mask[:, :, None]
        scores = torch.softmax(scores, dim=1)
        scores = scores.unsqueeze(-1)  # [B, T, H, 1]
        summary = torch.sum(scores * values, dim=1)  # [B, H, E]

        return summary.unsqueeze(1)  # [B, 1, H, E]

    def _fast_euclidean(self, queries, keys, values, q_lengths, k_lengths):
        summary = self._compute_summary(queries, keys, values, q_lengths.float_matrix, k_lengths.float_matrix)
        return queries + summary

    def _causal_fast_euclidean(self, queries, keys, values, q_lengths, k_lengths):
        # Q, K, V: [B, L, H, E]
        q_mask = q_lengths.float_matrix
        k_mask = k_lengths.float_matrix
        for i in range(0, queries.size(1)):
            if i == 0:
                # [B, 1, H, E]
                summary = self._compute_summary(queries[:, :1], keys[:, :1], values[:, :1], q_mask[:, :1], k_mask[:, :1])
            else:
                summary = torch.cat(
                    (summary, self._compute_summary(queries[:, :i+1], keys[:, :i+1], values[:, :i+1], q_mask[:, :1], k_mask[:, :1])),
                    1,
                )
        return queries + summary


# Register the attention implementation so that it becomes available in
# FastTransformers builder
AttentionRegistry.register(
    "fast_euclidean", FastEuclideanAttention,
    [
        ("query_dimensions", Int),
        ("n_heads", Int),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
