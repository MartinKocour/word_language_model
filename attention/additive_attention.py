import torch
from torch import nn

from fast_transformers.attention_registry import AttentionRegistry, Optional, Int, Choice, Float, EventDispatcherInstance
from fast_transformers.events import EventDispatcher, AttentionEvent


class BaseAttention(nn.Module):
    def __init__(self, query_dimensions, n_heads, attention_dropout=0.1, event_dispatcher="", device=None, dtype=None):
        super().__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths, key_lengths):
        # queries and keys are already projected
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Compute energy
        E = self.forward_score(queries, keys)  # [B, L, S, H]
        E = E.permute(0, 3, 1, 2)  # [B, H, L, S]

        # Apply mask
        if not attn_mask.all_ones:
            E = E + attn_mask.additive_matrix
        if not key_lengths.all_ones:
            E = E + key_lengths.additive_matrix[:, None, None]

        # Compute attention
        A = self.dropout(torch.softmax(E, dim=-1))  # [B, H,  L, S]
        V = torch.einsum("bhls,bshd->blhd", A, values)  # [B, L, H, D]

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))
        return V.contiguous()

    def forward_score(self, queries, keys):
        raise NotImplementedError("BaseAttention stub")


class AdditiveAttention(BaseAttention):
    def __init__(self, query_dimensions, n_heads, activation="gelu", attention_dropout=0.1, event_dispatcher="", device=None, dtype=None):
        super().__init__(query_dimensions, n_heads, attention_dropout, event_dispatcher, device, dtype)
        if activation == "tanh":
            self.act = nn.Tanh()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError("Unknown activation type")
        self.v = nn.Parameter(torch.ones(n_heads, query_dimensions, device=device, dtype=dtype))

    def extra_repr(self):
        S = " x ".join(map(str, self.v.shape))
        return f"(w): Tensor({S})"

    def forward_score(self, queries, keys):
        # queries and keys are already projected
        B, L, H, E = queries.shape
        _, S, _, _ = keys.shape

        # Compute energy
        Q = queries.reshape(B, L, 1, H, E)
        K = keys.reshape(B, 1, S, H, E)
        # broadcasted sum
        E = Q + K  # [B, L, S, H, E]
        E = self.act(E)
        E = torch.einsum("blshe,he->blsh", E, self.v)
        return E


# Register the attention implementation so that it becomes available in
# FastTransformers builder
AttentionRegistry.register(
    "additive", AdditiveAttention,
    [
        ("query_dimensions", Int),
        ("n_heads", Int),
        ("activation", Optional(Choice(["tanh", "relu", "gelu"]), "gelu")),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
