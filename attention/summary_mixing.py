""" SummaryMixing © 2023 by Samsung Electronics is licensed under CC BY-NC 4.0.

This library provides the basic building blocks for SummaryMixing.

Copied from https://github.com/SamsungLabs/SummaryMixing
Source: https://arxiv.org/abs/2307.07421

Authors
 * Titouan Parcollet 2023
 * Shucong Zhang 2023
 * Rogier van Dalen 2023
 * Sourav Bhattacharya 2023
 * Martin Kocour 2024
"""

import logging
import math

import numpy as np
import torch
import torch.nn as nn

from fast_transformers.attention_registry import AttentionRegistry, Optional, Int, Choice, Float, EventDispatcherInstance
from fast_transformers.events import EventDispatcher, AttentionEvent

logger = logging.getLogger(__name__)


class SummaryMixing(nn.Module):
    """This class implements SummaryMixing as defined
    in https://arxiv.org/abs/2307.07421

    Arguments
    ---------
    query_dimensions: int
        Feature dimension of the input tensor.
    n_heads : int
        Number of mixing heads.
    local_proj_hid_dim: list [int], optional
        A list of dimension specifying both the number of hidden layers
        as well as the size of them in the local projection branch
        (default: [512]).
    local_proj_out_dim: int, optional
        The dimension of the output of the local projection branch. This
        will be concatenated with the output of the summary branch
        (default: 512).
    summary_hid_dim: list [int], optional
        A list of dimension specifying both the number of hidden layers
        as well as the size of them in the summary projection branch
        (default: [512]).
    summary_out_dim: int, optional
        The dimension of the output of the summary projection branch. This
        will be concatenated with the output of the local branch
        (default: 512).
    activation: torch.nn.Module, optional
        Torch module specifying the activation function used in both the local
        and summary branches.
        (default: torch.nn.GELU)
    global_dropout: float, optional
        Amount of dropout applied when concatenating  the local and summary.
    mixing_mode: string, optional
        One of "SummaryMixing", "SummaryMixing-lite" or "SummaryMixing-fast". Changes the SummaryMixing cell
        according to the definition of the article. "SummaryMixing-lite" removes the
        local project branch. "SummaryMixing-expdecay" is another alternative using
        an exponential decay for the window, it's slower.


    Example
    -------
    >>> x = torch.rand(2,4,8)
    >>> sum = SummaryMixing(8)
    >>> out = sum(x)
    >>> print(out)
    torch.Size([2, 4, 8])
    """

    def __init__(
        self,
        query_dimensions,
        n_heads,
        activation="gelu",
        attention_dropout=0.1,
        event_dispatcher="",
        mixing_mode="SummaryMixing",
    ):
        super(SummaryMixing, self).__init__()

        if mixing_mode not in [
            "SummaryMixing",
            "SummaryMixing-lite",
            "SummaryMixing-expdecay",
            "SummaryMixing-fast",
        ]:
            raise ValueError(
                "The SummaryMixing mode should either be 'SummaryMixing', 'SummaryMixing-lite', 'SummaryMixing-fast' or 'SummaryMixing-expdecay'"
            )

        self.query_dimensions = query_dimensions
        if activation == "tanh":
            self.activation = nn.Tanh
        elif activation == "relu":
            self.activation = nn.ReLU
        elif activation == "gelu":
            self.activation = nn.GELU
        else:
            raise ValueError("Unknown activation type")

        self.mode = mixing_mode
        self.dropout = nn.Dropout(attention_dropout)

        if self.mode == "SummaryMixing" or self.mode == "SummaryMixing-expdecay":
            self.local_proj = ParallelLinear(
                n_neurons=query_dimensions*n_heads,
                input_shape=[None, None, n_heads, query_dimensions],
                n_split=n_heads,
                combine_out_dims=True
            )
            self.summary_local_merging = VanillaNN(
                input_shape=[None, None, n_heads, 2*query_dimensions],
                dnn_blocks=1,
                dnn_neurons=[query_dimensions*n_heads],
                activation=self.activation
            )

        if self.mode == "SummaryMixing-fast":
            raise NotImplementedError("SummaryMixing-fast mode is not implemented")
            # self.global_proj = VanillaNN(
            #     input_shape=[None, None, query_dimensions],
            #     dnn_blocks=1,
            #     dnn_neurons=self.local_proj_out_dim * 2,
            #     activation=activation,
            #     n_split=1,
            # )

            # self.summary_local_merging = VanillaNN(
            #     input_shape=[None, None, self.local_proj_out_dim * 2],
            #     dnn_blocks=1,
            #     dnn_neurons=[query_dimensions],
            #     activation=activation,
            # )
        else:
            self.summary_proj = ParallelLinear(
                n_neurons=query_dimensions*n_heads,
                input_shape=[None, None, n_heads, query_dimensions],
                n_split=n_heads,
                combine_out_dims=True
            )

        if self.mode == "SummaryMixing-expdecay":
            self.decay_constant = nn.Parameter(
                data=torch.tensor(0.995), requires_grad=False
            )

        self.apply(self._init_parameters)

    def forward(self, queries, keys, values, sum_mask, q_lengths, k_lengths):
        """This function simply goes forward!

        Arguments
        ---------
        queries, keys, values: torch.Tensor
            The expected shape is the standard one - [Batch, Time, Heads, Features]
        sum_mask: fast_transformers.masking.BaseMask
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        q_lengths, k_lengths: fast_transformers.masking.LengthMask
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """
        if self.mode == "SummaryMixing" or self.mode == "SummaryMixing-expdecay":
            return self._forward_mixing(queries, keys, values, sum_mask, q_lengths, k_lengths)
        elif self.mode == "SummaryMixing-fast":
            return self._forward_mixing_fast(queries, keys, values, sum_mask, q_lengths, k_lengths)
        elif self.mode == "SummaryMixing-lite":
            return self._forward_avgonly(queries, keys, values, sum_mask, q_lengths, k_lengths)

    def _forward_mixing(self, queries, keys, values, sum_mask, q_lengths, k_lengths):
        """Perform full SummaryMixing.

        Arguments
        ---------
        queries, keys, values: torch.Tensor
            The expected shape is the standard one - [Batch, Time, Heads, Features]
        sum_mask: fast_transformers.masking.BaseMask
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        q_lengths, k_lengths: fast_transformers.masking.LengthMask
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        B, T, H, F = queries.shape
        _, N, _, D = values.shape

        # f() (Eq. 1b)
        local_summary = self.local_proj(keys)
        if not k_lengths.all_ones:
            local_summary = local_summary * k_lengths.float_matrix[:, :, None]

        # s() (Eq. 2 and 1c)
        time_summary = self.summary_proj(queries)
        if not q_lengths:
            time_summary = time_summary * q_lengths.float_matrix[:, :, None]

        sum_mask = sum_mask.float_matrix if sum_mask is not None else None
        if self.mode == "SummaryMixing-expdecay":
            sum_mask = self._laplace_weights(T, self.decay_constant, sum_mask, keys.device)

        if sum_mask is None:
            # We normalise by real length by counting masking
            time_summary = torch.sum(time_summary, dim=1) / q_lengths.lengths[:, None]
            time_summary = time_summary.unsqueeze(1).repeat(1, T, 1, 1)
        else:
            # We must do a masked sum. The mask is [Time, Time] and the features are [B,T,F]
            # We therefore can do a matmul between [B,F,T] and [Time,Time].T to obtain [B,F,T] that we can re-transpose.
            # We need to be careful when dividing as padding is not included in sum_mask. We need to build the intersection
            # of both mask to know the actual real number of elements excluding padding.

            # full_mask_with_pad = torch.matmul(sum_mask, src_padding_mask)
            time_summary = torch.matmul(sum_mask, time_summary) / torch.sum(
                sum_mask, dim=1
            ).unsqueeze(-1)

        return self.summary_local_merging(
            self.dropout(torch.cat([local_summary, time_summary], dim=-1))
        ).contiguous()

    def _forward_mixing_fast(self, x, sum_mask, src_padding_mask):
        """Perform full SummaryMixing.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        sum_mask: torch.Tensor
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        src_padding_mask: torch.Tensor
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        B, T, F = x.shape

        global_proj = self.global_proj(x) * src_padding_mask
        split_global_proj = torch.split(global_proj, self.local_proj_out_dim, dim=-1)

        # split_global_proj[0] = local projection
        # split_global_proj[1] = summary projection
        if sum_mask is None:
            # We normalise by real length by counting masking
            time_summary = torch.sum(split_global_proj[1], dim=1) / torch.sum(
                src_padding_mask, dim=1
            )
            time_summary = time_summary.unsqueeze(1).repeat(1, T, 1)

        else:

            # We must do a masked sum. The mask is [Time, Time] and the features are [B,T,F]
            # We therefore can do a matmul between [B,F,T] and [Time,Time].T to obtain [B,F,T] that we can re-transpose.
            # We need to be careful when dividing as padding is not included in sum_mask. We need to build the intersection
            # of both mask to know the actual real number of elements excluding padding.

            # full_mask_with_pad = torch.matmul(sum_mask, src_padding_mask)

            time_summary = torch.matmul(sum_mask, split_global_proj[1]) / torch.sum(
                sum_mask, dim=1
            ).unsqueeze(-1)

        return self.summary_local_merging(
            self.dropout(torch.cat([split_global_proj[0], time_summary], dim=-1))
        )

    def _forward_avgonly(self, x, sum_mask, src_padding_mask):
        """Perform SummaryMixing-lite.

        Arguments
        ---------
        x: torch.Tensor
            The expected shape is the standard SpeechBrain one - [Batch, Time, Features]
        sum_mask: torch.Tensor
            (Time, Time) per time step mask that can be used to compute different sum between time-step.
            this can be useful for streaming, for instance, where each time step has a limited context.
        src_padding_mask: torch.Tensor
            (Batch, Time) corresponding to padding. We avoid padding when summarizing in time.
        """

        B, T, F = x.shape

        # s() We just do the mean over time
        # Then we repeat the output matrix T times along the time axis
        time_summary = self.summary_proj(x) * src_padding_mask
        time_summary = torch.sum(time_summary, dim=1) / torch.sum(
            src_padding_mask, dim=1
        )
        time_summary = time_summary.unsqueeze(1).expand(-1, T, -1)

        return time_summary

    def _init_parameters(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.zeros_(module.bias)

    def _laplace_weights(
        self,
        size: int,
        decay_constant,
        binary_mask=None,
        device="cpu",
        normalise=False,
    ):
        """
        Return a square matrix with the diagonal entries the maximum one in each row
        and the entries left and right decaying exponentially.
        This is like a discrete Laplacian distribution.
        If normalise is set to True, in each row, the entries add up to 1.

        Arguments
        ---------
        size: int
            The height and width of the returned matrix.
        decay_constant: float
            The exponential decay per position.
            This must be a positive value, and will normally be less than 1.
        binary_mask: torch.Tensor
            A binary mask applied before the rows are normalised.
        device: str
            Torch device to copy the generated masks to.
        """

        # Fill a matrix with integers indicating how far away each element is from
        # the diagonal.
        horizontal_distance_to_diagonal = torch.abs(
            torch.arange(size) - torch.arange(size).unsqueeze(-1)
        ).to(device)

        # A Laplacian-like shape with the correct decay, but where the diagonal
        # elements are all 1.
        absolute_laplacian = torch.exp(
            horizontal_distance_to_diagonal * torch.log(decay_constant)
        )

        if binary_mask is not None:
            absolute_laplacian = absolute_laplacian * binary_mask

        if normalise:
            # Normalise each row.
            normalised = absolute_laplacian / torch.sum(
                absolute_laplacian, dim=1, keepdim=True
            )
            return normalised

        return absolute_laplacian

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.A_weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.B_weights, a=math.sqrt(5))


# Register the attention implementation so that it becomes available in
# FastTransformers builder
AttentionRegistry.register(
    "summary_mixing", SummaryMixing,
    [
        ("query_dimensions", Int),
        ("n_heads", Int),
        ("activation", Optional(Choice(["tanh", "relu", "gelu"]), "gelu")),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, "")),
        ("mixing_mode", Optional(Choice(["SummaryMixing", "SummaryMixing-lite",
                                         "SummaryMixing-expdecay", "SummaryMixing-fast"]), "SummaryMixing"))
    ]
)


class ParallelLinear(torch.nn.Module):
    """Computes a parallel linear transformation y = wx + b.
    In practice the input and the output are split n_split times.
    Hence we create n_split parallel linear op that will operate on
    each split dimension. E.g. if x = [B,T,F] and n_split = 4
    then x = [B,T,4,F/4] and W = [4,F/4,out_dim/4].

    Arguments
    ---------
    n_neurons : int
        It is the number of output neurons (i.e, the dimensionality of the
        output).
    input_shape: tuple, optional
        It is the shape of the input tensor.
    input_size: int, optional
        Size of the input tensor.
    n_split: int, optional
        The number of split to create n_split linear transformations.
    bias : bool, optional
        If True, the additive bias b is adopted.
    combiner_out_dims : bool, optional
        If True, the output vector is reshaped to be [B, T, S].

    Example
    -------
    >>> x = torch.rand([64, 50, 512])
    >>> lin_t = ParallelLinear(n_neurons=64, input_size=512, n_split=4)
    >>> output = lin_t(x)
    >>> output.shape
    torch.Size([64, 50, 64])
    """

    def __init__(
        self,
        n_neurons,
        input_shape,
        input_size=None,
        n_split=1,
        bias=True,
        combine_out_dims=True,
    ):
        super().__init__()
        self.n_split = n_split
        self.combine_out_dims = combine_out_dims

        if input_shape is None and input_size is None:
            raise ValueError("Expected one of input_shape or input_size")

        if input_size is None:
            input_size = input_shape[-1]
            if len(input_shape) == 4:
                input_size = input_shape[-1] * input_shape[-2]

        if input_size % n_split != 0 or n_neurons % n_split != 0:
            raise ValueError("input_size and n_neurons must be dividible by n_split!")

        self.split_inp_dim = input_size // n_split
        self.split_out_dim = n_neurons // n_split

        self.weights = nn.Parameter(
            torch.empty(self.n_split, self.split_inp_dim, self.split_out_dim)
        )
        self.biases = nn.Parameter(torch.zeros(self.n_split, self.split_out_dim))

        self._reset_parameters()

    def extra_repr(self):
        return f"n_split={self.n_split}, in_features={self.split_inp_dim}, out_features={self.split_out_dim}, bias=True, combine_dims={self.combine_out_dims}"

    def _reset_parameters(self):
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.biases, a=math.sqrt(5))

    def forward(self, x):
        """Returns the linear transformation of input tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input to transform linearly, may be 3 or four dimensional.
            [B,T,F] or [B,T,n_split,F//n_split]
        """
        if x.ndim == 3:
            B, T, F = x.shape
            x = x.view(B, T, self.n_split, self.split_inp_dim)

        x = torch.einsum("btmf,mfh->btmh", x, self.weights) + self.biases

        if self.combine_out_dims:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        return x


class VanillaNN(nn.Sequential):
    """A simple vanilla Deep Neural Network.

    Arguments
    ---------
    input_shape : tuple
        Expected shape of the input tensors.
    activation : torch class
        A class used for constructing the activation layers.
    dnn_blocks : int
        The number of linear neural blocks to include.
    dnn_neurons : int or list[int]
        The number of neurons in the different linear layers.
        If a list is given, the length must correspond to the
        number of layers. If a int is given, all layers will
        have the same size.
    n_split: int
        The number of split to create n_split linear transformations.
        In practice the input and the output are split n_split times.
        Hence we create n_split parallel linear op that will operate on
        each split dimension. E.g. if x = [B,T,F] and n_split = 4
        then x = [B,T,4,F/4] and W = [4,F/4,out_dim/4]. This will happen
        in each layer of the VanillaNN.

    Example
    -------
    >>> inputs = torch.rand([10, 120, 60])
    >>> model = VanillaNN(input_shape=inputs.shape)
    >>> outputs = model(inputs)
    >>> outputs.shape
    torch.Size([10, 120, 512])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        dnn_blocks=2,
        dnn_neurons=512,
        n_split=1,
    ):
        super().__init__()

        if isinstance(dnn_neurons, list):
            if len(dnn_neurons) != dnn_blocks:
                msg = "The length of the dnn_neurons list must match dnn_blocks..."
                raise ValueError(msg)

        for block_index in range(dnn_blocks):
            if isinstance(dnn_neurons, list):
                current_nb_neurons = dnn_neurons[block_index]
            else:
                current_nb_neurons = dnn_neurons

            if n_split > 1:
                # ParrallelLinear does a costly reshape operation, hence we minimise this
                # cost by only doing this reshape for the last layer of the MLP.
                if block_index < (dnn_blocks - 1):
                    combine_out_dims = False
                else:
                    combine_out_dims = True
                self.append(
                    ParallelLinear(
                        n_neurons=current_nb_neurons,
                        input_shape=input_shape,
                        n_split=n_split,
                        bias=True,
                        combine_out_dims=combine_out_dims,
                    )
                )
            else:
                input_size = input_shape[-1] * input_shape[-2] if len(input_shape) > 3 else input_shape[-1]
                self.append(
                    nn.Linear(input_size, current_nb_neurons, bias=True)
                )
            self.append(activation())
