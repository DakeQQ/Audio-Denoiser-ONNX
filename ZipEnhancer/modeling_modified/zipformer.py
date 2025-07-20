#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao)
# Copyright (c) 2024 Alibaba, Inc. and its affiliates.
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import logging
import math
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from .scaling import \
    Identity  # more friendly to backward hooks than nn.Identity(), for diagnostic reasons.
from .scaling import \
    ScaledLinear  # not as in other dirs.. just scales down initial parameter values.
from .scaling import (ActivationDropoutAndLinear, BiasNorm,
                      ChunkCausalDepthwiseConv1d, Dropout2, FloatLike,
                      ScheduledFloat, limit_param_value,
                      penalize_abs_values_gt, softmax)


class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module (default=31).

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        attention_skip_rate: FloatLike = ScheduledFloat(
            (0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05),
                                                   (16000, 0.0),
                                                   default=0),
        const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25),
                                                         (4000.0, 0.025),
                                                         default=0),
        ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01),
                                                  (50000.0, 0.0)),
        ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01),
                                                  (50000.0, 0.0)),
        bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                     (4000.0, 0.02),
                                                     default=0),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(
            embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 3) // 4,
                                               dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim,
                                               dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim,
                                               (feedforward_dim * 5) // 4,
                                               dropout)

        self.nonlin_attention = NonlinAttention(
            embed_dim, hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal)

        self.conv_module2 = ConvolutionModule(
            embed_dim, cnn_module_kernel, causal=causal)

        # TODO: remove it
        self.bypass_scale = nn.Parameter(torch.full((embed_dim, ), 0.5))

        self.norm = BiasNorm(embed_dim)

    def get_sequence_dropout_mask(self, x: Tensor,
                                  dropout_rate: float) -> Optional[Tensor]:
        if (dropout_rate == 0.0 or not self.training
                or torch.jit.is_scripting() or torch.jit.is_tracing()):
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(
            x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        src_orig = src
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = src + self.feed_forward1(src)
        src = src + self.nonlin_attention(src, attn_weights[:1])
        src = src + self.self_attn1(src, attn_weights)
        src = src + self.conv_module1(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)
        src = src + self.feed_forward2(src)
        src = self.bypass_mid(src_orig, src)
        src = src + self.self_attn2(src, attn_weights)
        src = src + self.conv_module2(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask)
        src = src + self.feed_forward3(src)
        src = self.norm(src)
        src = self.bypass(src_orig, src)
        return src


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2),
                                              default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim, ), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 corresponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing(
        ) or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(
                self.bypass_scale,
                min=float(self.scale_min),
                max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand(
                    (batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                _rand_tensor = torch.rand((batch_size, 1), device=ans.device)
                mask = (_rand_tensor < straight_through_rate)
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor):
        return src_orig + (src - src_orig) * self.bypass_scale


class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, channels: int, downsample: int, dropout: FloatLike):
        super(SimpleDownsample, self).__init__()

        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None  # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample
        self.downsample_minus = self.downsample - 1

    def forward(self, src: Tensor) -> Tensor:
        (seq_len, batch_size, in_channels) = src.shape
        d_seq_len = (seq_len + self.downsample_minus) // self.downsample
        src_extra = src[seq_len - 1:].expand(d_seq_len * self.downsample - seq_len, batch_size, in_channels)
        src = torch.cat((src, src_extra), dim=0).reshape(d_seq_len, self.downsample, batch_size, in_channels)
        return (src * self.bias.softmax(dim=0).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        (seq_len, batch_size, num_channels) = src.shape
        return src.unsqueeze(1).expand(seq_len, self.upsample, batch_size, num_channels).reshape(-1, batch_size, num_channels)


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the Fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish individual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, embed_dim
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0, length_factor
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        """Reset the positional encodings."""
        T = x.size(0) + left_context_len

        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= T * 2 - 1:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(
            -(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        _tmp_tensor = ((x.abs() + compression_length).log()
                       - math.log(compression_length))
        x_compressed = (compression_length * x.sign() * _tmp_tensor)

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed
                  / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.unsqueeze(0).to(dtype=x.dtype)
        self.factor = self.pe.size(1) // 2

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        x_size = x.size(0)
        return self.pe[:, self.factor - x_size + 1:self.factor + x_size, :, ]


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5),
                                                      (4000.0, 0.0)),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim
                       + pos_head_dim) * num_heads
        self.query_dim = query_head_dim * num_heads
        self.query_dim_2 = 2 * self.query_dim

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(
            embed_dim,
            in_proj_dim,
            bias=True,
            initial_scale=query_head_dim**-0.25)

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(
            pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05)

        self.rows = torch.arange(start=1024, end=-1, step=-1, dtype=torch.int16)
        self.cols = torch.arange(1024, dtype=torch.int16)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.in_proj(x)
        seq_len, batch_size, qkp_len = x.shape
        q, k, p = torch.split(x, [self.query_dim, self.query_dim, qkp_len - self.query_dim_2], dim=-1)
        q = q.reshape(seq_len, batch_size, self.num_heads, self.query_head_dim).permute(2, 1, 0, 3)
        p = p.reshape(seq_len, batch_size, self.num_heads, self.pos_head_dim).permute(2, 1, 0, 3)
        k = k.reshape(seq_len, batch_size, self.num_heads, self.query_head_dim).permute(2, 1, 3, 0)
        attn_scores = torch.matmul(q, k)
        pos_scores = torch.matmul(p, self.linear_pos(pos_emb).reshape(-1, seq_len + seq_len - 1, self.num_heads, self.pos_head_dim).permute(2, 0, 3, 1))
        (num_heads, batch_size, time1, n) = pos_scores.shape
        rows = self.rows[-time1:].to(torch.int64)
        cols = self.cols[:seq_len].to(torch.int64)
        rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
        pos_scores = pos_scores.reshape(-1, n)
        pos_scores = torch.gather(pos_scores, dim=1, index=rows + cols)
        pos_scores = pos_scores.reshape(num_heads, batch_size, time1, seq_len)
        attn_scores.add_(pos_scores)
        return torch.softmax(attn_scores, dim=-1, dtype=attn_scores.dtype)


class SelfAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(
            embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(
            num_heads * value_head_dim,
            embed_dim,
            bias=True,
            initial_scale=0.05)

        self.num_heads = 4

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        (seq_len, batch_size, _) = x.shape
        x = self.in_proj(x).reshape(seq_len, batch_size, self.num_heads, -1).permute(2, 1, 0, 3)
        return self.out_proj(torch.matmul(attn_weights, x).permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, self.out_proj.in_features))


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(self, embed_dim: int, feedforward_dim: int,
                 dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation='SwooshL',
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

    def forward(self, x: Tensor):
        return self.out_proj(self.in_proj(x))


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        self.tanh = nn.Tanh()

        self.out_proj = ScaledLinear(
            hidden_channels, channels, bias=True, initial_scale=0.05)

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        x = self.in_proj(x)
        (seq_len, batch_size, _) = x.shape
        s, x, y = x.chunk(3, dim=2)
        x *= self.tanh(s)
        x = torch.matmul(attn_weights, x.unsqueeze(2).permute(2, 1, 0, 3)).permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)
        return self.out_proj(x * y)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        self.sigmoid = nn.Sigmoid()

        assert kernel_size % 2 == 1

        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(
                channels=bottleneck_dim, kernel_size=kernel_size)
            if causal else nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            ))

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation='SwooshR',
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
    ) -> Tensor:
        x, s = self.in_proj(x).chunk(2, dim=2)
        return self.out_proj(self.depthwise_conv((x * self.sigmoid(s)).permute(1, 2, 0)).permute(2, 0, 1))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
