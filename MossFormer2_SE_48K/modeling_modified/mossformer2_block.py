"""
This source code is modified by Shengkui Zhao based on https://github.com/lucidrains/FLASH-pytorch
This revised version replaces einsum and rearrange with basic tensor operations.
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from rotary_embedding_torch import RotaryEmbedding
from .conv_module import ConvModule, GLU, FFConvM_Dilated
from .fsmn import UniDeepFsmn, UniDeepFsmn_dilated
from torchinfo import summary
from .layer_norm import CLayerNorm, GLayerNorm, GlobLayerNorm, ILayerNorm

# Helper functions

def identity(t, *args, **kwargs):
    """
    Returns the input tensor unchanged.

    Args:
        t (torch.Tensor): Input tensor.
        *args: Additional arguments (ignored).
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        torch.Tensor: The input tensor.
    """
    return t

def append_dims(x, num_dims):
    """
    Adds additional dimensions to the input tensor.

    Args:
        x (torch.Tensor): Input tensor.
        num_dims (int): Number of dimensions to append.

    Returns:
        torch.Tensor: Tensor with appended dimensions.
    """
    if num_dims <= 0:
        return x
    return x.view(*x.shape, *((1,) * num_dims))  # Reshape to append dimensions

def exists(val):
    """
    Checks if a value exists (is not None).

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists, False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Returns a default value if the given value does not exist.

    Args:
        val: The value to check.
        d: Default value to return if val does not exist.

    Returns:
        The original value if it exists, otherwise the default value.
    """
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    """
    Calculates the amount of padding needed to make a number a multiple of another.

    Args:
        n (int): The number to pad.
        mult (int): The multiple to match.

    Returns:
        int: The padding amount required to make n a multiple of mult.
    """
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder  # Return the required padding

# Scale Normalization class

class ScaleNorm(nn.Module):
    """
    ScaleNorm implements a scaled normalization technique for neural network layers.

    Attributes:
        dim (int): Dimension of the input features.
        eps (float): Small value to prevent division by zero.
        g (nn.Parameter): Learnable parameter for scaling.
    """

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5  # Calculate scale factor
        self.eps = eps  # Set epsilon
        self.g = nn.Parameter(torch.ones(1))  # Initialize scaling parameter

    def forward(self, x):
        """
        Forward pass for the ScaleNorm layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Scaled and normalized output tensor.
        """
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale  # Compute norm
        return x / norm.clamp(min=self.eps) * self.g  # Normalize and scale

# Absolute positional encodings class

class ScaledSinuEmbedding(nn.Module):
    """
    ScaledSinuEmbedding provides sinusoidal positional encodings for inputs.

    Attributes:
        scale (nn.Parameter): Learnable scale factor for the embeddings.
        inv_freq (torch.Tensor): Inverse frequency used for sine and cosine calculations.
    """

    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1,))  # Initialize scale
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))  # Calculate inverse frequency
        self.register_buffer('inv_freq', inv_freq)  # Register as a buffer

    def forward(self, x):
        """
        Forward pass for the ScaledSinuEmbedding layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Positional encoding tensor of shape (batch_size, sequence_length, dim).
        """
        n, device = x.shape[1], x.device  # Extract sequence length and device
        t = torch.arange(n, device=device).type_as(self.inv_freq)  # Create time steps
        # Outer product of t and inv_freq to create sinusoidal embeddings
        # t shape: (n), inv_freq shape: (dim/2) -> sinu shape: (n, dim/2)
        sinu = t.unsqueeze(1) * self.inv_freq.unsqueeze(0)
        emb = torch.cat((sinu.sin(), sinu.cos()), dim=-1)  # Concatenate sine and cosine embeddings
        return emb * self.scale  # Scale the embeddings

class OffsetScale(nn.Module):
    """
    OffsetScale applies learned offsets and scales to the input tensor.

    Attributes:
        gamma (nn.Parameter): Learnable scale parameter for each head.
        beta (nn.Parameter): Learnable offset parameter for each head.
    """

    def __init__(self, dim, heads=1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))  # Initialize scale parameters
        self.beta = nn.Parameter(torch.zeros(heads, dim))  # Initialize offset parameters
        nn.init.normal_(self.gamma, std=0.02)  # Normal initialization for gamma

    def forward(self, x):
        """
        Forward pass for the OffsetScale layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: A list of tensors with applied offsets and scales for each head.
        """
        # x shape: (..., d), gamma/beta shape: (h, d)
        # Unsqueeze x to (..., 1, d) to enable broadcasting with gamma/beta
        # Result shape: (..., h, d)
        out = x.unsqueeze(-2) * self.gamma + self.beta
        return out.unbind(dim=-2)  # Unbind heads into a list

# Feed-Forward Convolutional Module

class FFConvM(nn.Module):
    """
    FFConvM is a feed-forward convolutional module with normalization and dropout.

    Attributes:
        dim_in (int): Input dimension of the features.
        dim_out (int): Output dimension after processing.
        norm_klass (nn.Module): Normalization class to be used.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass=nn.LayerNorm,
        dropout=0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Normalize input
            nn.Linear(dim_in, dim_out),  # Linear transformation
            nn.SiLU(),  # Activation function
            ConvModule(dim_out),  # Convolution module
            nn.Dropout(dropout)  # Apply dropout
        )

    def forward(self, x):
        """
        Forward pass for the FFConvM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        output = self.mdl(x)  # Pass through the model
        return output

class FFM(nn.Module):
    """
    FFM is a feed-forward module with normalization and dropout.

    Attributes:
        dim_in (int): Input dimension of the features.
        dim_out (int): Output dimension after processing.
        norm_klass (nn.Module): Normalization class to be used.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass=nn.LayerNorm,
        dropout=0.1
    ):
        super().__init__()
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Normalize input
            nn.Linear(dim_in, dim_out),  # Linear transformation
            nn.SiLU(),  # Activation function
            nn.Dropout(dropout)  # Apply dropout
        )

    def forward(self, x):
        """
        Forward pass for the FFM module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing.
        """
        output = self.mdl(x)  # Pass through the model
        return output

class FLASH_ShareA_FFConvM(nn.Module):
    """
    Fast Shared Dual Attention Mechanism with feed-forward convolutional blocks.
    Published in paper: "MossFormer: Pushing the Performance Limit of Monaural Speech Separation
    using Gated Single-Head Transformer with Convolution-Augmented Joint Self-Attentions", ICASSP 2023.
    (https://arxiv.org/abs/2302.11824)

    Args:
        dim (int): Input dimension.
        group_size (int, optional): Size of groups for processing. Defaults to 256.
        query_key_dim (int, optional): Dimension of the query and key. Defaults to 128.
        expansion_factor (float, optional): Factor to expand the hidden dimension. Defaults to 1.
        causal (bool, optional): Whether to use causal masking. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
        rotary_pos_emb (optional): Rotary positional embeddings for attention. Defaults to None.
        norm_klass (callable, optional): Normalization class to use. Defaults to nn.LayerNorm.
        shift_tokens (bool, optional): Whether to shift tokens for attention calculation. Defaults to True.
    """

    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # Initialize positional embeddings, dropout, and projections
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)

        # Feed-forward layers
        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        # Offset and scale for query and key
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

        self.gateActivate = nn.Sigmoid()

        self.inv_g = float(1.0 / self.group_size)

        self.pad_A = torch.zeros((1, self.group_size, 128), dtype=torch.int8)

        self.pad_B = torch.zeros((1, self.group_size, 1024), dtype=torch.int8)

    def forward(self, x, n, inv_n, *, mask=None):
        """
        Forward pass for FLASH layer.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, features).
            mask (Tensor, optional): Mask for attention. Defaults to None.

        Returns:
            Tensor: Output tensor after applying attention and projections.
        """

        # Pre-normalization step
        normed_x = x
        residual = x  # Save residual for skip connection

        # Token shifting if enabled
        if self.shift_tokens:
            x_shift, x_pass = normed_x.chunk(2, dim=-1)
            x_shift = F.pad(x_shift, (0, 0, 1, -1), value=0.)
            normed_x = torch.cat((x_shift, x_pass), dim=-1)

        # Initial projections. Note: v, u, qk are not padded here.
        v, u = self.to_hidden(normed_x).chunk(2, dim=-1)
        qk = self.to_qk(normed_x)

        # Offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        # The attention function handles its own padding and grouping internally.
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, n, inv_n, mask=mask)

        # Output calculation with gating. v and u here are the original unpadded tensors.
        # att_v and att_u are returned with the original sequence length.
        out = (att_u * v) * self.gateActivate(att_v * u)
        x = residual + self.to_out(out)  # Residual connection
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, n, inv_n, mask=None):
        """
        Calculate attention output using quadratic and linear attention mechanisms.
        This function replaces einsum and rearrange with basic tensor operations.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, features).
            quad_q, lin_q, quad_k, lin_k (Tensor): Query and key representations.
            v, u (Tensor): Value representations.
            mask (Tensor, optional): Mask for attention. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: Attention outputs for v and u.
        """
        # Rotate queries and keys with rotary positional embeddings
        if exists(self.rotary_pos_emb):
            quad_q, lin_q, quad_k, lin_k = map(self.rotary_pos_emb.rotate_queries_or_keys,
                                               (quad_q, lin_q, quad_k, lin_k))

        # Padding for group processing. This is required for the reshape operation.
        padding = padding_to_multiple_of(n, self.group_size)
        if padding > 0:
            pad = self.pad_A[:, :padding].float()
            quad_q = torch.cat([quad_q, pad], dim=1)
            quad_k = torch.cat([quad_k, pad], dim=1)
            lin_q = torch.cat([lin_q, pad], dim=1)
            lin_k = torch.cat([lin_k, pad], dim=1)
            pad = self.pad_B[:, :padding].float()
            v = torch.cat([v, pad], dim=1)
            u = torch.cat([u, pad], dim=1)

        padded_len = n + padding
        num_groups = padded_len // self.group_size

        # Group along sequence for attention using reshape
        # Reshapes (b, padded_len, d) -> (b, num_groups, group_size, d)
        def group(t):
            return t.reshape(1, num_groups, self.group_size, -1)

        quad_q, quad_k, lin_q, lin_k, v, u = map(group, (quad_q, quad_k, lin_q, lin_k, v, u))

        # --- Quadratic Attention ---
        # Similarity matrix via batch matrix multiplication
        # quad_q shape: (b, num_groups, g, d_qk), quad_k shape: (b, num_groups, g, d_qk)
        # Transpose last two dims of k for matmul: (b, num_groups, d_qk, g)
        sim = torch.matmul(quad_q, quad_k.transpose(-1, -2)) * self.inv_g

        attn = F.relu(sim) ** 2

        # If padding was added, directly zero out the corresponding columns
        # in the last group of the attention matrix. This is the most efficient method.
        if padding > 0:
            # This is an in-place operation that modifies the `attn` tensor directly.
            # It selects all batches, the last group, all query rows, and the last `padding` key columns,
            # and sets their values to 0.
            attn[:, -1, :, -padding:] = 0

        # Apply attention to values
        # attn shape: (b, num_groups, g, g), v/u shape: (b, num_groups, g, d_v)
        quad_out_v = torch.matmul(attn, v)
        quad_out_u = torch.matmul(attn, u)

        # --- Linear Attention ---

        # Flatten k and v to combine group and sequence dims for a single matmul
        # k: (b, num_groups, g, d_qk) -> (b, d_qk, padded_len)
        lin_k_flat = lin_k.permute(0, 3, 1, 2).reshape(1, 1, -1, padded_len)
        # v: (b, num_groups, g, d_v) -> (b, padded_len, d_v)
        v_flat = v.reshape(1, 1, padded_len, -1)
        # Matmul flat k with flat v: (b, d_qk, padded_len) @ (b, padded_len, d_v) -> (b, d_qk, d_v)
        lin_kv = torch.matmul(lin_k_flat, v_flat) * inv_n

        # Matmul q with kv. Unsqueeze kv to broadcast across groups.
        # (b, num_groups, g, d_qk) @ (b, 1, d_qk, d_v) -> (b, num_groups, g, d_v)
        lin_out_v = torch.matmul(lin_q, lin_kv)

        # Repeat for u
        u_flat = u.reshape(1, 1, padded_len, -1)
        lin_ku = torch.matmul(lin_k_flat, u_flat) * inv_n
        lin_out_u = torch.matmul(lin_q, lin_ku)

        # Combine quadratic and linear attention outputs
        att_v = quad_out_v + lin_out_v
        att_u = quad_out_u + lin_out_u

        # Ungroup and trim padding
        # Reshapes (b, num_groups, g, d) -> (b, padded_len, d) and then trims
        def ungroup_and_trim(t):
            unpadded = t.reshape(1, padded_len, -1)
            return unpadded[:, :n, :]

        return ungroup_and_trim(att_v), ungroup_and_trim(att_u)

class Gated_FSMN(nn.Module):
    """
    Gated Frequency Selective Memory Network (FSMN) class.

    This class implements a gated FSMN that combines two feedforward
    convolutional networks with a frequency selective memory module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        lorder (int): Order of the filter for FSMN.
        hidden_size (int): Number of hidden units in the network.
    """
    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        # Feedforward network for the first branch (u)
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        # Feedforward network for the second branch (v)
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        # Frequency selective memory network
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def forward(self, x):
        """
        Forward pass for the Gated FSMN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, sequence_length).

        Returns:
            Tensor: Output tensor after applying gated FSMN operations.
        """
        input_residual = x
        x_u = self.to_u(x)  # Process input through the first branch
        x_v = self.to_v(x)  # Process input through the second branch
        x_u = self.fsmn(x_u)  # Apply FSMN to the output of the first branch
        x = x_v * x_u + input_residual  # Combine outputs with the original input
        return x


class Gated_FSMN_Block(nn.Module):
    """
    A 1-D convolutional block that incorporates a gated FSMN.

    This block consists of two convolutional layers, followed by a
    gated FSMN and normalization layers.

    Args:
        dim (int): Dimensionality of the input.
        inner_channels (int): Number of channels in the inner layers.
        group_size (int): Size of the groups for normalization.
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm').
    """
    def __init__(self, dim, inner_channels=256, group_size=256, norm_type='scalenorm'):
        super(Gated_FSMN_Block, self).__init__()
        # Choose normalization class based on the provided type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # First convolutional layer with PReLU activation
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, inner_channels, kernel_size=1),
            nn.PReLU(),
        )
        self.norm1 = CLayerNorm(inner_channels)  # Normalization after first convolution
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)  # Gated FSMN layer
        self.norm2 = CLayerNorm(inner_channels)  # Normalization after FSMN
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)  # Final convolutional layer

    def forward(self, input):
        """
        Forward pass for the Gated FSMN Block.

        Args:
            input (Tensor): Input tensor of shape (batch_size, seq_len, dim).

        Returns:
            Tensor: Output tensor after processing through the block.
        """
        # input shape: (b, n, d) -> transpose to (b, d, n) for Conv1d
        conv1 = self.conv1(input.transpose(1, 2))
        norm1 = self.norm1(conv1)
        # transpose to (b, n, d) for FSMN
        seq_out = self.gated_fsmn(norm1.transpose(1, 2))
        # transpose back to (b, d, n) for normalization and Conv1d
        norm2 = self.norm2(seq_out.transpose(1, 2))
        conv2 = self.conv2(norm2)
        # transpose final output to (b, n, d) and add residual
        return conv2.transpose(1, 2) + input


class MossformerBlock_GFSMN(nn.Module):
    """
    Mossformer Block with Gated FSMN.

    This block combines attention mechanisms and gated FSMN layers
    to process input sequences.

    Args:
        dim (int): Dimensionality of the input.
        depth (int): Number of layers in the block.
        group_size (int): Size of the groups for normalization.
        query_key_dim (int): Dimension of the query and key in attention.
        expansion_factor (float): Expansion factor for feedforward layers.
        causal (bool): If True, enables causal attention.
        attn_dropout (float): Dropout rate for attention layers.
        norm_type (str): Type of normalization to use ('scalenorm' or 'layernorm').
        shift_tokens (bool): If True, shifts tokens in the attention layer.
    """
    def __init__(self, *, dim, depth, group_size=256, query_key_dim=128, expansion_factor=4., causal=False, attn_dropout=0.1, norm_type='scalenorm', shift_tokens=True):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # Rotary positional embedding for attention
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        # Create a list of Gated FSMN blocks
        self.fsmn = nn.ModuleList([Gated_FSMN_Block(dim) for _ in range(depth)])

        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Builds repeated UniDeep FSMN layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            lorder (int): Order of the filter for FSMN.
            hidden_size (int): Number of hidden units.
            repeats (int): Number of repetitions.

        Returns:
            Sequential: A sequential container with repeated layers.
        """
        repeats_list = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for i in range(repeats)
        ]
        return nn.Sequential(*repeats_list)

    def forward(self, x, n, *, mask=None):
        """
        Forward pass for the Mossformer Block with Gated FSMN.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Tensor, optional): Mask tensor for attention operations.

        Returns:
            Tensor: Output tensor after processing through the block.
        """
        inv_n = float(1.0 / n)
        for i, flash in enumerate(self.layers):  # Process through each layer
            x = flash(x, n, inv_n, mask=mask)
            x = self.fsmn[i](x)  # Apply corresponding Gated FSMN block

        return x


class MossformerBlock(nn.Module):
    """
    Mossformer Block with attention mechanisms.

    This block is designed to process input sequences using attention
    layers and incorporates rotary positional embeddings. It allows
    for configurable normalization types and can handle causal
    attention.

    Args:
        dim (int): Dimensionality of the input.
        depth (int): Number of attention layers in the block.
        group_size (int, optional): Size of groups for normalization. Default is 256.
        query_key_dim (int, optional): Dimension of the query and key in attention. Default is 128.
        expansion_factor (float, optional): Expansion factor for feedforward layers. Default is 4.
        causal (bool, optional): If True, enables causal attention. Default is False.
        attn_dropout (float, optional): Dropout rate for attention layers. Default is 0.1.
        norm_type (str, optional): Type of normalization to use ('scalenorm' or 'layernorm'). Default is 'scalenorm'.
        shift_tokens (bool, optional): If True, shifts tokens in the attention layer. Default is True.
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.0,
        causal=False,
        attn_dropout=0.1,
        norm_type='scalenorm',
        shift_tokens=True
    ):
        super().__init__()

        # Ensure normalization type is valid
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        # Select normalization class based on the provided type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size  # Group size for normalization

        # Rotary positional embedding for attention
        # Max rotary embedding dimensions of 32, partial Rotary embeddings, from Wang et al - GPT-J
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = nn.ModuleList([
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ])

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Builds repeated UniDeep FSMN layers.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            lorder (int): Order of the filter for FSMN.
            hidden_size (int): Number of hidden units.
            repeats (int, optional): Number of repetitions. Default is 1.

        Returns:
            Sequential: A sequential container with repeated layers.
        """
        repeats_list = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for _ in range(repeats)
        ]
        return nn.Sequential(*repeats_list)

    def forward(self, x, n, *, mask=None):
        """
        Forward pass for the Mossformer Block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, dim).
            mask (Tensor, optional): Mask tensor for attention operations.

        Returns:
            Tensor: Output tensor after processing through the block.
        """
        # Process input through each attention layer
        for flash in self.layers:
            x = flash(x, n, mask=mask)  # Apply attention layer with optional mask
        
        return x  # Return the final output tensor
