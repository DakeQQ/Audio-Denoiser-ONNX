"""
modified from https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py
Author: Shengkui Zhao
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .mossformer2_block import ScaledSinuEmbedding, MossformerBlock_GFSMN, MossformerBlock


EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x


class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization.

       Arguments
       ---------
       dim : int
        Dimension that you want to normalize.
       elementwise_affine : True
        Learnable per-element affine parameters.

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> CLN = CumulativeLayerNorm(10)
    >>> x_norm = CLN(x)
    """

    def __init__(self, dim, elementwise_affine=True):
        super(CumulativeLayerNorm, self).__init__(
            dim, elementwise_affine=elementwise_affine, eps=1e-8
        )

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor size [N, C, K, S] or [N, C, L]
        """
        # x: N x C x K x S or N x C x L
        # N x K x S x C
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1).contiguous()
            # N x K x S x C == only channel norm
            x = super().forward(x)
            # N x C x K x S
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dim() == 3:
            x = torch.transpose(x, 1, 2)
            # N x L x C == only channel norm
            x = super().forward(x)
            # N x C x L
            x = torch.transpose(x, 1, 2)
        return x


def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type.
    """

    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        return nn.GroupNorm(1, dim, eps=1e-8)
    else:
        return nn.BatchNorm1d(dim)


class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x


class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x


class IdentityBlock:
    """This block is used when we want to have identity transformation within the Dual_path block.

    Example
    -------
    >>> x = torch.randn(10, 100)
    >>> IB = IdentityBlock()
    >>> xhat = IB(x)
    """

    def _init__(self, **kwargs):
        pass

    def __call__(self, x):
        return x


class MossFormerM(nn.Module):
    """This class implements the transformer encoder based on MossFormer2 layers.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer2 blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size for segmenting sequence
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = MossFormerM(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        attn_dropout = 0.1
    ):
        super().__init__()

        self.mossformerM = MossformerBlock_GFSMN(
                           dim=d_model,
                           depth=num_blocks,
                           group_size=group_size,
                           query_key_dim=query_key_dim,
                           expansion_factor=expansion_factor,
                           causal=causal,
                           attn_dropout=attn_dropout
                              )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    def forward(
        self,
        src,
            n
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = self.mossformerM(src, n)
        output = self.norm(output)

        return output

class MossFormerM2(nn.Module):
    """This class implements the transformer encoder.

    Arguments
    ---------
    num_blocks : int
        Number of mossformer blocks to include.
    d_model : int
        The dimension of the input embedding.
    attn_dropout : float
        Dropout for the self-attention (Optional).
    group_size: int
        the chunk size
    query_key_dim: int
        the attention vector dimension
    expansion_factor: int
        the expansion factor for the linear projection in conv module
    causal: bool
        true for causal / false for non causal

    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = MossFormerM2(num_blocks=8, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size = 256,
        query_key_dim = 128,
        expansion_factor = 4.,
        attn_dropout = 0.1
    ):
        super().__init__()

        self.mossformerM = MossformerBlock(
                           dim=d_model,
                           depth=num_blocks,
                           group_size=group_size,
                           query_key_dim=query_key_dim,
                           expansion_factor=expansion_factor,
                           causal=causal,
                           attn_dropout=attn_dropout
                              )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        src,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """
        output = self.mossformerM(src)
        output = self.norm(output)

        return output

class Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
     out_channels : int
        Dimensionality of model output.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.

    Example
    ---------
        >>> comp_block = Computation_Block(64)
        >>> x = torch.randn(10, 64, 100)
        >>> x = comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100])
    """

    def __init__(
        self,
        num_blocks,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super(Computation_Block, self).__init__()

        ##Default MossFormer2 model
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        ##The previous MossFormer model
        #self.intra_mdl = MossFormerM2(num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def forward(self, x, n):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, S].
            where, B = Batchsize,
               N = number of filters
               S = sequence time index 
        """
        # B, N, S = x.shape
        # [B, S, N]
        intra = x.permute(0, 2, 1).contiguous() 

        intra = self.intra_mdl(intra, n)

        # [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # [B, N, S]
        if self.skip_around_intra:
            intra = intra + x

        return intra

class MossFormer_MaskNet(nn.Module):
    """
    The MossFormer MaskNet for mask prediction.

    This class is designed for predicting masks used in source separation tasks.
    It processes input tensors through various layers including convolutional layers, 
    normalization, and a computation block to produce the final output.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the MossFormer2 blocks.
    out_channels_final : int
        Number of channels that are finally outputted.
    num_blocks : int
        Number of layers in the Dual Computation Block.
    norm : str
        Normalization type ('ln' for LayerNorm, 'bn' for BatchNorm, etc.).
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        If True, applies skip connections around intra-block connections.
    use_global_pos_enc : bool
        If True, uses global positional encodings.
    max_length : int
        Maximum sequence length for input tensors.

    Example
    ---------
    >>> mossformer_masknet = MossFormer_MaskNet(64, 64, out_channels_final=8, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)  # Example input
    >>> x = mossformer_masknet(x)  # Forward pass
    >>> x.shape  # Expected output shape
    torch.Size([10, 2, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        out_channels_final,
        num_blocks=24,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super(MossFormer_MaskNet, self).__init__()
        
        # Initialize instance variables
        self.num_spks = num_spks  # Number of sources
        self.num_blocks = num_blocks  # Number of computation blocks
        self.norm = select_norm(norm, in_channels, 3)  # Select normalization type
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)  # Encoder convolutional layer
        self.use_global_pos_enc = use_global_pos_enc  # Flag for global positional encoding

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)  # Initialize positional embedding

        # Define the computation block
        self.mdl = Computation_Block(
            num_blocks,
            out_channels,
            norm,
            skip_around_intra=skip_around_intra,
        )

        # Output layers
        self.conv1d_out = nn.Conv1d(out_channels, out_channels * num_spks, kernel_size=1)  # For multiple speakers
        self.conv1_decoder = nn.Conv1d(out_channels, out_channels_final, 1, bias=False)  # Decoder layer
        self.prelu = nn.PReLU()  # Activation function
        self.activation = nn.ReLU()  # Final activation function

        # Gated output layers
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), 
            nn.Tanh()  # Non-linear activation
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), 
            nn.Sigmoid()  # Gating mechanism
        )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, S], where B is the batch size, 
            N is the number of channels, and S is the sequence length.

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [B, spks, N, S], where spks is the number of sources 
            (speakers) and is ordered such that the first index corresponds to the target speech.
        """

        # Normalize the input
        # [B, N, L]
        x = self.norm(x)

        # Apply encoder convolution
        # [B, N, L]
        x = self.conv1d_encoder(x)

        if self.use_global_pos_enc:
            base = x  # Store the base for adding positional embedding
            x = x.transpose(1, -1)  # Change shape to [B, L, N] for positional encoding
            emb = self.pos_enc(x)  # Get positional embeddings
            emb = emb.transpose(0, -1)  # Change back to [B, N, L]
            x = base + emb  # Add positional embeddings to the base

        # Process through the computation block
        # [B, N, S]
        x = self.mdl(x, x.shape[-1])
        x = self.prelu(x)  # Apply activation

        # Expand to multiple speakers
        # [B, N*spks, S]
        x = self.conv1d_out(x)
        B, _, S = x.shape  # Unpack the batch size and sequence length

        # Reshape to [B*spks, N, S]
        # This prepares the output for gating
        # [B*spks, N, S]
        x = x.view(B * self.num_spks, -1, S)

        # Apply gated output layers
        # [B*spks, N, S]
        x = self.output(x) * self.output_gate(x)  # Element-wise multiplication for gating

        # Decode to final output
        # [B*spks, N, S]
        x = self.conv1_decoder(x)

        # Reshape to [B, spks, N, S] for output
        # [B, spks, N, S]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)  # Final reshaping for output
        x = self.activation(x)  # Apply final activation

        # Transpose to [spks, B, N, S] for output
        # return the 1st spk signal as the target speech
        x = x.transpose(0, 1)
        return x[0].transpose(1, 2)  # Return only the first speaker's signal
