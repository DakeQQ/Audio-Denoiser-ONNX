"""
modified from https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/lobes/models/dual_path.py
#Author: Shengkui Zhao

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
    """This class implements the MossFormer2 block.

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

class MossFormerM2(nn.Module):
    """This class implements the MossFormer block.

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
    """
    Computation block for single-path processing.

    This block performs single-path processing using an intra-model (e.g., 
    MossFormerM) to process input data both within chunks and the full sequence
    allowing for flexibility in normalization and skip connections. 

    Arguments
    ---------
    num_blocks : int
        Number of blocks to use in the intra model.
    out_channels : int
        Dimensionality of the inter/intra model.
    norm : str, optional
        Normalization type. Default is 'ln' for Layer Normalization.
    skip_around_intra : bool, optional
        If True, adds a skip connection around the intra layer. Default is True.

    Example
    ---------
        >>> comp_block = Computation_Block(num_blocks=64, out_channels=64)
        >>> x = torch.randn(10, 64, 100)  # Sample input tensor
        >>> x = comp_block(x)  # Process through the computation block
        >>> x.shape  # Output shape
        torch.Size([10, 64, 100])
    """

    def __init__(
        self,
        num_blocks: int,
        out_channels: int,
        norm: str = "ln",
        skip_around_intra: bool = True,
    ):
        """
        Initializes the Computation_Block.

        Args:
            num_blocks (int): Number of blocks for the intra model.
            out_channels (int): Dimensionality of the output features.
            norm (str, optional): Normalization type. Defaults to 'ln'.
            skip_around_intra (bool, optional): If True, use skip connection 
                                                 around the intra layer. Defaults to True.
        """
        super(Computation_Block, self).__init__()

        # Initialize the intra-model (MossFormerM with recurrence)
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra  # Flag for skip connection

        # Set normalization type
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)  # Initialize normalization layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output tensor.

        Args:
            x (torch.Tensor): Input tensor of dimension [B, N, S], where:
                B = Batch size,
                N = Number of filters,
                S = Sequence length.

        Returns:
            out (torch.Tensor): Output tensor of dimension [B, N, S].
        """
        B, N, S = x.shape  # Get the shape of the input tensor
        
        # Permute to change the tensor shape from [B, N, S] to [B, S, N] for processing
        intra = x.permute(0, 2, 1).contiguous()

        # Process through the intra model
        intra = self.intra_mdl(intra)

        # Permute back to [B, N, S]
        intra = intra.permute(0, 2, 1).contiguous()
        
        # Apply normalization if specified
        if self.norm is not None:
            intra = self.intra_norm(intra)

        # Add skip connection around the intra layer if enabled
        if self.skip_around_intra:
            intra = intra + x

        out = intra  # Set the output tensor
        return out  # Return the processed output tensor

class MossFormer_MaskNet(nn.Module):
    """
    The MossFormer MaskNet for predicting masks for encoder output features.
    This implementation uses an upgraded MaskNet structure based on the 
    MossFormer2 model.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that will be input to the intra and inter blocks.
    num_blocks : int
        Number of layers in the Dual Computation Block.
    norm : str
        Type of normalization to apply.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        If True, adds skip connections around the intra layers.
    use_global_pos_enc : bool
        If True, utilizes global positional encodings.
    max_length : int
        Maximum sequence length for input data.

    Example
    ---------
        >>> mossformer_masknet = MossFormer_MaskNet(64, 64, num_spks=2)
        >>> x = torch.randn(10, 64, 2000)  # Sample input tensor
        >>> x = mossformer_masknet(x)  # Process through the MaskNet
        >>> x.shape  # Output shape
        torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 24,
        norm: str = "ln",
        num_spks: int = 2,
        skip_around_intra: bool = True,
        use_global_pos_enc: bool = True,
        max_length: int = 20000,
    ):
        """
        Initializes the MossFormer_MaskNet.

        Args:
            in_channels (int): Number of input channels from the encoder.
            out_channels (int): Number of output channels to be used in the 
                                computation blocks.
            num_blocks (int): Number of layers for the Dual Computation Block. Default is 24.
            norm (str): Type of normalization to apply. Default is 'ln'.
            num_spks (int): Number of speakers. Default is 2.
            skip_around_intra (bool): If True, adds skip connections around intra layers. Default is True.
            use_global_pos_enc (bool): If True, enables global positional encoding. Default is True.
            max_length (int): Maximum sequence length. Default is 20000.
        """
        super(MossFormer_MaskNet, self).__init__()
        
        self.num_spks = num_spks  # Store number of speakers
        self.num_blocks = num_blocks  # Store number of computation blocks
        
        # Initialize normalization layer based on the provided type
        self.norm = select_norm(norm, in_channels, 3)
        
        # 1D Convolutional layer to project input channels to output channels
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        
        self.use_global_pos_enc = use_global_pos_enc  # Flag for global positional encoding
        if self.use_global_pos_enc:
            # Initialize positional encoding layer
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        # Initialize the computation block for processing features
        self.mdl = Computation_Block(
            num_blocks,
            out_channels,
            norm,
            skip_around_intra=skip_around_intra,
        )

        # Output layer to project features to the desired number of speaker outputs
        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.conv1_decoder = nn.Conv1d(out_channels, in_channels, 1, bias=False)  # Decoder layer

        self.prelu = nn.PReLU()  # PReLU activation
        self.activation = nn.ReLU()  # ReLU activation for final output
        
        # Gated output layer to refine predictions
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the output tensor.

        Args:
            x (torch.Tensor): Input tensor of dimension [B, N, S], where:
                B = Batch size,
                N = Number of channels (filters),
                S = Sequence length.

        Returns:
            out (torch.Tensor): Output tensor of dimension [spks, B, N, S], 
                                where:
                spks = Number of speakers,
                B = Batch size,
                N = Number of filters,
                S = Number of time frames.
        """
        # [B, N, L] - Normalize the input tensor
        x = self.norm(x)

        # [B, N, L] - Apply 1D convolution to encode features
        x = self.conv1d_encoder(x)
        
        # If using global positional encoding, add the positional embeddings
        if self.use_global_pos_enc:
            base = x  # Store the original encoded features
            x = x.transpose(1, -1)  # Change shape to [B, L, N]
            emb = self.pos_enc(x)  # Get positional embeddings
            emb = emb.transpose(0, -1)  # Change shape back to [B, N, L]
            x = base + emb  # Add positional embeddings to encoded features

        # [B, N, S] - Process through the computation block
        x = self.mdl(x)
        x = self.prelu(x)  # Apply PReLU activation

        # [B, N*spks, S] - Project features to multiple speaker outputs
        x = self.conv1d_out(x)
        B, _, S = x.shape  # Get the shape after convolution

        # [B*spks, N, S] - Reshape for speaker outputs
        x = x.view(B * self.num_spks, -1, S)

        # [B*spks, N, S] - Apply gated output layer
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, S] - Decode back to original channel size
        x = self.conv1_decoder(x)

        # [B, spks, N, S] - Reshape output tensor to include speaker dimension
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)  # Apply ReLU activation

        # [spks, B, N, S] - Transpose to match output format
        x = x.transpose(0, 1)

        return x  # Return the output tensor

class MossFormer(nn.Module):
    """
    The End-to-End (E2E) Encoder-MaskNet-Decoder MossFormer model for speech separation.
    This implementation is based on the upgraded MaskNet architecture from the MossFormer2 model.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that will be input to the MossFormer2 blocks.
    num_blocks : int
        Number of layers in the Dual Computation Block.
    kernel_size : int
        Kernel size for the convolutional layers in the encoder and decoder.
    norm : str
        Type of normalization to apply (e.g., 'ln' for layer normalization).
    num_spks : int
        Number of sources (speakers) to separate.
    skip_around_intra : bool
        If True, adds skip connections around intra layers in the computation blocks.
    use_global_pos_enc : bool
        If True, uses global positional encodings in the model.
    max_length : int
        Maximum sequence length for input data.

    Example
    ---------
        >>> mossformer = MossFormer(num_spks=2)
        >>> x = torch.randn(1, 10000)  # Sample input tensor
        >>> outputs = mossformer(x)  # Process the input through the model
        >>> outputs[0].shape  # Output shape for first speaker
        torch.Size([1, 10000])
    """

    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        """
        Initializes the MossFormer model.

        Args:
            in_channels (int): Number of input channels from the encoder. Default is 512.
            out_channels (int): Number of output channels for the MaskNet blocks. Default is 512.
            num_blocks (int): Number of layers in the Dual Computation Block. Default is 24.
            kernel_size (int): Kernel size for convolutional layers. Default is 16.
            norm (str): Type of normalization to apply. Default is 'ln'.
            num_spks (int): Number of speakers to separate. Default is 2.
            skip_around_intra (bool): If True, adds skip connections around intra layers. Default is True.
            use_global_pos_enc (bool): If True, uses global positional encoding. Default is True.
            max_length (int): Maximum sequence length. Default is 20000.
        """
        super(MossFormer, self).__init__()
        self.num_spks = num_spks  # Store number of speakers
        
        # Initialize the encoder with 1 input channel and the specified output channels
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels, in_channels=1)

        # Initialize the MaskNet with the specified parameters
        self.mask_net = MossFormer_MaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        
        # Initialize the decoder to project output back to 1 channel
        self.dec = Decoder(
            in_channels=out_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            bias=False
        )

    def forward(self, input: torch.Tensor) -> list:
        """Processes the input through the encoder, mask net, and decoder.

        Args:
            input (torch.Tensor): Input tensor of shape [B, T], where B is the batch size and T is the input length.

        Returns:
            out (list): List of output tensors for each speaker, each of shape [B, T].
        """
        # Pass the input through the encoder to extract features
        x = self.enc(input)

        # Generate the mask for each speaker using the mask net
        mask = self.mask_net(x)

        # Duplicate the features for each speaker
        x = torch.stack([x] * self.num_spks)

        # Apply the mask to separate the sources
        sep_x = x * mask

        # Decoding process to reconstruct the separated sources
        est_source = torch.cat(
            [self.dec(sep_x[i]).unsqueeze(-1) for i in range(self.num_spks)],
            dim=-1,
        )

        # Match the estimated output length to the original input length
        T_origin = input.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))  # Pad if estimated length is shorter
        else:
            est_source = est_source[:, :T_origin, :]  # Trim if estimated length is longer

        out = []
        # Collect outputs for each speaker
        for spk in range(self.num_spks):
            out.append(est_source[:, :, spk])
        
        return out  # Return list of separated outputs


class MossFormer2_SS_16K(nn.Module):
    """
    Wrapper for the MossFormer2 model, facilitating external calls.

    Arguments
    ---------
    args : Namespace
        Contains the necessary arguments for initializing the MossFormer model, such as:
        - encoder_embedding_dim: Dimension of the encoder's output embeddings.
        - mossformer_sequence_dim: Dimension of the MossFormer sequence.
        - num_mossformer_layer: Number of layers in the MossFormer.
        - encoder_kernel_size: Kernel size for the encoder.
        - num_spks: Number of sources (speakers) to separate.
    """

    def __init__(self, args):
        """
        Initializes the MossFormer2_SS_16K wrapper.

        Args:
            args (Namespace): Contains configuration parameters for the model.
        """
        super(MossFormer2_SS_16K, self).__init__()
        # Initialize the main MossFormer model with parameters from args
        self.model = MossFormer(
            in_channels=args.encoder_embedding_dim,
            out_channels=args.mossformer_sequence_dim,
            num_blocks=args.num_mossformer_layer,
            kernel_size=args.encoder_kernel_size,
            norm="ln",
            num_spks=args.num_spks,
            skip_around_intra=True,
            use_global_pos_enc=True,
            max_length=20000
        )

    def forward(self, x: torch.Tensor) -> list:
        """Processes the input through the MossFormer model.

        Args:
            x (torch.Tensor): Input tensor of shape [B, T], where B is the batch size and T is the input length.

        Returns:
            outputs (list): List of output tensors for each speaker.
        """
        outputs = self.model(x)  # Forward pass through the MossFormer model
        return outputs  # Return the list of outputs
