import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging.version import parse as V
from torch.nn import init
from torch.nn.parameter import Parameter

from .fsmn import UniDeepFsmn
from .conv_module import ConvModule
from .mossformer import MossFormer
from .se_layer import SELayer
from .get_layer_from_string import get_layer
from .discriminator import Discriminator

# Check if the installed version of PyTorch is 1.9.0 or higher
is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class MossFormerGAN_SE_16K(nn.Module):
    """
    MossFormerGAN_SE_16K: A GAN-based speech enhancement model for 16kHz input audio.

    This model integrates a synchronous attention network (SyncANet) for 
    feature extraction. Depending on the mode (train or inference), it may 
    also include a discriminator for adversarial training.

    Args:
        args (Namespace): Arguments containing configuration parameters, 
                          including 'fft_len' and 'mode'.
    """

    def __init__(self, args):
        """Initializes the MossFormerGAN_SE_16K model."""
        super(MossFormerGAN_SE_16K, self).__init__()

        # Initialize SyncANet with specified number of channels and features
        self.model = SyncANet(num_channel=64, num_features=args.fft_len // 2 + 1)

        # Initialize discriminator if in training mode
        if args.mode == 'train':
            self.discriminator = Discriminator(ndf=16)
        else:
            self.discriminator = None

    def forward(self, x):
        """
        Defines the forward pass of the MossFormerGAN_SE_16K model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_channels, height, width].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensors representing the real and imaginary parts.
        """
        output_real, output_imag = self.model(x)  # Get real and imaginary outputs from the model
        return output_real, output_imag  # Return the outputs


class FSMN_Wrap(nn.Module):
    """
    FSMN_Wrap: A wrapper around the UniDeepFsmn module to facilitate 
    integration into the larger model architecture.

    Args:
        nIn (int): Number of input features.
        nHidden (int): Number of hidden features in the FSMN (default is 128).
        lorder (int): Order of the FSMN (default is 20).
        nOut (int): Number of output features (default is 128).
    """

    def __init__(self, nIn, nHidden=128, lorder=20, nOut=128):
        """Initializes the FSMN_Wrap module with specified parameters."""
        super(FSMN_Wrap, self).__init__()

        # Initialize the UniDeepFsmn module
        self.fsmn = UniDeepFsmn(nIn, nHidden, lorder, nHidden)

    def forward(self, x):
        """
        Defines the forward pass of the FSMN_Wrap module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, time, 2].

        Returns:
            torch.Tensor: Output tensor reshaped to [batch_size, channels, height, time].
        """
        # Shape of input x: [b, c, h, T, 2]

        # Permute x to reshape it for FSMN processing: [b, T, h, c]

        # Pass through the FSMN
        output = self.fsmn(x.permute(0, 2, 3, 1))  # output: [b*T, h, c]

        return output.permute(0, 3, 1, 2)  # Final output shape: [b, c, h, T]


class DilatedDenseNet(nn.Module):
    """
    DilatedDenseNet: A dilated dense network for feature extraction.

    This network consists of a series of dilated convolutions organized in a dense block structure,
    allowing for efficient feature reuse and capturing multi-scale information.

    Args:
        depth (int): The number of layers in the dense block (default is 4).
        in_channels (int): The number of input channels for the first layer (default is 64).
    """

    def __init__(self, depth=4, in_channels=64):
        """Initializes the DilatedDenseNet with specified depth and input channels."""
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)  # Padding for the first layer
        self.twidth = 2  # Temporal width for convolutions
        self.kernel_size = (self.twidth, 3)  # Kernel size for convolutions

        # Initialize dilated convolutions, padding, normalization, and FSMN for each layer
        for i in range(self.depth):
            dil = 2 ** i  # Dilation factor for the current layer
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1  # Calculate padding length
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))  # Convolution layer
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))  # Normalization
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))  # Activation function
            setattr(self, 'fsmn{}'.format(i + 1),
                    FSMN_Wrap(nIn=self.in_channels, nHidden=self.in_channels, lorder=5, nOut=self.in_channels))

    def forward(self, x):
        """
        Defines the forward pass for the DilatedDenseNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output tensor after processing through the dense network.
        """
        skip = x  # Initialize skip connection with input
        for i in range(self.depth):
            # Apply padding, convolution, normalization, activation, and FSMN in sequence
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            out = getattr(self, 'fsmn{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)  # Concatenate outputs for dense connectivity
        return out  # Return the final output


class DenseEncoder(nn.Module):
    """
    DenseEncoder: A dense encoding module for feature extraction from input data.

    This module consists of a series of convolutional layers followed by a 
    dilated dense network for robust feature learning.

    Args:
        in_channel (int): Number of input channels for the encoder.
        channels (int): Number of output channels for each convolutional layer (default is 64).
    """

    def __init__(self, in_channel, channels=64):
        """Initializes the DenseEncoder with specified input channels and feature size."""
        super(DenseEncoder, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),  # Initial convolution layer
            nn.InstanceNorm2d(channels, affine=True),  # Normalization layer
            nn.PReLU(channels)  # Activation function
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)  # Dilated Dense Network
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),  # Second convolution layer
            nn.InstanceNorm2d(channels, affine=True),  # Normalization layer
            nn.PReLU(channels)  # Activation function
        )

    def forward(self, x):
        """
        Defines the forward pass for the DenseEncoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channel, height, width].

        Returns:
            torch.Tensor: Output tensor after processing through the encoder.
        """
        x = self.conv_1(x)  # Process through the first convolutional layer
        x = self.dilated_dense(x)  # Process through the dilated dense network
        x = self.conv_2(x)  # Process through the second convolutional layer
        return x  # Return the final output


class SPConvTranspose2d(nn.Module):
    """
    SPConvTranspose2d: A spatially separable convolution transpose layer.

    This module implements a transposed convolution operation with spatial separability,
    allowing for efficient upsampling and feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the convolution kernel.
        r (int): Upsampling rate (default is 1).
    """

    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        """Initializes the SPConvTranspose2d with specified parameters."""
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)  # Padding for input
        self.out_channels = out_channels  # Store number of output channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))  # Convolution layer
        self.r = r  # Store the upsampling rate
        self.nchannels_div_r = self.conv.out_channels // self.r
        self.group_size = 101
        self.group_size_r = self.group_size * self.r

    def forward(self, x):
        """
        Defines the forward pass for the SPConvTranspose2d module.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, in_channels, height, width].

        Returns:
            torch.Tensor: Output tensor after transposed convolution operation.
        """
        x = self.pad1(x)  # Apply padding to input
        out = self.conv(x)  # Perform convolution operation
        out = out.view((1, self.r, self.nchannels_div_r, -1, self.group_size))  # Reshape output for separation
        out = out.permute(0, 2, 3, 4, 1)  # Rearrange dimensions
        out = out.contiguous().view((1, self.nchannels_div_r, -1, self.group_size_r))  # Final output shape
        return out  # Return the final output


class MaskDecoder(nn.Module):
    """
    MaskDecoder: A decoder module for estimating masks used in audio processing.

    This module utilizes a dilated dense network to capture features and 
    applies sub-pixel convolution to upscale the output. It produces 
    a mask that can be applied to the magnitude of audio signals.

    Args:
        num_features (int): The number of features in the output mask.
        num_channel (int): The number of channels in intermediate layers (default is 64).
        out_channel (int): The number of output channels for the final output mask (default is 1).
    """

    def __init__(self, num_features, num_channel=64, out_channel=1):
        """Initializes the MaskDecoder with specified parameters."""
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)  # Dense feature extraction
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)  # Sub-pixel convolution for upsampling
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))  # Convolution layer to produce mask
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)  # Normalization layer
        self.prelu = nn.PReLU(out_channel)  # Activation function
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))  # Final convolution layer
        self.prelu_out = nn.PReLU(num_features, init=-0.25)  # Final activation for output mask

    def forward(self, x):
        """
        Defines the forward pass for the MaskDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output mask tensor after processing through the decoder.
        """
        x = self.dense_block(x)  # Feature extraction using dilated dense block
        x = self.sub_pixel(x)  # Upsample the features
        x = self.conv_1(x)  # Convolution to estimate the mask
        x = self.prelu(self.norm(x))  # Apply normalization and activation
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)  # Final convolution and rearrangement
        return self.prelu_out(x).permute(0, 2, 1)  # Final output shape


class ComplexDecoder(nn.Module):
    """
    ComplexDecoder: A decoder module for estimating complex-valued outputs.

    This module processes features through a dilated dense network and a 
    sub-pixel convolution layer to generate two output channels representing 
    the real and imaginary parts of the complex output.

    Args:
        num_channel (int): The number of channels in intermediate layers (default is 64).
    """

    def __init__(self, num_channel=64):
        """Initializes the ComplexDecoder with specified parameters."""
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)  # Dense feature extraction
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)  # Sub-pixel convolution for upsampling
        self.prelu = nn.PReLU(num_channel)  # Activation function
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)  # Normalization layer
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))  # Convolution layer to produce complex outputs

    def forward(self, x):
        """
        Defines the forward pass for the ComplexDecoder.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width].

        Returns:
            torch.Tensor: Output tensor containing real and imaginary parts.
        """
        x = self.dense_block(x)  # Feature extraction using dilated dense block
        x = self.sub_pixel(x)  # Upsample the features
        x = self.prelu(self.norm(x))  # Apply normalization and activation
        x = self.conv(x)  # Generate complex output
        return x  # Return the output tensor


class SyncANet(nn.Module):
    """
    SyncANet: A synchronous audio processing network for separating audio signals.

    This network integrates dense encoding, synchronous attention blocks, 
    and separate decoders for estimating masks and complex-valued outputs.

    Args:
        num_channel (int): The number of channels in the network (default is 64).
        num_features (int): The number of features for the mask decoder (default is 201).
    """

    def __init__(self, num_channel=64, num_features=201):
        """Initializes the SyncANet with specified parameters."""
        super(SyncANet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)  # Dense encoder for input
        self.n_layers = 6  # Number of synchronous attention layers
        self.blocks = nn.ModuleList([])  # List to hold attention blocks

        # Initialize attention blocks
        for _ in range(self.n_layers):
            self.blocks.append(
                SyncANetBlock(
                    emb_dim=num_channel,
                    emb_ks=2,
                    emb_hs=1,
                    n_freqs=int(num_features // 2) + 1,
                    hidden_channels=num_channel * 2,
                    n_head=4,
                    approx_qk_dim=512,
                    activation='prelu',
                    eps=1.0e-5,
                )
            )

        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)  # Mask decoder
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)  # Complex decoder

    def forward(self, x):
        """
        Defines the forward pass for the SyncANet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, 2, height, width] representing complex signals.

        Returns:
            list: List containing the real and imaginary parts of the output tensor.
        """
        out_list = []  # List to store outputs
        mag = torch.sqrt(x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2).unsqueeze(1)  # Calculate magnitude
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)  # Calculate phase
        x_in = torch.cat([mag, x], dim=1)  # Concatenate magnitude and input for processing

        x = self.dense_encoder(x_in)  # Feature extraction using dense encoder
        for ii in range(self.n_layers):
            x = self.blocks[ii](x)  # Pass through attention blocks

        mask = self.mask_decoder(x)  # Estimate mask from features
        out_mag = mask * mag  # Apply mask to magnitude

        complex_out = self.complex_decoder(x)  # Generate complex output
        mag_real = out_mag * torch.cos(noisy_phase)  # Real part of the output
        mag_imag = out_mag * torch.sin(noisy_phase)  # Imaginary part of the output
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)  # Final real output
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)  # Final imaginary output
        out_list.append(final_real)  # Append real output to list
        out_list.append(final_imag)  # Append imaginary output to list

        return out_list  # Return list of outputs


class FFConvM(nn.Module):
    """
    FFConvM: A feedforward convolutional module combining linear layers, normalization, 
    non-linear activation, and convolution operations.

    This module processes input tensors through a sequence of transformations, including 
    normalization, a linear layer with a SiLU activation, a convolutional operation, and 
    dropout for regularization.

    Args:
        dim_in (int): The number of input features (dimensionality of input).
        dim_out (int): The number of output features (dimensionality of output).
        norm_klass (nn.Module): The normalization class to be applied (default is nn.LayerNorm).
        dropout (float): The dropout probability for regularization (default is 0.1).
    """

    def __init__(
            self,
            dim_in,
            dim_out,
            norm_klass=nn.LayerNorm,
            dropout=0.1
    ):
        """Initializes the FFConvM with specified parameters."""
        super().__init__()

        # Define the sequential model
        self.mdl = nn.Sequential(
            norm_klass(dim_in),  # Apply normalization to input
            nn.Linear(dim_in, dim_out),  # Linear transformation to dim_out
            nn.SiLU(),  # Non-linear activation using SiLU (Sigmoid Linear Unit)
            ConvModule(dim_out),  # Convolution operation on the output
            # nn.Dropout(dropout)  # Dropout layer for regularization
        )

    def forward(self, x):
        """
        Defines the forward pass for the FFConvM.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, dim_in].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, dim_out] after processing.
        """
        output = self.mdl(x)  # Pass input through the sequential model
        return output  # Return the processed output


class SyncANetBlock(nn.Module):
    """
    SyncANetBlock implements a modified version of the MossFormer (GatedFormer) module,
    inspired by the TF-GridNet architecture (https://arxiv.org/abs/2211.12433). 
    It combines gated triple-attention schemes and Finite Short Memory Network (FSMN) modules 
    to enhance computational efficiency and overall performance in audio processing tasks.

    Attributes:
        emb_dim (int): Dimensionality of the embedding.
        emb_ks (int): Kernel size for embeddings.
        emb_hs (int): Stride size for embeddings.
        n_freqs (int): Number of frequency bands.
        hidden_channels (int): Number of hidden channels.
        n_head (int): Number of attention heads.
        approx_qk_dim (int): Approximate dimension for query-key matrices.
        activation (str): Activation function to use.
        eps (float): Small value to avoid division by zero in normalization layers.
    """

    def __getitem__(self, key):
        """
        Allows accessing module attributes using indexing.

        Args:
            key: Attribute name to retrieve.

        Returns:
            The requested attribute.
        """
        return getattr(self, key)

    def __init__(
        self,
        emb_dim,
        emb_ks,
        emb_hs,
        n_freqs,
        hidden_channels,
        n_head=4,
        approx_qk_dim=512,
        activation="prelu",
        eps=1e-5,
    ):
        """
        Initializes the SyncANetBlock with the specified parameters.

        Args:
            emb_dim (int): Dimensionality of the embedding.
            emb_ks (int): Kernel size for embeddings.
            emb_hs (int): Stride size for embeddings.
            n_freqs (int): Number of frequency bands.
            hidden_channels (int): Number of hidden channels.
            n_head (int): Number of attention heads. Default is 4.
            approx_qk_dim (int): Approximate dimension for query-key matrices. Default is 512.
            activation (str): Activation function to use. Default is "prelu".
            eps (float): Small value to avoid division by zero in normalization layers. Default is 1e-5.
        """
        super().__init__()

        in_channels = emb_dim * emb_ks  # Calculate the number of input channels

        ## Intra modules: Modules for internal processing within the block
        self.Fconv = nn.Conv2d(emb_dim, in_channels, kernel_size=(1, emb_ks), stride=(1, 1), groups=emb_dim)
        self.intra_norm = LayerNormalization4D(emb_dim, eps=eps)  # Layer normalization
        self.intra_to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_channels,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.intra_to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_channels,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.intra_rnn = self._build_repeats(in_channels, hidden_channels, 20, hidden_channels, repeats=1)  # FSMN layers
        self.intra_mossformer = MossFormer(dim=emb_dim, group_size=n_freqs)  # MossFormer module

        # Linear transformation for intra module output
        self.intra_linear = nn.ConvTranspose1d(
            hidden_channels, emb_dim, emb_ks, stride=emb_hs
        )
        self.intra_se = SELayer(channel=emb_dim, reduction=1)  # Squeeze-and-excitation layer

        ## Inter modules: Modules for external processing between blocks
        self.inter_norm = LayerNormalization4D(emb_dim, eps=eps)  # Layer normalization
        self.inter_to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_channels,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.inter_to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_channels,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        self.inter_rnn = self._build_repeats(in_channels, hidden_channels, 20, hidden_channels, repeats=1)  # FSMN layers
        self.inter_mossformer = MossFormer(dim=emb_dim, group_size=256)  # MossFormer module

        # Linear transformation for inter module output
        self.inter_linear = nn.ConvTranspose1d(
            hidden_channels, emb_dim, emb_ks, stride=emb_hs
        )
        self.inter_se = SELayer(channel=emb_dim, reduction=1)  # Squeeze-and-excitation layer

        # Approximate query-key dimension calculation
        E = math.ceil(approx_qk_dim * 1.0 / n_freqs)
        assert emb_dim % n_head == 0  # Ensure emb_dim is divisible by n_head

        # Define attention convolution layers for each head
        for ii in range(n_head):
            self.add_module(
                f"attn_conv_Q_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                f"attn_conv_K_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, E, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((E, n_freqs), eps=eps),
                ),
            )
            self.add_module(
                f"attn_conv_V_{ii}",
                nn.Sequential(
                    nn.Conv2d(emb_dim, emb_dim // n_head, 1),
                    get_layer(activation)(),
                    LayerNormalization4DCF((emb_dim // n_head, n_freqs), eps=eps),
                ),
            )

        # Final attention concatenation projection
        self.add_module(
            "attn_concat_proj",
            nn.Sequential(
                nn.Conv2d(emb_dim, emb_dim, 1),
                get_layer(activation)(),
                LayerNormalization4DCF((emb_dim, n_freqs), eps=eps),
            ),
        )

        # Store parameters for further processing
        self.emb_dim = emb_dim
        self.emb_ks = emb_ks
        self.emb_hs = emb_hs
        self.n_head = n_head
        self.group_size = 101
        self.emb_dim_plus = self.group_size * self.attn_conv_K_0._modules["0"].out_channels
        self.emb_dim_factor = float(1.0 / self.emb_dim_plus ** 0.5)
        self.emb_dim_div_head = self.emb_dim // self.n_head

    def _build_repeats(self, in_channels, out_channels, lorder, hidden_size, repeats=1):
        """
        Constructs a sequence of UniDeepFSMN modules.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            lorder (int): Order of the filter.
            hidden_size (int): Hidden size for the FSMN.
            repeats (int): Number of times to repeat the module. Default is 1.

        Returns:
            nn.Sequential: A sequence of UniDeepFSMN modules.
        """
        repeats = [
            UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)
            for _ in range(repeats)
        ]
        return nn.Sequential(*repeats)

    def forward(self, x):
        old_T = x.shape[-2].unsqueeze(0)
        intra_rnn = self.intra_norm(x)  # Normalize input for intra-process
        intra_rnn = self.Fconv(intra_rnn)  # Apply depthwise convolution
        intra_rnn = intra_rnn.squeeze(0).permute(1, 2, 0).contiguous()
        intra_rnn_u = self.intra_to_u(intra_rnn)  # Linear transformation
        intra_rnn_v = self.intra_to_v(intra_rnn)  # Linear transformation
        intra_rnn_u = self.intra_rnn(intra_rnn_u.unsqueeze(0)).squeeze(0)  # Apply FSMN
        intra_rnn = intra_rnn_v * intra_rnn_u  # Element-wise multiplication
        intra_rnn = intra_rnn.transpose(1, 2)  # Reshape back
        intra_rnn = self.intra_linear(intra_rnn)  # Linear projection
        intra_rnn = intra_rnn.transpose(1, 2)  # Reshape for mossformer
        intra_rnn = self.intra_mossformer(intra_rnn)  # Apply MossFormer
        intra_rnn = intra_rnn.permute(2, 0, 1).unsqueeze(0).contiguous()  # Final reshape
        intra_rnn = self.intra_se(intra_rnn)  # Squeeze-and-excitation layer
        intra_rnn = intra_rnn + x  # Residual connection

        # Inter-process
        input_ = intra_rnn
        inter_rnn = self.inter_norm(input_)  # Normalize input for inter-process
        inter_rnn = (inter_rnn.squeeze(0).permute(2, 0, 1).contiguous())  # Reshape for processing
        inter_rnn = F.unfold(inter_rnn.unsqueeze(-1), (self.emb_ks, 1), stride=(self.emb_hs, 1))  # Extract sliding windows
        inter_rnn = inter_rnn.transpose(1, 2)  # Reshape for further processing
        inter_rnn_u = self.inter_to_u(inter_rnn)  # Linear transformation
        inter_rnn_v = self.inter_to_v(inter_rnn)  # Linear transformation
        inter_rnn_u = self.inter_rnn(inter_rnn_u.unsqueeze(0)).squeeze(0)  # Apply FSMN
        inter_rnn = inter_rnn_v * inter_rnn_u  # Element-wise multiplication
        inter_rnn = inter_rnn.transpose(1, 2)  # Reshape back
        inter_rnn = self.inter_linear(inter_rnn)  # Linear projection
        inter_rnn = inter_rnn.transpose(1, 2)  # Reshape for mossformer
        inter_rnn = self.inter_mossformer(inter_rnn)  # Apply MossFormer
        inter_rnn = inter_rnn.permute(2, 0, 1).unsqueeze(0).contiguous()  # Permute for SE layer
        inter_rnn = self.inter_se(inter_rnn)  # Squeeze-and-excitation layer
        inter_rnn = inter_rnn.transpose(-1, 2) + input_  # Residual connection

        all_Q, all_K, all_V = [], [], []

        # Compute query, key, and value for each attention head
        for ii in range(self.n_head):
            all_Q.append(self["attn_conv_Q_%d" % ii](inter_rnn))  # Query
            all_K.append(self["attn_conv_K_%d" % ii](inter_rnn))  # Key
            all_V.append(self["attn_conv_V_%d" % ii](inter_rnn))  # Value

        Q = torch.cat(all_Q, dim=0)  # Concatenate all queries
        K = torch.cat(all_K, dim=0)  # Concatenate all keys
        V = torch.cat(all_V, dim=0)  # Concatenate all values

        # Reshape for attention calculation
        Q = Q.transpose(1, 2).flatten(start_dim=2)
        K = K.transpose(1, 2).flatten(start_dim=2).transpose(1, 2)
        V = V.transpose(1, 2).flatten(start_dim=2)  # Reshape for attention calculation

        # Compute scaled dot-product attention
        attn_mat = torch.matmul(Q, K) * self.emb_dim_factor  # Attention matrix
        attn_mat = F.softmax(attn_mat, dim=2)  # Softmax over attention scores
        V = torch.matmul(attn_mat, V)  # Weighted sum of values

        V = V.reshape(self.n_head, -1, self.emb_dim_div_head, self.group_size)  # Reshape back
        batch = self["attn_concat_proj"](V.transpose(1, 2).contiguous().view(1, self.emb_dim, -1, self.group_size))  # Final linear projection

        # Combine inter-process result with attention output
        return batch + inter_rnn  # Return the output tensor


class LayerNormalization4D(nn.Module):
    """
    LayerNormalization4D applies layer normalization to 4D tensors 
    (e.g., [B, C, T, F]), where B is the batch size, C is the number of channels,
    T is the temporal dimension, and F is the frequency dimension.

    Attributes:
        gamma (torch.Parameter): Learnable scaling parameter.
        beta (torch.Parameter): Learnable shifting parameter.
        eps (float): Small value for numerical stability during variance calculation.
    """

    def __init__(self, input_dimension, eps=1e-5):
        """
        Initializes the LayerNormalization4D layer.

        Args:
            input_dimension (int): The number of channels in the input tensor.
            eps (float, optional): Small constant added for numerical stability.
        """
        super().__init__()
        param_size = [1, input_dimension, 1, 1]
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))  # Scale parameter
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))  # Shift parameter
        init.ones_(self.gamma)  # Initialize gamma to 1
        init.zeros_(self.beta)  # Initialize beta to 0
        self.eps = eps  # Set the epsilon value

    def forward(self, x):

        # Compute mean and standard deviation along the specified dimension
        mu_ = x.mean(dim=1, keepdim=True)  # [B, 1, T, F]
        std_ = torch.sqrt(
            x.var(dim=1, unbiased=False, keepdim=True) + self.eps
        )  # [B, 1, T, F]

        # Normalize the input tensor and apply learnable parameters
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta  # [B, C, T, F]
        return x_hat


class LayerNormalization4DCF(nn.Module):
    """
    LayerNormalization4DCF applies layer normalization to 4D tensors 
    (e.g., [B, C, T, F]) specifically designed for DCF (Dynamic Channel Frequency) inputs.
    
    Attributes:
        gamma (torch.Parameter): Learnable scaling parameter.
        beta (torch.Parameter): Learnable shifting parameter.
        eps (float): Small value for numerical stability during variance calculation.
    """

    def __init__(self, input_dimension, eps=1e-5):
        """
        Initializes the LayerNormalization4DCF layer.

        Args:
            input_dimension (tuple): A tuple containing the dimensions of the input tensor 
                                     (number of channels, frequency dimension).
            eps (float, optional): Small constant added for numerical stability.
        """
        super().__init__()
        assert len(input_dimension) == 2, "Input dimension must be a tuple of length 2."
        param_size = [1, input_dimension[0], 1, input_dimension[1]]  # Shape based on input dimensions
        self.gamma = Parameter(torch.Tensor(*param_size).to(torch.float32))  # Scale parameter
        self.beta = Parameter(torch.Tensor(*param_size).to(torch.float32))  # Shift parameter
        init.ones_(self.gamma)  # Initialize gamma to 1
        init.zeros_(self.beta)  # Initialize beta to 0
        self.eps = eps  # Set the epsilon value

    def forward(self, x):
        # Compute mean and standard deviation along the specified dimensions
        mu_ = x.mean(dim=(1, 3), keepdim=True)  # [B, 1, T, 1]
        std_ = torch.sqrt(
            x.var(dim=(1, 3), unbiased=False, keepdim=True) + self.eps
        )  # [B, 1, T, F]

        # Normalize the input tensor and apply learnable parameters
        x_hat = ((x - mu_) / std_) * self.gamma + self.beta  # [B, C, T, F]
        return x_hat
