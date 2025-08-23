import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.nn.parameter import Parameter
import numpy as np
import os


class UniDeepFsmn(nn.Module):
    """
    UniDeepFsmn is a neural network module that implements a single-deep feedforward sequence memory network (FSMN).

    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        lorder (int): Length of the order for the convolution layers.
        hidden_size (int): Number of hidden units in the linear layer.
        linear (nn.Linear): Linear layer to project input features to hidden size.
        project (nn.Linear): Linear layer to project hidden features to output dimensions.
        conv1 (nn.Conv2d): Convolutional layer for processing the output in a grouped manner.
        padding_template (torch.Tensor): A buffer used as a template for padding.
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size

        # Initialize the layers
        self.linear = nn.Linear(input_dim, hidden_size)  # Linear transformation to hidden size
        self.project = nn.Linear(hidden_size, output_dim, bias=False)  # Project hidden size to output dimension
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim,
                               bias=False)  # Convolution layer

        # --- Start of updated code ---
        # Pre-allocate a padding template in the init stage.
        # We create it with a batch size of 1 and will expand it dynamically in the forward pass.
        # Shape is (batch, channels, length, width), which corresponds to (1, output_dim, pad_length, 1).
        pad_size = self.lorder - 1
        padding = th.zeros(1, self.output_dim, pad_size, 1)

        # Register as a persistent buffer, so it's part of the model's state
        # but not considered a parameter. This handles device placement (e.g., .to(device)).
        self.register_buffer('padding', padding)
        # --- End of updated code ---

    def forward(self, input):
        f1 = F.relu(self.linear(input))  # Apply linear layer followed by ReLU activation
        p1 = self.project(f1)  # Project to output dimension
        x = th.unsqueeze(p1, 1)  # Add a dimension for compatibility with Conv2d
        x_per = x.permute(0, 3, 2, 1)  # Permute dimensions for convolution
        y = th.cat([self.padding, x_per, self.padding], dim=2)
        out = x_per + self.conv1(y)  # Add original input to convolution output
        out1 = out.permute(0, 3, 2, 1)  # Permute back to original dimensions
        return input + out1.squeeze(1)


class UniDeepFsmn_dual(nn.Module):
    """
    UniDeepFsmn_dual is a neural network module that implements a dual-deep feedforward sequence memory network (FSMN).

    This class extends the UniDeepFsmn by adding a second convolution layer for richer feature extraction.

    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        lorder (int): Length of the order for the convolution layers.
        hidden_size (int): Number of hidden units in the linear layer.
        linear (nn.Linear): Linear layer to project input features to hidden size.
        project (nn.Linear): Linear layer to project hidden features to output dimensions.
        conv1 (nn.Conv2d): First convolutional layer for processing the output.
        conv2 (nn.Conv2d): Second convolutional layer for further processing the features.
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super(UniDeepFsmn_dual, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # Initialize the layers
        self.linear = nn.Linear(input_dim, hidden_size)  # Linear transformation to hidden size
        self.project = nn.Linear(hidden_size, output_dim, bias=False)  # Project hidden size to output dimension
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim, bias=False)  # First convolution layer
        self.conv2 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim // 4, bias=False)  # Second convolution layer

    def forward(self, input):
        """
        Forward pass for the UniDeepFsmn_dual model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of the same shape as input, enhanced by the network.
        """
        f1 = F.relu(self.linear(input))  # Apply linear layer followed by ReLU activation
        p1 = self.project(f1)  # Project to output dimension
        x = th.unsqueeze(p1, 1)  # Add a dimension for compatibility with Conv2d
        x_per = x.permute(0, 3, 2, 1)  # Permute dimensions for convolution
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])  # Pad for causal convolution
        conv1_out = x_per + self.conv1(y)  # Add original input to first convolution output
        z = F.pad(conv1_out, [0, 0, self.lorder - 1, self.lorder - 1])  # Pad for second convolution
        out = conv1_out + self.conv2(z)  # Add output of second convolution
        out1 = out.permute(0, 3, 2, 1)  # Permute back to original dimensions
        return input + out1.squeeze()  # Return enhanced input


class DilatedDenseNet(nn.Module):
    """
    DilatedDenseNet implements a dense network structure with dilated convolutions.

    This architecture enables wider receptive fields while maintaining a lower number of parameters. 
    It consists of multiple convolutional layers with dilation rates that increase at each layer.

    Attributes:
        depth (int): Number of convolutional layers in the network.
        in_channels (int): Number of input channels for the first layer.
        pad (nn.ConstantPad2d): Padding layer to maintain dimensions.
        twidth (int): Width of the kernel used in convolution.
        kernel_size (tuple): Kernel size for convolution operations.
    """

    def __init__(self, depth=4, lorder=20, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)  # Padding for the input
        self.twidth = lorder * 2 - 1  # Width of the kernel
        self.kernel_size = (self.twidth, 1)  # Kernel size for convolutions

        # Initialize layers dynamically based on depth
        for i in range(self.depth):
            dil = 2 ** i  # Calculate dilation rate
            pad_length = lorder + (dil - 1) * (lorder - 1) - 1  # Calculate padding length
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((0, 0, pad_length, pad_length), value=0.))  # Padding for dilation
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1), groups=self.in_channels, bias=False))  # Convolution layer with dilation
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))  # Normalization layer
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))  # Activation layer

    def forward(self, x):
        """
        Forward pass for the DilatedDenseNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor after applying dense layers.
        """
        skip = x  # Initialize skip connection
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)  # Apply padding
            out = getattr(self, 'conv{}'.format(i + 1))(out)  # Apply convolution
            out = getattr(self, 'norm{}'.format(i + 1))(out)  # Apply normalization
            out = getattr(self, 'prelu{}'.format(i + 1))(out)  # Apply PReLU activation            
            skip = th.cat([out, skip], dim=1)  # Concatenate the output with the skip connection
        return out  # Return the final output

class UniDeepFsmn_dilated(nn.Module):
    """
    UniDeepFsmn_dilated combines the UniDeepFsmn architecture with a dilated dense network 
    to enhance feature extraction while maintaining efficient computation.

    Attributes:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        depth (int): Depth of the dilated dense network.
        lorder (int): Length of the order for the convolution layers.
        hidden_size (int): Number of hidden units in the linear layer.
        linear (nn.Linear): Linear layer to project input features to hidden size.
        project (nn.Linear): Linear layer to project hidden features to output dimensions.
        conv (DilatedDenseNet): Instance of the DilatedDenseNet for feature extraction.
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None, depth=2):
        super(UniDeepFsmn_dilated, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.depth = depth
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # Initialize layers
        self.linear = nn.Linear(input_dim, hidden_size)  # Linear transformation to hidden size
        self.project = nn.Linear(hidden_size, output_dim, bias=False)  # Project hidden size to output dimension
        self.conv = DilatedDenseNet(depth=self.depth, lorder=lorder, in_channels=output_dim)  # Dilated dense network for feature extraction

    def forward(self, input):
        """
        Forward pass for the UniDeepFsmn_dilated model.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of the same shape as input, enhanced by the network.
        """
        f1 = F.relu(self.linear(input))  # Apply linear layer followed by ReLU activation
        p1 = self.project(f1)  # Project to output dimension
        x = th.unsqueeze(p1, 1)  # Add a dimension for compatibility with Conv2d
        x_per = x.permute(0, 3, 2, 1)  # Permute dimensions for convolution
        out = self.conv(x_per)  # Pass through the dilated dense network
        out1 = out.permute(0, 3, 2, 1)  # Permute back to original dimensions

        return input + out1.squeeze()  # Return enhanced input
