import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th
import numpy as np
import os

class UniDeepFsmn(nn.Module):
    """UniDeepFsmn is a neural network module implementing a unified deep
    Feedforward Sequential Memory Network (FSMN) for sequence-to-sequence tasks.

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        lorder (int, optional): The order of the linear filter. If None, the filter order is not used.
        hidden_size (int, optional): The number of hidden units in the first linear layer.

    Attributes:
        input_dim (int): The dimension of the input features.
        output_dim (int): The dimension of the output features.
        lorder (int): The order of the linear filter.
        hidden_size (int): The number of hidden units.
        linear (nn.Linear): Linear transformation layer from input_dim to hidden_size.
        project (nn.Linear): Linear transformation layer from hidden_size to output_dim without bias.
        conv1 (nn.Conv2d): Convolutional layer for processing the output of the project layer.
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        """Initializes the UniDeepFsmn model.

        Args:
            input_dim (int): The number of input features.
            output_dim (int): The number of output features.
            lorder (int, optional): The order of the linear filter. Default is None.
            hidden_size (int, optional): The number of hidden units. Default is None.
        """
        super(UniDeepFsmn, self).__init__()

        self.input_dim = input_dim  # Store input dimension
        self.output_dim = output_dim  # Store output dimension

        if lorder is None:
            return  # Exit if no filter order is specified

        self.lorder = lorder  # Store the linear filter order
        self.hidden_size = hidden_size  # Store hidden layer size

        # Define linear transformation from input to hidden layer
        self.linear = nn.Linear(input_dim, hidden_size)

        # Define linear transformation from hidden layer to output
        self.project = nn.Linear(hidden_size, output_dim, bias=False)

        # Define convolutional layer for processing output with specified filter order
        self.conv1 = nn.Conv2d(output_dim, output_dim, [lorder + lorder - 1, 1], [1, 1], groups=output_dim, bias=False)

    def forward(self, input):
        """Forward pass of the UniDeepFsmn model.

        Args:
            input (torch.Tensor): The input tensor with shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor after processing through the model.
        """
        # Apply ReLU activation after linear transformation
        f1 = F.relu(self.linear(input))

        # Project to output dimension
        p1 = self.project(f1)

        # Permute dimensions for the convolutional layer input
        x_per = p1.permute(1, 3, 2, 0)

        # Pad the input for convolutional layer
        y = F.pad(x_per, [0, 0, self.lorder - 1, self.lorder - 1])

        # Apply convolution and add the result back to the original input
        out = x_per + self.conv1(y)

        # Return the combined output
        return input + out.permute(3, 0, 2, 1)
