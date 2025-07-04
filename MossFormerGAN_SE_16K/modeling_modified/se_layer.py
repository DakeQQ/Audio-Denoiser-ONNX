from torch import nn
import torch

class SELayer(nn.Module):
    """
    Squeeze-and-Excitation (SE) Layer.

    This layer implements the Squeeze-and-Excitation mechanism, which adaptively
    recalibrates channel-wise feature responses by explicitly modeling 
    interdependencies between channels. It enhances the representational power
    of a neural network by emphasizing informative features while suppressing
    less useful ones.

    Args:
        channel (int): The number of input channels.
        reduction (int, optional): Reduction ratio for the dimensionality
            of the intermediate representations. Default is 16.
    """
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Adaptive average pooling to reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.avg_pool_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # First linear layer
            nn.ReLU(inplace=True),                    # Activation layer
            nn.Linear(channel // reduction, channel),  # Second linear layer
            nn.Sigmoid()                              # Sigmoid activation for scaling
        )
        
        # Adaptive max pooling to reduce spatial dimensions to 1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.max_pool_layer = nn.Sequential(
            nn.Linear(channel, channel // reduction),  # First linear layer
            nn.ReLU(inplace=True),                    # Activation layer
            nn.Linear(channel // reduction, channel),  # Second linear layer
            nn.Sigmoid()                              # Sigmoid activation for scaling
        )

    def forward(self, x):
        x_avg = self.avg_pool(x).view(1, -1)  # Squeeze: apply average pooling
        x_avg = self.avg_pool_layer(x_avg)  # Excitation: pass through layers

        x_max = self.max_pool(x).view(1, -1)  # Squeeze: apply max pooling
        x_max = self.max_pool_layer(x_max)  # Excitation: pass through layers

        # Scale the input features by the computed channel weights
        y = (x_avg + x_max).view(1, -1, 1, 1) * x
        return y  # Return the recalibrated output
