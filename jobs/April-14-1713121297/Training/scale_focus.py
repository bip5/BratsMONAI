import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class ScaleFocus(nn.Module):
    def __init__(self,num_layers=2,num_filters=8,num_scales=2):
        super().__init__()

        num_layers = num_layers
        num_scales = num_scales
        num_filters= num_filters
        # Define the 3D convolutional layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList([nn.InstanceNorm3d(8) for _ in range(num_layers)])
        for i in range(num_layers):
            in_channels = 4 if i == 0 else num_filters
            self.conv_layers.append(nn.Conv3d(in_channels, num_filters, kernel_size=3, padding=1))

        # Pooling layers for different scales
        self.pools = [nn.MaxPool3d(kernel_size=2**(i+1), stride=2**(i+1)) for i in range(num_scales)]

        # Learnable weights for scaling at different scales for each layer
        self.weights = nn.ParameterList([nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in range(num_scales)]) for _ in range(num_layers)])

        # Initialize weights
        for layer_weights in self.weights:
            for weight in layer_weights:
                init.normal_(weight, mean=0.0, std=0.01)

        # Final convolution
        self.final_conv = nn.Conv3d(num_filters, 3, kernel_size=1)

    def pad_to_match(self, x, target_size):
        """
        Pad the input tensor 'x' to match the 'target_size'.
        """
        # Calculate padding amounts for each dimension
        pad = [0, target_size[4]-x.shape[4]]  # Padding for width
        pad = pad + [0, target_size[3] - x.shape[3]]  # Padding for height
        pad = pad + [0, target_size[2] - x.shape[2]]  # Padding for depth
        return F.pad(x, pad)

    def forward(self, x):
        original_size = x.shape  # Store the original size of the input

        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x_conv = conv(x)

            # Pool and pad to match at different scales
            # expanded = [self.pad_to_match(pool(x_conv), original_size) for pool in self.pools]

            # Combine with original using layer-specific weights
            x_combined = x_conv
            # for j, exp in enumerate(expanded):
                # x_combined =x_combined + self.weights[i][j] * exp

            # Apply layer normalization after expansion and combination
            x_norm = norm(x_combined)
            x = F.relu(x_norm)

        # Final convolution
        x = self.final_conv(x)

        return x
