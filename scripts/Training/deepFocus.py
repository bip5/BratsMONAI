import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DeepFocusCNN(nn.Module):
    def __init__(self):
        super(DeepFocusCNN, self).__init__()

        # Number of layers and scales
        num_layers = 5
        num_scales = 5
     
          # Define the 3D convolutional layers and layer norm layers
        self.conv_layers = nn.ModuleList()
        # self.norm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = 4 if i == 0 else 8
            conv = nn.Conv3d(in_channels, 8, kernel_size=3, padding=1)
            norm = nn.LayerNorm([8, 128, 128, 128])  # Adjust the size [channels, D, H, W] as per your layer output
            self.conv_layers.append(conv)
            # self.norm_layers.append(norm)
        self.norm_layers = nn.ModuleList([nn.InstanceNorm3d(8) for _ in range(num_layers)])
        # Pooling layers for different scales
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool3d(kernel_size=4, stride=4)
        self.pool3 = nn.MaxPool3d(kernel_size=8, stride=8)
        self.pool4 = nn.MaxPool3d(kernel_size=16, stride=16)
        self.pool5 = nn.MaxPool3d(kernel_size=32, stride=32)
        self.pool6 = nn.MaxPool3d(kernel_size=64, stride=64)
        self.pool7 = nn.MaxPool3d(kernel_size=128, stride=128)

        # Learnable weights for scaling at different scales for each layer
        self.weights = nn.ParameterList([nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in range(num_scales)]) for _ in range(num_layers)])

        # Initialize weights
        for layer_weights in self.weights:
            for weight in layer_weights:
                init.normal_(weight, mean=0.0, std=0.01)

        # Final convolution
        self.final_conv = nn.Conv3d(8, 3, kernel_size=1)
    def expand(self, x, original_size):
        """
        Expand the input tensor 'x' to match the 'original_size'.
        """
        scale_factor = [os // xs for os, xs in zip(original_size, x.shape[2:])]
        return x.repeat_interleave(scale_factor[0], dim=2)\
                .repeat_interleave(scale_factor[1], dim=3)\
                .repeat_interleave(scale_factor[2], dim=4)
    # def upsample_and_scale(self, x_pooled, scale_factor, weight, original_size):
            # """
            # Upsample x_pooled to match the original_size and scale it with weight.
            # """
            # # Upsample to the size of the original channel
            # x_upsampled = F.interpolate(x_pooled, size=original_size, mode='nearest', align_corners=True)

            # # Scale the upsampled map
            # x_scaled = x_upsampled * weight

            # return x_scaled
        
    def forward(self, x):
        original_size = x.shape[2:]  # Store the original spatial size of the input

        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            x_conv = conv(x)

            # Pool and expand at different scales
            expanded = [self.expand(pool(x_conv), original_size) for pool in [self.pool1, self.pool2, self.pool3, self.pool4, self.pool5]]#, self.pool6, self.pool7]]

            # Combine with original using layer-specific weights
            # Use an out-of-place operation for combination
            x_combined = x_conv
            for j, exp in enumerate(expanded):
                x_combined = x_combined + self.weights[i][j] * exp

            # Apply layer normalization after expansion and combination
            x_norm = norm(x_combined)
            x = F.relu(x_norm)
        # for i, conv in enumerate(self.conv_layers):
            # x_conv = conv(x)

            # # Pool at different scales
            # pooled_maps = [pool(x_conv) for pool in [self.pool1, self.pool2, self.pool3, self.pool4, self.pool5, self.pool6, self.pool7]]

            # # Upsample, scale and combine with original
            # x_combined = x_conv.clone()
            # for j, pooled in enumerate(pooled_maps):
                # x_combined += self.upsample_and_scale(pooled, scale_factor=2**j, weight=self.weights[i][j], original_size=x_conv.shape[2:])

            # # Apply normalization and activation
            # x_norm = self.norm_layers[i](x_combined)
            # x = F.relu(x_norm)
        # Final layer
        x = self.final_conv(x)

        return x






