import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class HiddenFocus(nn.Module):
    def __init__(self, num_output_channels=16,num_layers=7):
        super(HiddenFocus, self).__init__()

        num_layers = num_layers # number of layers in the model

        # Use num_output_channels as a configurable parameter
        self.num_output_channels = num_output_channels 

        # Define the 3D convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.added_channels = nn.ModuleList()
        self.scale_weights = nn.ModuleList()

        # Additional channel propagated through layers
        self.additional_channels = []

        # Define layers
        for i in range(num_layers):
            in_channels = 5 if i == 0 else (num_output_channels+1)
            self.conv_layers.append(nn.Conv3d(in_channels, num_output_channels, kernel_size=3, padding=1))
            
            # 1x1 Convolution to propagate an additional channel
            self.added_channels.append(nn.Conv3d(in_channels-1, 1, kernel_size=1))

            # Pooling layer
            kernel_stride_size = 2**(i + 1)
            self.pool_layers.append(nn.MaxPool3d(kernel_size=kernel_stride_size, stride=kernel_stride_size))

        # Scale weights for each channel in each layer
        self.scale_weights = nn.ModuleList()
        for _ in range(num_layers):
            layer_weights = nn.ParameterList([nn.Parameter(torch.Tensor(1)) for _ in range(num_output_channels)])
            self.scale_weights.append(layer_weights)

        self.norm_layers = nn.ModuleList([nn.InstanceNorm3d(num_output_channels) for _ in range(num_layers)])
        # Initialize scale weights
        for layer_weights in self.scale_weights:
            for weight in layer_weights:
                init.normal_(weight, mean=0.0, std=0.01)

        # Final convolution
        self.final_conv = nn.Conv3d(num_output_channels + num_layers, 3, kernel_size=1)

    def expand(self, x, original_size):
        """
        Expand the input tensor 'x' to match the 'original_size'.
        """
        scale_factor = [os // xs for os, xs in zip(original_size, x.shape[2:])]
        # print('scale factor',scale_factor)
        return x.repeat_interleave(scale_factor[0], dim=2)\
                .repeat_interleave(scale_factor[1], dim=3)\
                .repeat_interleave(scale_factor[2], dim=4)
                
    def forward(self, x):
        original_size=x.shape[2:]
        # print('original s-ze',original_size)
        # Store the original spatial size of the input
        for i, (conv, pool, add_channel, weights,norm) in enumerate(zip(self.conv_layers, self.pool_layers, self.added_channels, self.scale_weights,self.norm_layers)):
            additional_channel=add_channel(x)            
            self.additional_channels.append(additional_channel)  
           
            x = torch.cat((x, additional_channel), dim=1)
            x_conv = conv(x)

            # Pooling
            x_pooled = pool(x_conv)

            # Upscaling and scaling
            x_scaled = torch.zeros_like(x_conv)
            for j in range(self.num_output_channels):
                weight = self.scale_weights[i][j]
                # Apply scaling for each channel
                x_scaled[:, j:j+1, :, :, :] = self.expand(x_pooled[:, j:j+1, :, :, :], original_size) * weight                
            
            x_norm=norm(x_scaled)
            x=F.relu(x_norm)
           
            
           

        # Concatenate all propagated channels for the final layer
        for add_ch in self.additional_channels:           
            x = torch.cat((x, add_ch), dim=1)
        self.additional_channels=[] #reset to avoid passing to next iteration
        # Final convolution
        x = self.final_conv(x)

        return x





