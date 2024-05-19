# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch.nn as nn
from Training.Convolutionsprj import Convolution as convolution
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.upsample import UpSample
from monai.networks.layers.utils import get_norm_layer
from monai.utils import InterpolateMode, UpsampleMode
from Training.CustomActivation import get_act_layer
import torch


def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, bias: bool = False
):
    return Convolution(
        spatial_dims, in_channels, out_channels, strides=stride, kernel_size=kernel_size, bias=bias, conv_only=True
    )

class CustomConvLayer(nn.Module):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
        super(CustomConvLayer, self).__init__()
        conv_types = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        conv_layer = conv_types[spatial_dims]
        self.out_channels=out_channels
        # Create a convolutional layer with double the output channels
        self.conv_layer = conv_layer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=bias,device='cuda:0',dtype='float32')

    def forward(self, x):
        # Apply convolution
        conv_output = self.conv_layer(x)

        #Split the output into two halves
        split_size = out_channels
        first_half, second_half = torch.split(conv_output, split_size, dim=1)

       # Reshape for softmax application
        original_shape = first_half.shape
        first_half_flattened = first_half.view(original_shape[0], original_shape[1], -1)

        #Apply softmax across spatial dimensions
        first_half_softmax = torch.softmax(first_half_flattened, dim=2)

        #Reshape back to original dimensions
        first_half_softmax = first_half_softmax.view(original_shape)

        #Multiply with the second half
        modified_output = first_half_softmax * second_half

        return conv_output #modified_output

def modified_get_conv_layer(spatial_dims, in_channels, out_channels, kernel_size=3, stride=1, bias=False):
    # print('kernel', kernel_size)
    return convolution(spatial_dims, in_channels, out_channels, kernel_size, stride, bias)


# Example of usage
# conv_layer = modified_get_conv_layer(spatial_dims, in_channels, out_channels)
# output = conv_layer(input_tensor)


def get_upsample_layer(
    spatial_dims: int, in_channels: int, upsample_mode: UpsampleMode | str = "nontrainable", scale_factor: int = 2
):
    return UpSample(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=in_channels,
        scale_factor=scale_factor,
        mode=upsample_mode,
        interp_mode=InterpolateMode.LINEAR,
        align_corners=False,
    )

class ResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.conv2 = get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x

class modified_ResBlock(nn.Module):
    """
    ResBlock employs skip connection and two convolution blocks and is used
    in SegResNet based on `3D MRI brain tumor segmentation using autoencoder regularization
    <https://arxiv.org/pdf/1810.11654.pdf>`_.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        norm: tuple | str,
        kernel_size: int = 3,
        act: tuple | str = ("RELU", {"inplace": True}),
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions, could be 1, 2 or 3.
            in_channels: number of input channels.
            norm: feature normalization type and arguments.
            kernel_size: convolution kernel size, the value should be an odd number. Defaults to 3.
            act: activation type and arguments. Defaults to ``RELU``.
        """

        super().__init__()

        if kernel_size % 2 != 1:
            raise AssertionError("kernel_size should be an odd number.")

        self.norm1 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.norm2 = get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
        self.act = get_act_layer(act)
        self.conv1 = modified_get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )
        self.conv2 = modified_get_conv_layer(
            spatial_dims, in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size
        )

    def forward(self, x):
        identity = x

        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act(x)
        x = self.conv2(x)

        x += identity

        return x