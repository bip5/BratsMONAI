import torch
import torch.nn as nn
import torch.optim as optim

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, units, learning_rate=0.0000001):
        super(Layer, self).__init__()
        self.units = units
        self.blocks = nn.ModuleList([ConvBlock3D(in_channels, out_channels) for _ in range(units)])
        self.optimizers = [optim.Adam(self.blocks[i].parameters(), lr=learning_rate) for i in range(units)]
       

    def forward(self, x, y=None):
        outputs = []
        loss_array = []
        for i in range(self.units):
            x_out = self.blocks[i](x)

            if y is not None:
                y_out = self.blocks[i](y)
                loss = torch.sum(torch.abs(x_out - y_out) + torch.abs(x_out - y))
                loss_array.append(loss)
            outputs.append(x_out)

        return torch.cat(outputs, dim=1), loss_array

class LayerNet(nn.Module):
    def __init__(self, base_channels, out_channels, layers, units_per_layer):
        super(LayerNet, self).__init__()
        self.layers = nn.ModuleList()
        in_channels = base_channels
        
        for i in range(layers):
            # For the final layer, use only a single unit
            units = 1 if i == layers - 1 else units_per_layer

            layer = Layer(in_channels, out_channels, units)
            self.layers.append(layer)
            in_channels = units * out_channels  # Update in_channels for the next layer

    def forward(self, x, y=None):
        total_loss = []
        for layer in self.layers:
            x, layer_loss = layer(x, y)
            if y is not None:
                total_loss.extend(layer_loss)
        
        return x, total_loss

# # Example initialization
# model = LayerNet(base_channels=4, out_channels=3, layers=10, units_per_layer=10)
