import torch
from monai.networks.nets import SwinUNETR

class SwinUNETRWithDebug(SwinUNETR):
    def forward(self, x):
        print(f"Input: {x.shape}")
        for name, module in self.named_children():
            if name == 'swinViT':
                x = module(x)
                x = self._handle_swinvit_output(name, x)
            elif name == 'encoder':
                x = self._forward_encoder(x, module)
            elif name == 'decoder':
                x = self._forward_decoder(x, module)
            else:
                x = module(x)
            x = self._handle_module_output(name, x)
        return x
    
    def _forward_encoder(self, x, encoder):
        for name, module in encoder.named_children():
            x = module(x)
            x = self._handle_module_output(f"encoder - {name}", x)
        return x
    
    def _forward_decoder(self, x, decoder):
        for name, module in decoder.named_children():
            x = module(x)
            x = self._handle_module_output(f"decoder - {name}", x)
        return x
    
    def _handle_swinvit_output(self, name, output):
        if isinstance(output, list):
            print(f"After {name} (list):")
            for i, t in enumerate(output):
                print(f"  - Element {i}: {t.shape}")
            # Return the first element for now to continue forward pass
            output = output[0]
        return output

    def _handle_module_output(self, name, output):
        if isinstance(output, list):
            print(f"After {name} (list):")
            for i, t in enumerate(output):
                print(f"  - Element {i}: {t.shape}")
            output = output[0]  # Use the first element to continue forward pass
        else:
            print(f"After {name}: {output.shape}")
        return output

# Load the SwinUNETR model with the debugging subclass
model = SwinUNETRWithDebug(
    img_size=(96, 96, 96),
    in_channels=1,
    out_channels=2,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=False,
)

# Make sure all parameters are trainable
for param in model.parameters():
    param.requires_grad = True

# Create a dummy input tensor
input_tensor = torch.randn(1, 1, 96, 96, 96)

# Perform forward propagation with the modified model
try:
    output = model(input_tensor)
    print(f"Output: {output.shape}")
except RuntimeError as e:
    print(f"RuntimeError: {e}")
