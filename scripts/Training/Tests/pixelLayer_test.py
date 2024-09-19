import torch
import pytest
from Training.pixelLayer import PixelLayer
from Training.network import model as base_model
from Input.config import roi
# Assuming PixelLayer and SegResNet are imported

device= torch.device('cuda:0')
@pytest.fixture
def setup_model():
    """Fixture to set up the PixelLayer model with dummy base model"""    
    roi= (192,192,144)
    channels = 4
    model = PixelLayer(base_model=base_model, roi=roi, channels=channels)
    return model

def test_weight_shape_mismatch(setup_model):
    """Test that the shape of the weights matches the expected shape"""
    model = setup_model
    input_shape = (model.channels, *model.roi)
    weights_shape = model.alpha.shape

    assert weights_shape == input_shape, f"Expected weight shape {input_shape}, but got {weights_shape}"

def test_weight_elements_not_all_same(setup_model):
    """Test that the weights are not all the same"""
    model = setup_model
    alpha_values = model.alpha.detach().cpu().numpy()
    unique_values = len(set(alpha_values.flatten()))

    assert unique_values > 1, "All weight elements are the same, but expected randomness in initialization."

def test_output_shape(setup_model):
    """Test that the output shape matches the input shape"""
    model = setup_model.to(device)
    input_tensor = torch.rand((1, model.channels, *model.roi)).to(device)  # Batch size = 1

    output_tensor = model(input_tensor)
    
    input_shape = list(input_tensor.shape)
    expected_output_shape = input_shape
    expected_output_shape[1] = output_shape[1]-1
    assert expected_output_shape == list(output_tensor.shape), f"Expected output shape {expected_output_shape}, but got {output_tensor.shape}"

