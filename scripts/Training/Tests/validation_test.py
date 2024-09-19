import pytest
import torch
from monai.data import DataLoader

from Input.config import(
training_mode,
loss_type,

)
from Input.dataset import IslesDataset, make_atlas_dataset
from Input.localtransforms import val_transform_isles
from Evaluation.evaluation import(
dice_metric,
dice_metric_batch,
workers
)
from Training.training import (
device,
validate

)


def test_simplified_validate():
    # Simplified test just to ensure pytest runs
    result = "simplified test"
    assert result == "simplified test"
    
@pytest.fixture
def mock_val_loader():
    def loader():
    
        full_val = IslesDataset("/scratch/a.bip5/BraTS/dataset-ISLES22^public^unzipped^version"  ,transform=val_transform_isles )
        val_loader=DataLoader(full_val, batch_size=1, shuffle=False,num_workers=workers)
        sample = next(iter(val_loader))
        
        print(type(sample),'type(sample)')
        
        yield sample
        
    return loader()
    
@pytest.fixture
def mock_globals(monkeypatch):
    monkeypatch.setattr("Training.training.device",torch.device("cuda:0"))

def test_validate_with_identical_inputs(mock_val_loader,mock_globals):
    
    # custom inference to return idential outputs
    def custom_inference(val_data):
        return [val_data["mask"],val_data["mask"]]
     
    _,best_metric,_,metric = validate(mock_val_loader, 0,0,0,custom_inference=custom_inference)
    
    assert metric>0.99, f"Dice score should be identical but got a score of {metric}"
    assert best_metric>0.99, f"Best metric for dice score should be >0.99 since we're testing identical mask but got a score of {best_metric}"
    
    print(f"Test passed. Dice score: {metric}, Best metric: {best_metric}")
    
    
# def test_make_dataset_atlas(mock_globals):
    # data_dir = '/scratch/a.bip5/ATLAS_2/Training/'
    
    
    