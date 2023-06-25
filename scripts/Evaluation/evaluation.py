import sys
sys.path.append('/scratch/a.bip5/BraTS 2021/scripts/')

from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
import torch
from Input.config import (
VAL_AMP,
)

dice_metric = DiceMetric(include_background=True, reduction="mean")
dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

def inference(input,model):

    def _compute(input,model):
        return sliding_window_inference(
            inputs=input,
            roi_size=(192,192, 144),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input,model)
    else:
        return _compute(input,model)