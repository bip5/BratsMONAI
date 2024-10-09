from monai.losses import DiceLoss,MaskedDiceLoss,DiceCELoss,DiceFocalLoss
from Input.config import loss_type
import torch
import monai
import torch.nn as nn
import torch.nn.functional as F

class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceCrossEntropyLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        true = true.float()
        probs = F.softmax(logits, dim=1)
        
        # Dice Loss Component
        intersection = torch.sum(probs * true, dim=[2, 3, 4])
        dice_numerator = 2.0 * intersection + self.smooth
        dice_denominator = torch.sum(probs ** 2, dim=[2, 3, 4]) + torch.sum(true ** 2, dim=[2, 3, 4]) + self.smooth
        dice_loss = 1 - dice_numerator / dice_denominator
        dice_loss = dice_loss.mean()

        # Cross Entropy Loss Component
        cross_entropy_loss = -torch.mean(true * torch.log(probs + 1e-8))  # Adding epsilon to avoid log(0)

        # Combined Loss
        combined_loss = dice_loss + cross_entropy_loss

        return combined_loss


def edgy_dice_loss(logits, targets):
    # Apply sigmoid to logits to convert to probabilities
    preds = torch.sigmoid(logits)  
    reduce_axis = torch.arange(2, len(logits.shape)).tolist() 
    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    intersection = torch.sum(preds * targets,dim=reduce_axis)
    
    # Calculate the Edgy Dice score
    ground_o = torch.sum(targets, dim=reduce_axis)
    pred_o = torch.sum(preds, dim=reduce_axis)
    denominator = ground_o + pred_o + 1e-5  # Adding a small epsilon for numerical stability
    neg_part = denominator - 2 * intersection
   
    # print(2*intersection, '2TP')
    # print(denominator, 'denomiantor')
    # print(neg_part,'neg_part')
    numerator = 2 * intersection - neg_part +1e-5
    
    # print(numerator,'numerator')
    edgy_dice_score = numerator / denominator
    
    # Loss is 1 minus the score to allow for minimization
    loss = 1 - edgy_dice_score
    loss_mean =loss.mean()
    # print(loss_mean, 'edgy dice loss')
    # original_dice =1 - (2*intersection+1e-5)/(denominator + 1e-5)
    # calc_dice = original_dice.mean()
    # print(calc_dice, 'calculated original dice loss')
    return loss_mean

def inv_dice_loss(logits, targets):
    # Apply sigmoid to logits to convert to probabilities
    preds = torch.sigmoid(logits)  
    reduce_axis = torch.arange(2, len(logits.shape)).tolist() 
    # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
    inner = torch.sum(preds * targets,dim=reduce_axis)
    
    # Calculate the Edgy Dice score
    ground_o = torch.sum(targets, dim=reduce_axis)
    pred_o = torch.sum(preds, dim=reduce_axis)
    union = ground_o + pred_o  # Adding a small epsilon for numerical stability
    outer = union - 2 * inner
   
    # print(2*intersection, '2TP')
    # print(denominator, 'denomiantor')
    # print(neg_part,'neg_part')
    numerator = 2 * inner - outer +1e-5
    denominator= 2*inner + 1e-5
    # print(numerator,'numerator')
    inverted_dice_score = numerator / denominator
    
    # Loss is 1 minus the score to allow for minimization
    loss = 1 - inverted_dice_score
    loss_mean = loss.mean()
    # print(loss_mean, 'edgy dice loss')
    # original_dice =1 - (2*intersection+1e-5)/(denominator + 1e-5)
    # calc_dice = original_dice.mean()
    # print(calc_dice, 'calculated original dice loss')
    return loss_mean
    

        
if loss_type == 'MaskedDiceLoss':
    loss_function = MaskedDiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
elif loss_type == 'CE':
    loss_function = DiceCELoss(include_background=True, to_onehot_y=False, softmax=True, other_act=None, squared_pred=True, jaccard=False, reduction='mean', smooth_nr=1e-05, smooth_dr=1e-05, batch=False, ce_weight=None, weight=None, lambda_dice=1.0, lambda_ce=1.0)#DiceCrossEntropyLoss() #
elif loss_type == 'EdgyDice': 
    loss_function = edgy_dice_loss
elif loss_type == 'InvDice': 
    loss_function = inv_dice_loss
elif loss_type=='DiceFocal':
    loss_function = DiceFocalLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=False)
else:
    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=False, to_onehot_y=False, sigmoid=False)
    
