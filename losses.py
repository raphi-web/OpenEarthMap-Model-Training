import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    """Combined Dice Loss and Categorical Cross-Entropy Loss"""
    def __init__(self, dice_weight=0.5, ce_weight=0.5, mode='multiclass'):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = smp.losses.DiceLoss(mode=mode)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        ce = self.ce_loss(y_pred, y_true)
        return self.dice_weight * dice + self.ce_weight * ce


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss = smp.losses.FocalLoss("multiclass",alpha=alpha,gamma=gamma,reduction=reduction)

    def forward(self, y_pred, true):
       return self.loss(y_pred, true)

class TverskyLoss(nn.Module):
    """Tversky Loss - generalization of Dice loss"""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # weight for false positives
        self.beta = beta  # weight for false negatives
        self.smooth = smooth

        self.loss = smp.losses.TverskyLoss("multiclass",alpha=alpha,beta=beta,smooth=smooth)

    def forward(self, y_pred, y_true):
        y_pred = F.softmax(y_pred, dim=1)
        # One-hot encode targets

        return self.loss(y_pred, y_true)

class AdvancedCombinedLoss(nn.Module):
    """Advanced combined loss with multiple loss functions"""

    def __init__(self, dice_weight=0.4, ce_weight=0.3, focal_weight=0.2, tversky_weight=0.1):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)

    def forward(self, y_pred, y_true):
        dice = self.dice_loss(y_pred, y_true)
        ce = self.ce_loss(y_pred, y_true)
        focal = self.focal_loss(y_pred, y_true)
        tversky = self.tversky_loss(y_pred, y_true)

        total_loss = (self.dice_weight * dice +
                      self.ce_weight * ce +
                      self.focal_weight * focal +
                      self.tversky_weight * tversky)

        return total_loss
