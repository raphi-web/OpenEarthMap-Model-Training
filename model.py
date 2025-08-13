import pytorch_lightning
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import CombinedLoss, AdvancedCombinedLoss, FocalLoss

class SegModel(pytorch_lightning.LightningModule):
    def __init__(self, n_classes: int, learning_rate: float, loss_type='combined', dropout_rate=0.1, weight_decay=1e-4,
                 gradient_clip_val=1.0):
        super().__init__()
        self.model = smp.create_model(
            arch='Unet',
            encoder_name='resnet34',
            in_channels=3,
            classes=n_classes)

        if loss_type == 'combined':
            self.loss_fn = CombinedLoss(dice_weight=0.5, ce_weight=0.5)
        elif loss_type == 'advanced':
            self.loss_fn = AdvancedCombinedLoss()
        elif loss_type == 'dice_focal':
            self.loss_fn = CombinedLoss(dice_weight=0.6, ce_weight=0.0)
            self.focal_loss = FocalLoss(alpha=1, gamma=2)
        else:
            self.loss_fn = smp.losses.DiceLoss(mode='multiclass')

        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.n_classes = n_classes
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val

    def forward(self, x):
        return self.model(x)

    def _validate(self, y_hat, y):
        pred_masks = torch.argmax(F.softmax(y_hat, dim=1), dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_masks,
            y.argmax(dim=1) if y.dim() > 3 else y,
            mode='multiclass',
            num_classes=self.n_classes)

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        return iou, f1_score, accuracy

    def _compute_loss(self, y_hat, y):
        y_target = y.argmax(dim=1).long() if y.dim() > 3 else y.long()
        if self.loss_type == 'dice_focal':
            dice_loss = smp.losses.DiceLoss(mode='multiclass')(y_hat, y_target)
            focal_loss = self.focal_loss(y_hat, y_target)
            loss = 0.6 * dice_loss + 0.4 * focal_loss
            return loss
        elif self.loss_type == 'advanced':
            loss, loss_components = self.loss_fn(y_hat, y_target)
            return loss
        else:
            return self.loss_fn(y_hat, y_target)

    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)

        loss = self._compute_loss(y_hat, y)
        iou, f1_score, accuracy = self._validate(y_hat, y)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        self.log('train_f1', f1_score, prog_bar=True)
        self.log('train_acc', accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        y_hat = self(x)

        loss = self._compute_loss(y_hat, y)
        iou, f1_score, accuracy = self._validate(y_hat, y)

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_iou', iou, prog_bar=True)
        self.log('val_f1', f1_score, prog_bar=True)
        self.log('val_acc', accuracy, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay)

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def on_before_optimizer_step(self, optimizer):
        """Gradient clipping"""
        if self.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_val)
