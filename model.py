import pytorch_lightning
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
import torch.optim as optim

from losses import CombinedLoss, AdvancedCombinedLoss, FocalLoss, TverskyLoss


class SegModel(pytorch_lightning.LightningModule):
    # Model configuration constants

    def __init__(
            self, model, n_classes: int,
            learning_rate: float,
            data_loader_len: int,
            epochs: int,
            loss_type='combined',
            dropout_rate=0.1,
            weight_decay=1e-4,
            gradient_clip_val=1.0):
        super().__init__()

        self.model = model
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.data_loader_len = data_loader_len
        self.epochs = epochs
        self.loss_type = loss_type
        self.loss_fn = self._create_loss_function(loss_type)

    def _create_loss_function(self, loss_type):
        """Create and configure the loss function based on the specified type."""
        loss_functions = {
            'combined': lambda: CombinedLoss(),
            'advanced': lambda: AdvancedCombinedLoss(),
            'focal': lambda: FocalLoss(),
            'tversky': lambda: TverskyLoss()
        }

        if loss_type in loss_functions:
            return loss_functions[loss_type]()
        else:
            return smp.losses.DiceLoss(mode='multiclass')

    def forward(self, x):
        return self.model(x)

    def _validate(self, y_pred, y):
        pred_masks = torch.argmax(F.softmax(y_pred, dim=1), dim=1)
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_masks,
            y.argmax(dim=1) if y.dim() > 3 else y,
            mode='multiclass',
            num_classes=self.n_classes)

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")

        return iou, f1_score, accuracy

    def _compute_loss(self, y_pred, y):
        y_target = y.argmax(dim=1).long() if y.dim() > 3 else y.long()
        return self.loss_fn(y_pred, y_target)

    def _compute_step(self, batch):
        """Common computation logic for training and validation steps."""
        x, y = batch['x'], batch['y']
        y_pred = self(x)
        loss = self._compute_loss(y_pred, y)
        iou, f1_score, accuracy = self._validate(y_pred, y)
        return loss, iou, f1_score, accuracy

    def _log_metrics(self, prefix, loss, iou, f1_score, accuracy):
        """Log metrics with the specified prefix."""
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        self.log(f'{prefix}_iou', iou, prog_bar=True)
        self.log(f'{prefix}_f1', f1_score, prog_bar=True)
        self.log(f'{prefix}_acc', accuracy, prog_bar=True)

    def training_step(self, batch, batch_idx):
        loss, iou, f1_score, accuracy = self._compute_step(batch)
        self._log_metrics('train', loss, iou, f1_score, accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou, f1_score, accuracy = self._compute_step(batch)
        self._log_metrics('val', loss, iou, f1_score, accuracy)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01,
            steps_per_epoch=self.data_loader_len,
            epochs=self.epochs)

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
