from training.trainer import Trainer
import torch
import numpy as np
from training.loss import Loss

class UNetTrainer(Trainer):
    """
    Trainer class for UNet model.
    Inherits from the base Trainer class.
    """
    
    def __init__(self, model, optimizer, criterions, train_loader, val_loader, 
                 log_dir, chkp_dir, scheduler=None, device='cpu', epoch_goal=10):
        """
        Initialize UNetTrainer with same parameters as parent Trainer class.
        
        Args:
            model: UNet model to be trained
            config: Configuration object containing training parameters
            optimizer: Optimizer for training
            criterions: List of loss functions
            train_loader: Training data loader
            val_loader: Validation data loader
            log_dir: Directory for logging
            scheduler: Learning rate scheduler (optional)
            device: Device to run training on
            epoch_goal: Total number of epochs to train
            epoch_trained: Number of epochs already trained
        """
        super().__init__(
            model=model,
            name='unet',
            optimizer=optimizer,
            criterions=criterions,
            train_loader=train_loader,
            val_loader=val_loader,
            log_dir=log_dir,
            chkp_dir=chkp_dir,
            scheduler=scheduler,
            device=device,
            epoch_goal=epoch_goal,
        )

    def _calculate_loss(self, output, target, requires_grad):
        """
        Calculate loss for UNet model.
        """
        loss = torch.tensor(0.0, device=self.device, requires_grad=requires_grad)
        log_loss = {}
        for criterion in self.criterions:
            if isinstance(criterion, Loss):
                criterion_loss = criterion(output, target)
                loss = loss + criterion_loss
                log_loss[criterion.name] = criterion_loss.item()
        log_loss['total_loss'] = loss.item()
        return loss, log_loss