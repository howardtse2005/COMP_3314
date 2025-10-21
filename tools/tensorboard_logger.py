import os
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime


class TensorBoardLogger:
    def __init__(self, log_dir='runs', exp_name=''):
        """
        TensorBoard Logger to replace Visdom
        
        Args:
            log_dir: Base directory for TensorBoard logs
            exp_name: Experiment name; if None, uses current timestamp
        """

        self.exp_name = exp_name + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')

        self.log_dir = os.path.join(log_dir, self.exp_name)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.epoch = 0
        self.loss_history = {}  # Store loss history by epoch
        self.loss_export_dir = 'results/loss'
        os.makedirs(self.loss_export_dir, exist_ok=True)
        print(f'TensorBoard logs will be saved to {self.log_dir}')
    
    def set_epoch(self, epoch):
        """Set the current epoch for logging."""
        self.epoch = epoch
        
    def add_scalar(self, tag, value):
        """Add scalar value to TensorBoard."""
        self.writer.add_scalar(tag, value, self.epoch)
        
        # Store value for loss curve plotting
        if 'loss' in tag.lower():
            if tag not in self.loss_history:
                self.loss_history[tag] = []
            
            # Check if we already have a value for this epoch
            if not self.loss_history[tag] or self.loss_history[tag][-1][0] != self.epoch:
                self.loss_history[tag].append((self.epoch, value))
            else:
                # Update the last value for this epoch
                self.loss_history[tag][-1] = (self.epoch, value)
    
    def add_scalars(self, main_tag, tag_value_dict):
        """Add multiple scalars with the same main tag."""
        self.writer.add_scalars(main_tag, tag_value_dict, self.epoch)
        
        # Store values for loss curve plotting
        for tag, value in tag_value_dict.items():
            full_tag = f"{main_tag}/{tag}"
            if full_tag not in self.loss_history:
                self.loss_history[full_tag] = []
            
            # Check if we already have a value for this epoch
            if not self.loss_history[full_tag] or self.loss_history[full_tag][-1][0] != self.epoch:
                self.loss_history[full_tag].append((self.epoch, value))
            else:
                # Update the last value for this epoch
                self.loss_history[full_tag][-1] = (self.epoch, value)
    
    def add_image(self, tag, img_tensor):
        """Add image to TensorBoard."""
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)
        
        self.writer.add_image(tag, img_tensor, self.epoch)
    
    def add_images(self, tag, img_tensor):
        """Add batch of images to TensorBoard."""
        if isinstance(img_tensor, np.ndarray):
            img_tensor = torch.from_numpy(img_tensor)
        
        self.writer.add_images(tag, img_tensor, self.epoch)
    
    def log(self, message, tag='info'):
        """Log a text message (printed to console as there's no direct text logging in TensorBoard)."""
        print(f"[{tag}] {message}")
    
    def log_dict(self, dict_obj, tag):
        """Log a dictionary of values to TensorBoard."""
        for key, value in dict_obj.items():
            if isinstance(value, (int, float)):
                self.add_scalar(f"{tag}/{key}", value)

    def plot_many(self, tag_value_dict, main_tag="metrics"):
        """Plot multiple values with a common main tag."""
        self.add_scalars(main_tag, tag_value_dict)
    
    def img_many(self, tag_img_dict):
        """Add multiple images to TensorBoard."""
        for tag, tensor in tag_img_dict.items():
            if tensor.dim() == 4:  # Batch of images
                self.add_images(tag, tensor)
            else:  # Single image
                self.add_image(tag, tensor)
    
    def export_loss_curves(self, filename=None):
        """Export loss curves as JPG images with epochs on the x-axis."""
        if not self.loss_history:
            return None
        
        if filename is None:
            filename = f'{self.exp_name}.jpg'
        else:
            filename = f'{filename}.jpg'
        
        save_path = os.path.join(self.loss_export_dir, filename)
        
        # Create subplots based on number of loss categories
        num_losses = len(self.loss_history)
        fig, axes = plt.subplots(num_losses, 1, figsize=(10, 3 * num_losses))
        
        if num_losses == 1:
            axes = [axes]  # Make it iterable for single subplot
        
        # Plot each loss curve
        for ax, (loss_name, history) in zip(axes, self.loss_history.items()):
            epochs, values = zip(*history) if history else ([], [])
            ax.plot(epochs, values)
            ax.set_title(loss_name)
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.grid(True)
            
            # Try to keep integer ticks on x-axis if reasonable
            if len(epochs) <= 20:  # only for reasonable number of epochs
                ax.set_xticks(list(epochs))
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
        print(f"Loss curves exported to {save_path}")
        return save_path
    

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
