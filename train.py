import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

import data.preprocess_pipeline as pp
from data.dataset import Dataset
from config import Config as cfg
from model.hnet import HNet
from model.unet import UNet
from training.loss import FocalWithLogitsLoss, DiceWithLogitsLoss, BCEWithLogitsLoss
from training.trainer_unet import UNetTrainer
from training.trainer_hnet import HNetTrainer
from training.loss import PixelWiseCrossEntropyLoss

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#<---------Initialize model ------------>
if cfg.model_type == 'unet':
    print("Using UNet model")
    model = UNet()
elif cfg.model_type == 'hnet':
    print("Using HNet model")
    model = HNet()
else:
    raise ValueError(f"Unknown model type: {cfg.model_type}")

if cfg.use_checkpoint:
    print(f"Loading pretrained model from: {cfg.pretrained_model}")
    model.load_state_dict(torch.load(cfg.pretrained_model, map_location=device))
    
model.to(device)

#<---------Initialize optimizer ------------>
# Use SGD as Adam was not released yet in original U-Net paper
print("Using SGD optimizer")
optimizer = optim.SGD(model.parameters(), **cfg.sgd_params) 
#<---------Initialize scheduler ------------>
scheduler = None
#<---------Initialize loss functions ------------>
criterions = []
print("Using Pixel-wise Cross Entropy loss")
criterions.append(PixelWiseCrossEntropyLoss(**cfg.ce_params))
#<---------Initialize data loaders ------------>
augmentations_train = []

# Add elastic deformations before resize
if getattr(cfg, 'use_elastic', False):
    augmentations_train.append(
        pp.ElasticDeformation(
            alpha=cfg.elastic_alpha,
            sigma=cfg.elastic_sigma,
            n_copy=cfg.elastic_n_copies,
            p_apply=cfg.elastic_prob,
            random_rotate=cfg.random_rotate
        )
    )
    
train_dataset = Dataset(
    dataset_img_path=cfg.dir_img_tr,
    dataset_mask_path=cfg.dir_mask_tr,
    augmentations=augmentations_train,
    temp_dir=cfg.dir_temp_tr,
    keep_temp=cfg.keep_temp_tr
)
print(f"Dataset length: {len(train_dataset)}")
print(f"Number of images: {train_dataset.get_num_imgs()}")

# Remove validation set to match original U-Net paper
val_loader = None

# Create train data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=1)

#<---------Initialize trainer ------------>
trainer_params = {
    'model': model,
    'optimizer': optimizer,
    'criterions': criterions,
    'train_loader': train_loader,
    'val_loader': val_loader,  # no validation
    'log_dir': cfg.log_path,
    'chkp_dir': cfg.checkpoint_path,
    'scheduler': scheduler,
    'device': device,
    'epoch_goal': cfg.epoch
}
if cfg.model_type == 'unet':
    trainer = UNetTrainer(**trainer_params)
elif cfg.model_type == 'hnet':
    trainer = HNetTrainer(**trainer_params)
else:
    raise ValueError(f"Unknown model type: {cfg.model_type}")

trainer.train()