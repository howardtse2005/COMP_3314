from data.dataset import Dataset
from model.hnet import HNet
from model.unet import UNet 
from training.trainer_unet import UNetTrainer
import cv2
from tqdm import tqdm
import numpy as np
import torch
import os
import datetime
from config import Config as cfg
import torch.nn.functional as F

from testing.benchmark import calculate_metrics
from testing.visualization import create_visualization, write_evaluation_results

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

test_img_path = cfg.dir_img_test
test_mask_path = cfg.dir_mask_test
checkpoint_path = cfg.pretrained_model


#--------------------- Main Test Function ---------------------

def test(test_data_path='data/test_example.txt',
         save_path='results/images',
         eval_path='results/eval',
         pretrained_model=checkpoint_path,
         test_img_path = test_img_path,
         test_mask_path = test_mask_path,
         threshold=0.5):
    
    # Create timestamp for folder names
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    os.makedirs(eval_path, exist_ok=True)
    result_folder_name = f"evaluation_{timestamp}"
    timestamped_save_path = os.path.join(save_path, result_folder_name)
    os.makedirs(timestamped_save_path, exist_ok=True)
    
    print(f"Results will be saved to: {timestamped_save_path}")
    
    # Load dataset
    test_dataset = Dataset(
        dataset_img_path=test_img_path,
        dataset_mask_path=test_mask_path,
        temp_dir=cfg.dir_temp_ts,
    )

    # Build model and trainer
    device = torch.device("cuda")
    num_gpu = torch.cuda.device_count()
    
    if cfg.model_type == 'unet':
        model = UNet()
        print("Using UNet architecture")
        print("Using Attention UNet architecture") 
    elif cfg.model_type == 'hnet':
        model = HNet()
        print("Using HNet architecture")
    else:
        print("Unknown model type for testing")
        
    model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
    model.to(device)
    
    trainer = UNetTrainer(
        model=model,
        optimizer=None,  
        criterions=None,
        train_loader=None,
        val_loader=None,
        log_dir=None,  
        chkp_dir=None
        ).to(device)

    model.load_state_dict(trainer.checkpointer.load(pretrained_model, multi_gpu=True))
    model.eval()
    
    # Store predictions and ground truths
    all_predictions = []
    all_groundtruths = []
    
    print("Processing full-resolution images...")
    
    # Process each image
    for idx in tqdm(range(len(test_dataset))):
        # Get image and mask from dataset
        img_tensor, gt_tensor = test_dataset[idx]
        
        # Prepare input batch - add batch dimension
        img_batch = img_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
        
        # Get ground truth
        gt_np = gt_tensor.numpy().astype(np.float32)
        
        # Perform inference
        with torch.no_grad():
            if cfg.model_type in ['hnet', 'unet', 'attention_unet', 'segformer']:
                pred = torch.sigmoid(model(img_batch))
            else:
                pred = torch.sigmoid(model(img_batch)[0])
            
            # Remove batch dimension and move to CPU
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy()
        
        # Store results
        all_predictions.append(pred_np)
        all_groundtruths.append(gt_np)
        
        # Create visualization - convert image tensor back to numpy for visualization
        img_vis = img_tensor.permute(1, 2, 0).numpy() * 255.0
        img_vis = img_vis.astype(np.uint8)
        
        visualization = create_visualization(img_vis, gt_np, pred_np)
        save_name = os.path.join(timestamped_save_path, f"fullres_{idx:04d}.png")
        cv2.imwrite(save_name, visualization)
    
    # Calculate all metrics
    print(f"Calculating metrics for {len(all_predictions)} images...")
    metrics_results = calculate_metrics(all_predictions, all_groundtruths, threshold)
    
    # Write evaluation results
    eval_file = os.path.join(eval_path, f"{result_folder_name}.txt")
    write_evaluation_results(eval_file, metrics_results, timestamp, pretrained_model, timestamped_save_path)
    
    print(f"Evaluation results saved to {eval_file}")
    print(f"Images saved to {timestamped_save_path}")


if __name__ == '__main__':
    test()