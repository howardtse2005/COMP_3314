import data.preprocess_pipeline as pp
from torch.utils.data import Dataset
from os.path import exists, join
import os, cv2
import numpy as np
import torch, tqdm
class Dataset(Dataset):
    """
    Create a torch Dataset from data
    """

    def __init__(self, dataset_img_path:str, dataset_mask_path:str, augmentations:list[pp.Augmentation]=[],
                    temp_dir:str='data/temp', keep_temp:bool=False):
        super().__init__()
        self.pp = pp.PreprocessPipeline(
            temp_dir=temp_dir,
            keep_temp=keep_temp,
            augmentations=augmentations
        )
        if not exists(dataset_img_path):
            raise FileNotFoundError(f"Dataset path {dataset_img_path} does not exist.")
        if not exists(dataset_mask_path):
            raise FileNotFoundError(f"Dataset path {dataset_mask_path} does not exist.")
        
        self.image_files, self.mask_files = self._get_files(dataset_img_path, dataset_mask_path)
        with tqdm.tqdm(total=len(self.image_files), desc="Loading dataset") as pbar:
            for img_file, mask_file in zip(self.image_files, self.mask_files):
                img_path = join(dataset_img_path, img_file)
                mask_path = join(dataset_mask_path, mask_file)
                img, mask = self._read_image(img_path, mask_path)
                self.pp(img, mask)
                pbar.update(1)
        self.img_dir = os.path.join(temp_dir, 'imgs')
        self.mask_dir = os.path.join(temp_dir, 'masks')
        self.output_img_files = sorted(os.listdir(self.img_dir))
        self.output_mask_files = sorted(os.listdir(self.mask_dir))
        if len(self.output_img_files) != len(self.output_mask_files):
            raise ValueError("Number of images and masks do not match after preprocessing.")
        

    
    def __len__(self):
        return len(self.output_img_files)
    
    def __getitem__(self, index):
        if index < 0 or index >= len(self.output_img_files):
            raise IndexError("Index out of range.")

        img = cv2.imread(os.path.join(self.img_dir, self.output_img_files[index]))
        mask = cv2.imread(os.path.join(self.mask_dir, self.output_mask_files[index]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask {self.output_mask_files[index]} could not be read.")
        if img is None:
            raise ValueError(f"Image {self.output_img_files[index]} could not be read.")
        
        img = img.astype(np.float32) / 255.0  # normalize image to [0, 1]
        
        if mask.max() >= 1:  # normalize mask to [0, 1]
            mask = (mask > 127).astype(np.float32)
        else:
            mask = mask.astype(np.float32)
            
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).float()
        return img, mask # img out is (C, H, W) and mask out is (H, W) in tensor

    def _get_files(self, img_dir, mask_dir):
        '''
        Get all image and mask files in the specified directories. and check validity
        '''
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        mask_files = [f for f in os.listdir(mask_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sort files by numeric order instead of lexicographic order for compatibility with older results
        def numeric_sort_key(filename):
            import re
            # Extract numeric part from filename
            numbers = re.findall(r'\d+', filename)
            return int(numbers[0]) if numbers else 0
        
        image_files.sort(key=numeric_sort_key)
        mask_files.sort(key=numeric_sort_key)
        
        if not image_files:
            raise FileExistsError(f"No image files found in {img_dir}")
        if not mask_files:
            raise FileExistsError(f"No mask files found in {mask_dir}")
        # Remove the length check since we'll filter for matching pairs
        return image_files, mask_files
    
    
    def _read_image(self, img_path, mask_path):
        '''
        Read an image and its corresponding mask. And check if they are valid.
        images are assumed to be binary in single channel
        '''
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Image {img_path} could not be read.")
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Mask {mask_path} could not be read.")
        if img.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Image {img_path} and mask {mask_path} dimensions do not match.")
        return img, mask
    
    def get_num_imgs(self):
        return len(self.image_files)