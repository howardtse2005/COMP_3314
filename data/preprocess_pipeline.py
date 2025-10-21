import cv2
import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from os.path import exists, join
from os import listdir, makedirs, remove
class Augmentation:
    # Base class for all augmentations
    def __init__(self, name:str, n_copy:int=1):
        '''
        Base class for augmentation modules in the preprocessing pipeline.
        Args:
            name (str): Name of the augmentation.
            n_copy (int): Number of copies to generate for each input image.
            use_raw (bool): If True, the output from the augmentation module will be generated base on raw images.
                by the end of the pipeline. All raw images would be ditched
                For example, if the input to an pipeline is 10, with 2 modules using n_copies=100,
                In first scenario, 
                set the first module use_raw=True and second to use_raw=True, the total output after 1st module will be 1000(10 * 100) 
                the total output after the 2nd module will be 2000(1000 + 100*10). So there will be 2000 by the end
                
                In Second sceneario,
                if the first module use_raw=True and second to use_raw=False, the total output after 1st module will be 1000(10 * 100)
                the total output after the 2nd module will be 100000 (1000 * 100)
        '''
        self.name = name
        self.n_copy = n_copy

class PreprocessPipeline:
    def __init__(self, temp_dir:str, keep_temp:bool = False, augmentations:list[Augmentation]=[]):
        self.augmentations = augmentations
        self.save_id = 0
        self.img_dir = temp_dir + '/imgs'
        self.mask_dir = temp_dir + '/masks'
        self.keep_temp = keep_temp
    
    def __call__(self, img, mask):        
        imgs_out, masks_out = [], []
        if not isinstance(img, np.ndarray) or not isinstance(mask, np.ndarray):
            raise TypeError("Input img and mask must be cv2 images")
        
        for i, aug in enumerate(self.augmentations):
            if isinstance(aug, Augmentation):
                if len(imgs_out) == 0:
                    imgs_out, masks_out = aug([img], [mask])
                else:
                    imgs_out, masks_out = aug(imgs_out, masks_out)
            else:
                raise TypeError(f"Unsupported augmentation type: {type(aug)}")
            
        if len(imgs_out) == 0 or len(masks_out) == 0:
            imgs_out, masks_out = [img], [mask]
        self._save(imgs_out, masks_out)

    def __del__(self):
        if not self.keep_temp:
            if exists(self.img_dir):
                for f in listdir(self.img_dir):
                    remove(join(self.img_dir, f))
            if exists(self.mask_dir):
                for f in listdir(self.mask_dir):
                    remove(join(self.mask_dir, f))
            print(f"Temporary files in {self.img_dir} and {self.mask_dir} have been deleted.")
        else:
            print(f"Temporary files in {self.img_dir} and {self.mask_dir} are kept as keep_temp is set to True.")
            
    def _save(self, imgs, masks):
        if not exists(self.img_dir):
            makedirs(self.img_dir)
        if not exists(self.mask_dir):
            makedirs(self.mask_dir)
        
        for i, (img, mask) in enumerate(zip(imgs, masks)):
            img_path = f"{self.img_dir}/{self.save_id}.png"
            mask_path = f"{self.mask_dir}/{self.save_id}.png"
            cv2.imwrite(img_path, img)
            cv2.imwrite(mask_path, mask)
            self.save_id += 1
    
class ElasticDeformation(Augmentation):
    """
    Elastic deformation augmentation as used in the original U-Net paper
    (Simard-style elastic distortions). Applies the same deformation to image and mask.
    """
    def __init__(self, alpha: float = 34.0, sigma: float = 4.0, n_copy: int = 20, p_apply: float = 1.0, seed: int | None = None, random_rotate: bool = False):
        super().__init__('ElasticDeformation', n_copy=n_copy)
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.p_apply = float(p_apply)
        self.rs = np.random.RandomState(seed)
        self.random_rotate = random_rotate

    def __call__(self, imgs: list[np.ndarray], masks: list[np.ndarray]):
        imgs_out, masks_out = [], []
        for img, mask in zip(imgs, masks):
            h, w = img.shape[:2]
            # Kernel size for Gaussian blur approximating sigma
            k = int(4 * self.sigma) * 2 + 1
            for _ in range(self.n_copy):
                if self.rs.rand() <= self.p_apply:
                    # Random rotation before deformation (if enabled)
                    if self.random_rotate:
                        # sample angle from [-180, 180) using the instance RNG for reproducibility
                        angle = float(self.rs.uniform(-180.0, 180.0))
                        M_rot = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
                        # rotate image and mask; use linear for image, nearest for mask
                        img_rot = cv2.warpAffine(img, M_rot, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                        mask_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
                        img_to_def = img_rot
                        mask_to_def = mask_rot
                    else:
                        img_to_def = img
                        mask_to_def = mask

                    # Random displacement fields
                    dx = (self.rs.rand(h, w).astype(np.float32) * 2 - 1)
                    dy = (self.rs.rand(h, w).astype(np.float32) * 2 - 1)
                    dx = cv2.GaussianBlur(dx, (k, k), self.sigma) * self.alpha
                    dy = cv2.GaussianBlur(dy, (k, k), self.sigma) * self.alpha

                    # Meshgrid and maps
                    x, y = np.meshgrid(np.arange(w), np.arange(h))
                    map_x = (x + dx).astype(np.float32)
                    map_y = (y + dy).astype(np.float32)

                    # Remap image (bilinear) and mask (nearest)
                    img_def = cv2.remap(img_to_def, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                    mask_def = cv2.remap(mask_to_def, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)

                    # Ensure mask stays binary
                    if mask_def.ndim == 3:
                        mask_def = cv2.cvtColor(mask_def, cv2.COLOR_BGR2GRAY)
                    if mask_def.max() > 1:
                        mask_def = ((mask_def > 127).astype(np.uint8)) * 255

                    imgs_out.append(img_def)
                    masks_out.append(mask_def)
                else:
                    imgs_out.append(img.copy())
                    masks_out.append(mask.copy())
        return imgs_out, masks_out