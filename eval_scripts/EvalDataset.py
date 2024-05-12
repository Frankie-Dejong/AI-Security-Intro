import torch 
from torch.utils.data import Dataset
from os.path import join as pjoin
import os
from tqdm import tqdm as tqdm
from PIL import Image
import numpy as np
import json
import torch.nn.functional as F

class ImageDataset(Dataset):
    def __init__(self, path, transforms=None, phase='GT', eval_size=50, batch_idx=0):
        self.path = path
        self.transforms = transforms
        self.phase = phase
        
        print(f"Load Pictures from {self.path}")
        self.images = []
        
        if eval_size is None:
            eval_size = len(os.listdir(path))
        for files in sorted(os.listdir(path))[batch_idx * eval_size:batch_idx * eval_size + eval_size]:
            self.images.append(pjoin(path, files)) 
            
        print(f"There are total {len(self.images)} samples in {phase} set")
        
    def __len__(self):
        return len(self.images)
    
    def load_image(self, path):
        img = Image.open(path).convert('RGB')
        return img
    
    def __getitem__(self, index):
        name = self.images[index]
        image = self.load_image(name)
        if self.transforms is None:
            image = np.array(image).astype(np.uint8)
        else:
            image = self.transforms(image)
        
        return image
        
        
class ImageTextPairDataset(Dataset):
    def __init__(self, image_path, text_path, transforms=None, phase='GT', eval_size=50, batch_idx=0):
        self.image_path = image_path
        self.text_path = text_path
        self.transforms = transforms
        self.phase = phase
        
        print(f"Load Pictures from {self.image_path}")
        self.images = []
        
        if eval_size is None:
            eval_size = len(os.listdir(self.image_path))
        for files in sorted(os.listdir(self.image_path))[batch_idx * eval_size:batch_idx * eval_size + eval_size]:
            self.images.append(pjoin(self.image_path, files)) 
        
        print(f"There are total {len(self.images)} samples in {phase} set")
        
        print(f"self.text_path is {self.text_path}")
        assert self.text_path.endswith('.json')
        with open(text_path, mode='r') as file:
            annotations = json.load(file)
        self.annotation = annotations
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        name = self.images[index]
        image = Image.open(name)
        if self.transforms is None:
            pass
        else:
            image = self.transforms(image)
        image_idx = name.split('/')[-1].split('.')[0]
        text = self.annotation[image_idx+'.jpg']['Beard_and_Age']
        return image, text
        
            
class MaskPairDataset(Dataset):
    def __init__(self, mask_path, prompt_path, phase='Test'):
        self.mask_path = mask_path
        self.prompt_path = prompt_path
        
        print(f"Load Masks from {self.mask_path}")
        self.masks = []
        
        for files in sorted(os.listdir(self.mask_path)):
            self.masks.append(pjoin(self.mask_path, files))
            
        print(f"There are total {len(self.masks)} samples in {phase} set")
        
    def __len__(self):
        return len(self.masks)
    
    def __getitem__(self, index):
        name = self.masks[index]
        mask = Image.open(name)
        mask = mask.resize((32, 32), Image.NEAREST)
        flattened_mask = list(mask.getdata())
        flattened_mask = torch.tensor(flattened_mask)
        flattened_mask = np.array(flattened_mask)
        
        index = name.split('/')[-1].split('.')[0]
        prompt_mask = torch.load(pjoin(self.prompt_path, index+'.pt'))
        prompt_mask = torch.argmax(prompt_mask, dim=0)
        prompt_mask = np.array(prompt_mask).astype(np.uint8)
        return flattened_mask, prompt_mask
    
