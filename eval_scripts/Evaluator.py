import torch
import abc
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
from EvalDataset import ImageDataset, ImageTextPairDataset, MaskPairDataset
from torch.utils.data import DataLoader
from tqdm import tqdm as tqdm
from InceptionV3 import InceptionV3
import numpy as np
from scipy import linalg
import clip

FID_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class Evaluator(ABC):
    def __init__(self, val_set):
        pass
    
    @abstractmethod
    def evaluate(self):
        pass
    
class FIDEvaluator(Evaluator):
    def __init__(self, 
                 gt_path, 
                 pred_path, 
                 eval_size=50,
                 batch_idx=0,
                 transforms=FID_transform, 
                 batch_size=50, 
                 num_workers=4, 
                 device='cuda'):
        self.gt_path = gt_path
        self.pred_path = pred_path
        self.model = InceptionV3(device)
        self.gt_set = ImageDataset(gt_path, transforms, 'GT', eval_size=eval_size, batch_idx=batch_idx)
        self.pred_set = ImageDataset(pred_path, transforms, 'Pred', eval_size=eval_size, batch_idx=batch_idx)
        self.gt_loader = DataLoader(self.gt_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        self.pred_loader = DataLoader(self.pred_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        self.device = device
        self.batch_size = batch_size
        
    @torch.no_grad()    
    def get_activations(self, dataloader, dims=2048):
        res = np.empty((len(dataloader) * self.batch_size, dims))
        idx = 0
        self.model.eval()
        for batch in tqdm(dataloader):
            batch = batch.to(self.device)
            pred = self.model(batch).squeeze(3).squeeze(2).cpu().numpy()
            res[idx:idx+pred.shape[0]] = pred
            idx += pred.shape[0]
        return res
        
    def calculate_fid(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)
        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)
        assert mu1.shape == mu2.shape
        assert sigma1.shape == sigma2.shape
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
            
        tr_covmean = np.trace(covmean)
        return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)
        
        
    def evaluate(self):
        gt_feats = self.get_activations(self.gt_loader)
        pred_feats = self.get_activations(self.pred_loader)
        gt_mu = np.mean(gt_feats, axis=0)
        gt_sigma = np.cov(gt_feats, rowvar=False)
        pred_mu = np.mean(pred_feats, axis=0)
        pred_sigma = np.cov(pred_feats, rowvar=False)
        assert pred_feats.shape == gt_feats.shape
        return self.calculate_fid(gt_mu, gt_sigma, pred_mu, pred_sigma)
    
    
class ClipEvaluator(Evaluator):
    def __init__(self,  
                 pred_path,
                 text_path, 
                 eval_size=50,
                 batch_idx=0,
                 transforms=None, 
                 batch_size=1, 
                 num_workers=4, 
                 device='cuda'):
        self.pred_path = pred_path
        self.text_path = text_path
        
        self.pred_set = ImageTextPairDataset(pred_path, text_path, transforms, 'Pred', eval_size=eval_size, batch_idx=batch_idx)
        self.device = device
        self.batch_size = batch_size
        
        model, preprocess = clip.load('ViT-B/32', device=device)
        model.eval()
        self.model = model
        self.preprocess = preprocess
    
    @torch.no_grad()
    def calculate_sim(self):
        res = []
        for sample in tqdm(self.pred_set):
            image = sample[0]
            text = sample[1]
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            text = clip.tokenize([text]).to(self.device)
            
            image_feats = self.model.encode_image(image)
            text_feats = self.model.encode_text(text)
    
            image_feats = image_feats.cpu().numpy()
            text_feats = text_feats.cpu().numpy()
            
            image_feats = image_feats / np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
            text_feats = text_feats / np.sqrt(np.sum(text_feats ** 2, axis=1, keepdims=True))
            
            sim = image_feats @ text_feats.T
            
            res.append(np.mean(sim))
        return sum(res) / len(res), max(res)

    
    def evaluate(self):
        return self.calculate_sim()


class MaskAccuracyEvaluator(Evaluator):
    def __init__(self, mask_path, prompt_path):
        self.dataset = MaskPairDataset(mask_path, prompt_path)
        
    def evaluate(self):
        accs = []
        for mask, prompt in self.dataset:
            acc = sum(mask==prompt) / (mask.shape[-1])
            accs.append(acc)
        return sum(accs) / len(accs)