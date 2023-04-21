import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn as nn
import clip
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import requests
import os
from typing import Set
from utils import mutual_information_2d
#from dataset import PrecompDataset
from torchvision.datasets import CocoCaptions
from torchvision.transforms import ToTensor
import time
import itertools

import torch.nn as nn
import os
import subprocess
from abc import ABC
from typing import Tuple, Callable, Any, Union, List, Dict

import clip as oai_clip
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

import src
from src.slip import CLIP_VITB16, CLIP_VITL16, SLIP_VITB16, SLIP_VITL16, SimpleTokenizer
from src.video_transformer import SpaceTimeTransformer

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
import tqdm
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import src
from src import PATHS, Dotdict
from src.data_utils import DebiasExperiment, get_neptune_run
from src.datasets import Flickr30K
from src.metrics import t2v_metrics

from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim

class Adversary(nn.Module):

    @staticmethod
    def from_cfg(cfg: Union[dict, src.Dotdict]):
        cfg = src.Dotdict(cfg)
        adv_model = Adversary(n_input=cfg.ADV_N_INPUT, n_output=cfg.ADV_N_OUTPUT, hidden_size=cfg.ADV_HIDDEN_SIZE)
        return adv_model.to(cfg.ADV_DEVICE)

    def __init__(self, n_input, n_output=1, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_output),
        )

    def forward(self, x):
        return torch.sigmoid(self.network(x))
    
class BiasClassifier(nn.Module):
    def __init__(self, n_input, n_class=3, hidden_size=32):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_input, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_class),
        )

    def forward(self, x):
        return self.network(x)    

def load_occ1_data(gender_occ_df, data_dir, model, tokenizer=None, transform=None, device='cuda'):
    images = []
    occups = []
    genders = []
    CONTEXT_LEN = model.clip.context_length
    
    for index, e_row in tqdm.tqdm(gender_occ_df.iterrows()):
        occup = str(e_row['search_term'])
        file_name = str(e_row['order']) + '.jpg'
        image = Image.open(os.path.join(data_dir, occup, file_name)).convert('RGB')
        image_gender = e_row.image_gender

        image_gender = 1 if image_gender == 'man' else 2 if image_gender == 'woman' else 0

        occups.append(clip.tokenize(occup).to(device))
        images.append(transform(image).to(device))
        genders.append(image_gender)
    images  = torch.stack(images)
    # load text tensor 
    unique_occups = []
    for e_occup in gender_occ_df['search_term'].unique():
        unique_occups.append(tokenizer(e_occup).to(device))

    text_input = torch.zeros(len(unique_occups), CONTEXT_LEN, dtype=torch.long)
    for i, e_occup in enumerate(unique_occups):
        text_input[i, :len(e_occup[0])] = e_occup[0]
    text_input = text_input.to(device)

    # the gender labels of images
    # 0 represents gender-neutral
    # 1 represents male
    # 2 represents female
    attrs = torch.from_numpy(np.asarray(genders))  

    batch_size = 256
    total_batch = images.size()[0]//batch_size + 1
    with torch.no_grad():
        image_features = []
        text_features = []
        for i in range(total_batch):
            image_features.append(model.encode_image(images[batch_size * i:batch_size * (i+1)]).float())
            text_features.append(model.encode_text(text_input[batch_size * i:batch_size * (i+1)]).float())
        image_features = torch.cat(image_features, dim=0)
        text_features = torch.cat(text_features, dim=0)
    
    return image_features, text_features, attrs

def fairness_intervention_ranking(sim, idx, labels, top_n=100):
    argsrt_idx = sim[idx].argsort()[::-1]
    
    idx_male = np.where(labels == 1)[0]
    argsrt_idx_male = [e_idx for e_idx in argsrt_idx if e_idx in idx_male]
    
    idx_female = np.where(labels == 2)[0]
    argsrt_idx_idx_female = [e_idx for e_idx in argsrt_idx if e_idx in idx_female]

    #idx_neutral = np.where(labels == 0)[0]
    
    ret_list = list(itertools.chain(*zip(argsrt_idx_male, argsrt_idx_idx_female)))
    
    ret_list = ret_list[:top_n]
    return ret_list

def dcg_weight_bias(bias, k):
    # position-aware metric
    r = np.ones_like(bias)[:k]
    w =  1.0/ np.log2(np.concatenate([np.asarray([2.0]), np.arange(2, r.size + 1)]))
    #print(w)
    return np.mean(w * bias)

def mean_dcg_weight_bias(bias_list, k):
    return np.mean([np.abs(dcg_weight_bias(b, k)) for b in bias_list])

def meanBias(bias_list, k):
    return np.mean([np.abs(np.mean(b[:k])) for b in bias_list])

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def mean_precision(rs, k):
    return np.mean([precision_at_k(r, k) for r in rs])
    
    
def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs, k):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r[:k]) for r in rs])

def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max

def mndcg_k(rs, k, method=0):
    return np.mean([ndcg_at_k(r, k, method) for r in rs])

COCO_ROOT = 'data/val2014'
try:
    DATASET = CocoCaptions(root='data/val2014', annFile = 'data/COCO/annotations/captions_val2014.json')
except:
    pass

    #CONTEXT_LEN = model.clip.context_length
class image_caps_attr_dataset(Dataset):
    def __init__(self, images,caps, attrs, transform=None, context_len=77, device='cuda:0'):
        
        self.img_paths = images
        self.cap_paths = caps
        self.attr_paths = attrs
        self.transform = transform
        self.context_len = context_len
        self.device = device
        self.images, self.caps, self.attrs = self._get_data()
        
    def _get_data(self):
        images = []
        captions = []
        with open(self.cap_paths, 'r') as f:
            lines = f.readlines()
            for line in lines:
                caption = line.strip()
                captions.append(clip.tokenize(caption).to(self.device))

        captions = captions[::5]
            

        ids = []
        with open(self.img_paths, 'r') as f:
            lines = f.readlines()
            for line in lines:
                i = int(line.strip())
                if i not in ids:
                    ids.append(i)
        
        # load image tensor            
        count = 0
        for img_id in ids:
            coco = DATASET.coco
            path = coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(COCO_ROOT, path)).convert('RGB')
            images.append(self.transform(image).to(self.device))
            count += 1
            
        # load text tensor 
        text_input = torch.zeros(len(captions), self.context_len, dtype=torch.long)
        for i, caption in enumerate(captions):
            text_input[i, :len(caption[0])] = caption[0]
        text_input = text_input.to(self.device)
        
        # the gender labels of images
        # 0 represents gender-neutral
        # 1 represents male
        # 2 represents female
        with open(self.attr_paths, 'rb') as f:
            attrs = np.load(f)
        attrs = torch.from_numpy(attrs)        
        
        return images, text_input, attrs
        
    def __len__(self):
        return len(self.caps)

    def __getitem__(self, idx):
        image = self.images[idx] # Image from PIL module
        cap = self.caps[idx]
        attr = self.attrs[idx]
        return {'img': image, 'text': cap, 'attrs': attr}
    
# step 1: match number of bag size images with input text, matched images are a bag
# step 2: aggregate these representations and attribute labels
# this simulate the real-world user feedbacks about social bias.

def bagging_by_first_matching_clip(image_features, text_features, images, text, attribute_labels, bag_size = 10):
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features  /= text_features.norm(dim=-1, keepdim=True)
        image_features_np = image_features.cpu().numpy()
        text_features_np = text_features.cpu().numpy()
        sim = text_features_np @ image_features_np.T
        
        #print(image_features)
        npt = sim.shape[0] # sim: (N, N) matrix
        #image_features_bag = []
        bag_labels = []
        image_bags = []
        text_bags = []
        for i in range(npt):
            inds = np.argsort(sim[i])[::-1]
            #ranks[i] = np.where(inds == i)[0][0]
            bag_inds = inds[:bag_size]
            attribute_in_bag = attribute_labels[bag_inds.copy()]
            images_in_bag = images[bag_inds.copy()]
            text_in_bag = text[bag_inds.copy()]
            image_bags.append(images_in_bag.cpu())
            text_bags.append(text_in_bag.cpu())
            
            att1 = torch.sum(attribute_in_bag == 1)
            att2 = torch.sum(attribute_in_bag == 2)
            if torch.abs(att1-att2) <= 1:
                bag_labels.append(0)
            else:
                bag_labels.append(1)
                
            #bag_rep = image_features[bag_inds.copy(), :].mean(0, keepdim=True)
     
            #image_features_bag.append(bag_rep)
        #image_features_bag = torch.cat(image_features_bag, dim=0)
        image_bags = torch.stack(image_bags)
        text_bags = torch.stack(text_bags)

    return image_bags, text_bags, bag_labels

def random_bagging(images, text, attribute_labels, bag_size = 10):
    with torch.no_grad():
        #print(image_features)
        #image_features_bag = []
        total_bags = int(np.ceil(images.size()[0]//bag_size))
        bag_labels = []
        image_bags = []
        text_bags = []
        shuffle_idx = np.random.permutation(images.size()[0])
        attribute_labels = attribute_labels[shuffle_idx]
        images = images[shuffle_idx]
        text = text[shuffle_idx]

        for i in range(total_bags):
            attribute_in_bag = attribute_labels[bag_size * i:bag_size * (i+1)]
            images_in_bag = images[bag_size * i:bag_size * (i+1)]
            text_in_bag = text[bag_size * i:bag_size * (i+1)]
            image_bags.append(images_in_bag)
            text_bags.append(text_in_bag)
            
            att1 = torch.sum(attribute_in_bag == 1)
            att2 = torch.sum(attribute_in_bag == 2)
            if torch.abs(att1-att2) <= 1:
                bag_labels.append(0)
            else:
                bag_labels.append(1)
        image_bags = torch.stack(image_bags)
        text_bags = torch.stack(text_bags)

    return image_bags, text_bags, bag_labels

def original_order_bagging(images, text, attribute_labels, bag_size = 10):
    with torch.no_grad():

        total_bags = int(np.ceil(images.size()[0]//bag_size))
        bag_labels = []
        image_bags = []
        text_bags = []
        for i in range(total_bags):
            attribute_in_bag = attribute_labels[bag_size * i:bag_size * (i+1)]
            images_in_bag = images[bag_size * i:bag_size * (i+1)]
            text_in_bag = text[bag_size * i:bag_size * (i+1)]
            image_bags.append(images_in_bag)
            text_bags.append(text_in_bag)
            
            att1 = torch.sum(attribute_in_bag == 1)
            att2 = torch.sum(attribute_in_bag == 2)
            if torch.abs(att1-att2) <= 1:
                bag_labels.append(0)
            else:
                bag_labels.append(1)
        image_bags = torch.stack(image_bags)
        text_bags = torch.stack(text_bags)

    return image_bags, text_bags, bag_labels

# def bag_function_specific_bagging(images, text, attribute_labels, bag_size = 10):
#     with torch.no_grad():
#         # produce gender-balanced bags.
        



class SYNMIL_image_caps_attr_dataset(Dataset):
    def __init__(self, images,caps, attrs, init_model, bag_size=10, transform=None, context_len=77, bagging_mode='original_order', device='cuda:0'):
        
        self.img_paths = images
        self.cap_paths = caps
        self.attr_paths = attrs
        self.transform = transform
        self.context_len = context_len
        self.device = device
        self.model = init_model
        self.bag_size = bag_size
        self.bagging_mode = bagging_mode
        self.image_bags, self.text_bags, self.bag_labels = self._get_data()
        
    def _get_data(self):
        images = []
        captions = []
        with open(self.cap_paths, 'r') as f:
            lines = f.readlines()
            for line in lines:
                caption = line.strip()
                captions.append(clip.tokenize(caption).to(self.device))

        captions = captions[::5]
            
        
        ids = []
        with open(self.img_paths, 'r') as f:
            lines = f.readlines()
            for line in lines:
                i = int(line.strip())
                if i not in ids:
                    ids.append(i)
        
        # load image tensor            
        count = 0
        for img_id in ids:
            coco = DATASET.coco
            path = coco.loadImgs(img_id)[0]['file_name']
            image = Image.open(os.path.join(COCO_ROOT, path)).convert('RGB')
            images.append(self.transform(image).to(self.device))
            count += 1
        images   = torch.stack(images)
        # load text tensor 
        text_input = torch.zeros(len(captions), self.context_len, dtype=torch.long)
        for i, caption in enumerate(captions):
            text_input[i, :len(caption[0])] = caption[0]
        text_input = text_input.to(self.device)
        
        # the gender labels of images
        # 0 represents gender-neutral
        # 1 represents male
        # 2 represents female
        with open(self.attr_paths, 'rb') as f:
            attrs = np.load(f)
        attrs = torch.from_numpy(attrs)        
        
        batch_size = 256
        total_batch = images.size()[0]//batch_size + 1
        with torch.no_grad():
            image_features = []
            text_features = []
            for i in range(total_batch):
                image_features.append(self.model.clip.encode_image(images[batch_size * i:batch_size * (i+1)]).float())
                text_features.append(self.model.clip.encode_text(text_input[batch_size * i:batch_size * (i+1)]).float())
            image_features = torch.cat(image_features, dim=0)
            text_features = torch.cat(text_features, dim=0)
        
        if self.bagging_mode == 'matching':
            image_bags, text_bags, bag_labels = bagging_by_first_matching_clip(image_features, text_features, images, text_input, attrs, self.bag_size)
        elif self.bagging_mode == 'random':
            image_bags, text_bags, bag_labels = random_bagging(images, text_input, attrs, self.bag_size)
        elif self.bagging_mode == 'original_order':
            image_bags, text_bags, bag_labels = original_order_bagging(images, text_input, attrs, self.bag_size)
            
        return image_bags, text_bags, bag_labels
        
    def __len__(self):
        return len(self.bag_labels)

    def __getitem__(self, idx):
        image = self.image_bags[idx] # Image from PIL module
        cap = self.text_bags[idx]
        attr = self.bag_labels[idx]
        return {'img': image, 'text': cap, 'attrs': attr}
    
def evaluate(similarity, gender):
    """
    Evaluate the recall and bias performance
    Input:
        similarity: A numpy array of shape [N, N].
    Output:
        recall: Recall@1, Recall@5, Recall@10
        bias:   Bias@[1...10]
    """
    npt = similarity.shape[0]
    ranks = np.zeros(npt)
    male = np.zeros(npt)
    female = np.zeros(npt)

    for i in range(npt):
        inds = np.argsort(similarity[i])[::-1]
        ranks[i] = np.where(inds == i)[0][0]

    # recall
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # bias
    biases = []
    for k in range(1, 11):
        for i in range(npt):
            inds = np.argsort(similarity[i])[::-1]
            inds = inds[:k]
            male[i] = (gender[inds] == 1).sum()
            female[i] = (gender[inds] == 2).sum()

        bias = (male - female) / (male + female + 1e-12)
        bias = bias.mean()
        biases.append(bias)

    return (r1, r5, r10), biases

def eval_bias(image_features, text_features, gender):
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features  /= text_features.norm(dim=-1, keepdim=True)
    image_features = image_features.cpu().numpy()
    text_features = text_features.cpu().numpy()
    sim = text_features @ image_features.T

    recall, bias = evaluate(sim, gender)
#     print("Recall: ", recall)
#     print("Bias: ", bias)
#     print(f"Time cost: {time.time()-begin}")
    return recall, bias

def load_image_text_features(model, transform=None, context_len=77, device="cuda"):
    #load test features 
    images = []
    captions = []

    ids = []
    with open('SCAN/data/coco_precomp/testall_ids.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            i = int(line.strip())
            if i not in ids:
                ids.append(i)

    with open('SCAN/data/coco_precomp/testall_caps.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            caption = line.strip()
            captions.append(clip.tokenize(caption).to(device))

    captions = captions[::5]
    count = 0
    for img_id in ids:
        coco = DATASET.coco
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(COCO_ROOT, path)).convert('RGB')
        images.append(transform(image).to(device))
        count += 1

    test_images   = torch.stack(images)
    test_text_input = torch.zeros(len(captions), context_len, dtype=torch.long)
    for i, caption in enumerate(captions):
        test_text_input[i, :len(caption[0])] = caption[0]
    test_text_input = test_text_input.to(device)


    test_batch_size = 256
    total_batch = test_images.size()[0]//test_batch_size + 1
    with torch.no_grad():
        test_image_features = []
        test_text_features = []
        for i in range(total_batch):
            test_image_features.append(model.encode_image(test_images[test_batch_size * i:test_batch_size * (i+1)]).float())
            test_text_features.append(model.encode_text(test_text_input[test_batch_size * i:test_batch_size * (i+1)]).float())
        test_image_features = torch.cat(test_image_features, dim=0)
        test_text_features = torch.cat(test_text_features, dim=0)
    return test_image_features, test_text_features


def compute_con_loss(model, imgs, text, con_weight, device):
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    logits_per_image, logits_per_text = model.clip(imgs, text)
    ground_truth = torch.arange(len(imgs),dtype=torch.long,device=device)
    total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
    return con_weight * total_loss

def train_step_adv(
    batch,
    model_cl,
    model_adv,
    optimizer_cl,
    optimizer_adv,
    loss_fn_adv,
    debias_class: str,
    optim="adv",
    orig_embeddings: torch.Tensor = None,
    regu_weight: float = None,
    regu_p: int = None,
    adv_type = 'sim',
    con_weight = None,
    device = 'cuda'
):
    """

    :param batch:
    :param text:
    :param model_cl:
    :param model_adv:
    :param optimizer_cl:
    :param optimizer_adv:
    :param loss_fn_adv:
    :param optim:
    :param orig_embeddings:
    :param regu_weight:
    :return:
    """
    imgs = batch["img"]
    text = batch["text"]
    if optim == "adv":
        with torch.no_grad():
            if adv_type == 'rep':
                adv_inp = torch.cat([model_cl.encode_image(imgs), model_cl.encode_text(text)], dim=1)
            elif adv_type == 'sim':
                adv_inp = model_cl(imgs, text) / 20
    else:
        model_cl.zero_grad()
        if adv_type == 'rep':
            adv_inp = torch.cat([model_cl.encode_image(imgs), model_cl.encode_text(text)], dim=1)
        elif adv_type == 'sim':
            adv_inp = model_cl(imgs, text) / 20

    # unsqueeze because model takes in a scalar
    model_adv.zero_grad()
    adv_pred = model_adv(adv_inp.float())
    # unsqueeze and repeat the labels because it's the same text et al
    sensitive_categorical = batch["attrs"]
    # minimize loss between
    loss = loss_fn_adv(adv_pred, sensitive_categorical)
    reg_loss = None
    con_loss = None
    # Backpropagation
    if optim == "cl":
        loss = -loss # confuse adversary
        
        if con_weight != 0: # contrastive learning loss, keep representation
            con_loss = compute_con_loss(
                model_cl, imgs, text, con_weight, device)
            if con_loss is not None:
                loss += con_loss
                con_loss = con_loss.item()
        else:
            con_loss = 0
            
        # no reg loss 
        #         if regu_weight != 0:
        #             reg_loss = compute_reg_loss(
        #                 model_cl, text, orig_embeddings, regu_weight, regu_p
        #             )
        #             if reg_loss is not None:
        #                 loss += reg_loss
        #                 reg_loss = reg_loss.item()
        #         else:
        #             reg_loss = 0

        optimizer_cl.zero_grad()
        loss.backward()
        optimizer_cl.step()
    else:
        optimizer_adv.zero_grad()
        loss.backward()
        optimizer_adv.step()
    # calc acc
    acc = (adv_pred.argmax(axis=-1) == sensitive_categorical).sum() / adv_pred.shape[0]
    return acc.item(), loss.item(), con_loss, reg_loss


def train_step_MIL_adv(
    batch,
    model_cl,
    model_adv,
    optimizer_cl,
    optimizer_adv,
    loss_fn_adv,
    debias_class: str,
    bag_size = 10,
    optim="adv",
    orig_embeddings: torch.Tensor = None,
    regu_weight: float = None,
    regu_p: int = None,
    adv_type = 'rep',
    con_weight = None,
    device = 'cuda'
):
    """

    :param batch:
    :param text:
    :param model_cl:
    :param model_adv:
    :param optimizer_cl:
    :param optimizer_adv:
    :param loss_fn_adv:
    :param optim:
    :param orig_embeddings:
    :param regu_weight:
    :return:
    """
    imgs = batch["img"]
    text = batch["text"]
    imgs = imgs.view(imgs.size(0) * imgs.size(1), imgs.size(2), imgs.size(3), imgs.size(4))
    text = text.view(text.size(0) * text.size(1), text.size(2))
    if optim == "adv":
        with torch.no_grad():
            if adv_type == 'rep':
                image_feat = model_cl.encode_image(imgs)
                text_feat = model_cl.encode_text(text)
                
                total_bags = int(np.ceil(image_feat.size()[0]//bag_size))
                image_features_bag = []
                text_features_bag = []
                for i in range(total_bags):
                    image_features_bag.append(image_feat[bag_size * i:bag_size * (i+1), :].mean(0, keepdim=True))
                    text_features_bag.append(text_feat[bag_size * i:bag_size * (i+1), :].mean(0, keepdim=True))
                image_features_bag = torch.cat(image_features_bag, dim=0)
                text_features_bag = torch.cat(text_features_bag, dim=0)
                adv_inp = torch.cat([image_features_bag, text_features_bag], dim=1)
            elif adv_type == 'sim':
                adv_inp = model_cl(imgs, text) / 20
    else:
        model_cl.zero_grad()
        if adv_type == 'rep':
            image_feat = model_cl.encode_image(imgs)
            text_feat = model_cl.encode_text(text)

            total_bags = int(np.ceil(image_feat.size()[0]//bag_size))
            image_features_bag = []
            text_features_bag = []
            for i in range(total_bags):
                image_features_bag.append(image_feat[bag_size * i:bag_size * (i+1), :].mean(0, keepdim=True))
                text_features_bag.append(text_feat[bag_size * i:bag_size * (i+1), :].mean(0, keepdim=True))
            image_features_bag = torch.cat(image_features_bag, dim=0)
            text_features_bag = torch.cat(text_features_bag, dim=0)
            adv_inp = torch.cat([image_features_bag, text_features_bag], dim=1)
        elif adv_type == 'sim':
            adv_inp = model_cl(imgs, text) / 20

    # unsqueeze because model takes in a scalar
    model_adv.zero_grad()
    adv_pred = model_adv(adv_inp.float())
    # unsqueeze and repeat the labels because it's the same text et al
    sensitive_categorical = batch["attrs"]
    # minimize loss between
    loss = loss_fn_adv(adv_pred, sensitive_categorical)
    reg_loss = None
    con_loss = None
    # Backpropagation
    if optim == "cl":
        loss = -loss # confuse adversary
        
        if con_weight is not None: # contrastive learning loss, keep representation
            con_loss = compute_con_loss(
                model_cl, imgs, text, con_weight, device)
            if con_loss is not None:
                loss = (1 - con_weight) * loss + con_weight * con_loss
                #loss += con_loss
                con_loss = con_loss.item()
        else:
            con_loss = 0
            
        # no reg loss 
        #         if regu_weight != 0:
        #             reg_loss = compute_reg_loss(
        #                 model_cl, text, orig_embeddings, regu_weight, regu_p
        #             )
        #             if reg_loss is not None:
        #                 loss += reg_loss
        #                 reg_loss = reg_loss.item()
        #         else:
        #             reg_loss = 0

        optimizer_cl.zero_grad()
        loss.backward()
        optimizer_cl.step()
    else:
        optimizer_adv.zero_grad()
        loss.backward()
        optimizer_adv.step()
    # calc acc
    acc = (adv_pred.argmax(axis=-1) == sensitive_categorical).sum() / adv_pred.shape[0]
    return acc.item(), loss.item(), con_loss, reg_loss