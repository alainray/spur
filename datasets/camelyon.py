import json
import multiprocessing
from functools import partial
from easydict import EasyDict as edict
import h5py
import numpy as np
import wilds
from PIL import Image
# from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as tt
import torch


    
def get_dataset(args, split='train', **kwargs):

    datapath = args.dataset_paths['camelyon']
    a = wilds.get_dataset(root_dir=datapath, dataset="camelyon17", download=False)
    return  a.get_subset(split)


def get_splits(args, splits = ['train','val'], bs=128, **kwargs):
    def collate_fn(batch):
        x_batch, y_batch, metadata_batch = zip(*batch)  # Unpack the batch
        x_batch = torch.stack(x_batch)  # Convert x_batch to a tensor
        y_batch = torch.stack(y_batch)  # Convert y_batch to a tensor
        return x_batch, y_batch
    
    n_channels = 3

    tf = [tt.ToTensor(),
    tt.Normalize([0.5] * n_channels, [0.5] * n_channels)]
    
    transform = tt.Compose(tf)
    result = dict()
    
    for s in splits:
        ds = get_dataset(args, s)   
        ds.transform = transform
        result[s] = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    return result
    

if __name__ == '__main__':
    args = edict()
    args.spurious_probability = 0.5
    args.dataset_paths = {'camelyon': "../datasets"}
    #dataset = get_dataset(args,p=0.625, env="nobg",split='train',binarize=False)
#   print(dataset[0])
    dls = get_splits(args,splits = ['train','val','test'], bs=128)
    print(dls['test'])
    x,y  = next(iter(dls['test']))
    print(x.shape, y.shape)
   
