import torch
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize 
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from os.path import join
import pandas as pd

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")
        
class Waterbirds(Dataset):
    def __init__(self, root="", split="train", transform=None):
        self.root = root
        self.transform = transform
        self.splits = {'train': 0, 'val': 1, 'test': 2}
        meta = pd.read_csv(join(self.root,"metadata.csv"))
        self.data = meta[meta['split'] == self.splits[split]] # filter metadata by dataset split
        self.imgs = list(self.data['img_filename'])
        self.labels = list(self.data['y'])
        self.groups = list(10*self.data['y'] + self.data['place']) # first digit is label, second is spurious attribute
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        img_path = join(self.root,img)
        img = pil_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx], self.groups[idx]
    
def get_splits(args, splits=['train','val'], bs=128, **kwargs):

    n_channels = 3

    tf = [Resize((224,224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
    
    transform = Compose(tf)
    result = dict()
    
    for s in splits:
        ds = Waterbirds(root=args.dataset_paths['waterbirds'],split=s, transform=tf)
        ds.transform = transform
        result[s] = DataLoader(ds, batch_size=bs, shuffle=s=="train")
    return result


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()
    args.spurious_probability = 0.5
    args.dataset_paths = {'waterbirds': "../datasets/waterbird_complete95_forest2water2"}
    #dataset = get_dataset(args,p=0.625, env="nobg",split='train',binarize=False)
#   print(dataset[0])
    dls = get_splits(args,splits = ['train','val','test'], bs=128)
    print(dls['test'])
    x,y,g  = next(iter(dls['test']))
    print(x.shape, y.shape, g.shape)
   