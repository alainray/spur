import torch
from torch.utils.data import TensorDataset
from easydict import EasyDict as edict
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
def mnist_cifar(root, split, binarize=False):

    all_splits = torch.load(root)
    sp = split
    if split == "id":
        sp = 'train'
    ds = all_splits[sp]
    if binarize:
        ds['targets'] = (ds['targets'] > 4).float()

    dataset = TensorDataset(ds['data'], ds['targets'], ds['group'])

    if split in ['train', 'id']:
        generator1 = torch.Generator().manual_seed(42)
        dss = random_split(dataset, [9000, 1000], generator=generator1 )
        if split == 'train':
            return dss[0]
        elif split == 'id':
            return dss[1]

    return dataset

def get_dataset(args, split, corr, **kwargs):

    datapath = args.dataset_paths['mnistcifar']
    if args.output_dims == 10:
        datapath += f"/MNIST_CIFAR_{corr}.pth"
    else:
        datapath += f"/MNIST_CIFAR_binary_{corr}.pth"
    #print(datapath)
    a = mnist_cifar(datapath, split=split)
    return  a


def get_splits(args, splits=['train','val'], bs=10000, **kwargs):

    n_channels = 3

    tf = [tt.ToTensor(),
    tt.Normalize([0.5] * n_channels, [0.5] * n_channels)]
    
    transform = tt.Compose(tf)
    result = dict()
    
    for s in splits:
        ds = get_dataset(args, s, kwargs['corr'])   
        ds.transform = transform
        result[s] = DataLoader(ds, batch_size=bs, shuffle=True)
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
   