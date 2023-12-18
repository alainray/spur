import datasets.synbols as synbols
from params import args
import torchvision
from torchvision.transforms import Resize
import torch
import datasets.camelyon as camelyon
import datasets.mnistcifar as mc
import datasets.waterbirds as wb
# Generic dataset

# Dataset options: 

dataset_function = {
    'cmnist': None,
    'synmnist': synbols.get_splits,
    'camelyon': camelyon.get_splits,
    'mnistcifar': mc.get_splits,
    'waterbirds': wb.get_splits,
}

def get_spurious_samples():
    train_dataset = torchvision.datasets.MNIST(root='../datasets', train=True, transform=None, download=True)
    resize = Resize((32,32))
    x = resize(train_dataset.data)
    y = train_dataset.targets
    sorted_indices = torch.argsort(y)
    # sort x using the sorted indices
    sorted_x = x[sorted_indices].float().unsqueeze(1).cuda()
    sorted_y = y[sorted_indices].float().cuda()

    z = torch.zeros_like(sorted_x)
    r_imgs = torch.cat([sorted_x,z,z], dim=1).cuda()
    g_imgs = torch.cat([z,sorted_x,z], dim=1).cuda()
    return r_imgs, g_imgs, sorted_y



def make_dataloaders(args): # entrega un dict con llaves los ambientes posibles ()

    dls = dict()
    task_dl = dict()
    eval_dl =  dict()
    # Create task dataloader
    #ds_name = args.task_args.dataset['name']
    #ds_options = {k: v for k,v in args.task_args.dataset.items() if k != 'name'}

    #dls['task'] = dataset_function[ds_name](args, **ds_options)
    
    # Create task dataloaders (only useful for IRM)
    for ds_id, task_ds in args.task_datasets.items():
        ds_name = task_ds['name']
        ds_options = {k: v for k,v in task_ds.items() if k != 'name'}
        task_dl[ds_id] = dataset_function[ds_name](args, **ds_options)

    dls['task'] = task_dl

    # Create play dataloader
    if 'play' in args.mode:
        ds_name = args.play_dataset['name']
        ds_options = {k: v for k,v in args.play_dataset.items() if k != 'name'}

    # Create eval dataloaders
    for ds_id, eval_ds in args.eval_datasets.items():
        ds_name = eval_ds['name']
        ds_options = {k: v for k,v in eval_ds.items() if k != 'name'}
        eval_dl[ds_id] = dataset_function[ds_name](args, **ds_options)
    
    dls['eval'] = eval_dl

    return dls


if __name__ == '__main__':

    print(make_dataloaders(args))