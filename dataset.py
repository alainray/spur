import datasets.synbols as synbols
from collections import defaultdict
from params import args
# Generic dataset

# Dataset options: 

dataset_function = {
    'cmnist': None,
    'synmnist': synbols.get_splits
}

def make_dataloaders(args): # entrega un dict con llaves los ambientes posibles ()

    dls = dict()
    eval_dl =  dict()
    # Create task dataloader
    ds_name = args.task_args.dataset['name']
    ds_options = {k: v for k,v in args.task_args.dataset.items() if k != 'name'}

    dls['task'] = dataset_function[ds_name](args, **ds_options)
           
    # Create play dataloader
    if 'play' in args.task_mode:
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