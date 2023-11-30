#!/usr/bin/env python
# coding: utf-8

# In[1]:


from easydict import EasyDict  as edict
import torch
import sys
from torch.optim import Adam
from tqdm.notebook import tqdm
from dataset import make_dataloaders
from train import train,evaluate_splits
from models import create_model
from numpy.random import choice
from torch.utils.data import Subset, DataLoader
from utils import update_metrics, save_stats
def create_args(exp_params):
    args = edict()
    task_args = edict()
    args.exp_id="X"
    args.seed = exp_params['seed']
    # --------------- MODEL ---------------------
    args.model = 'scnn'
    args.hidden_dim = 100
    args.output_dims = 1  # Equals number of classes
    
    # ---------- TRAINING PARAMS ----------------
    args.load_pretrained = True                                    # Do we start with a pretrained model or not 
    args.pretrained_path = f"models/scnn_{exp_params['method']}_{exp_params['dataset']}_{exp_params['corr']}_True_{exp_params['seed']}_best_.pth" # path to pretrained model
    task_args.n_interventions = 0                                  # Amount of times we stop training before applying intervention (forgetting, playing)
    task_args.total_iterations = exp_params['max_iters']                         # 9452 = 1 epoch
    
    # ---------- MODEL PERSISTENCE --------------
    args.save_model = False                                        # Save last model
    args.save_best = False                                          # Save best performing model
    args.save_stats = True                                         # Save final performance metrics
    args.save_model_folder = 'models'                              # Folder where models are stored 
    args.save_grads_folder = 'grads'                               # Folder where gradients are saved to
    args.save_model_path = ".pth"                                  # suffix for saved model (name depends on model settings)
    args.use_comet = False
    
    # ------------------- TRAINING METHOD ------------------------------
    args.max_cur_iter = 0
    args.task_iter = 0
    args.mode = ["task"                                             # task = train on dataset defined in task_args 
                   #, 'play'                                        #  play = train on dataset defined in play_args 
                   #,'forget'                                       # forget = after training on task, forget using method defined in args.forget_method
                       ]
    args.base_method = "erm"                                      # gdro = group distributionally robust optimization
                                                                    # rw = reweight losses
                                                                    # erm = Empirical Risk Minimization 
    
    # --------- DATASET -----------------------------------------------------------------------------
    args.eval_datasets = dict()                                    # Which datasets to evaluate
    args.task_datasets = dict()     
    args.dataset_paths = {'synmnist': "../datasets/SynMNIST",      # Path for each dataset
                          'mnistcifar': "../datasets/MNISTCIFAR"}
    args.task_datasets['env1'] = {'name': exp_params['dataset'], 'corr': float(exp_params['corr'])
                                  , 'splits': ['train', 'test'], 'bs': 10000, "binarize": True}
    
    # All datasets listed on eval_datasets will be evaluated. One dataset per key, however, each dataset may evaluate multiple splits.
    for ds_id, ds in args.task_datasets.items():
        args.eval_datasets[f'task_{ds_id}'] = ds 
    args.eval_datasets['eval'] = {'name': exp_params['dataset'], 'corr': 0.0, 'splits': ['val'], 'bs': 50000, "binarize": True}
    # -------- METRICS -----------------------------------------------------------------------------
    args.metrics = ['acc', 'loss','worst_group_loss', 'worst_group_acc', "best_group_loss", "best_group_acc"]
    # --------------- Consolidate all settings on args --------------------
    args.task_args = task_args
    return args

def load_model(model, weights_path):

    w = torch.load(weights_path)
    s_dict = w['model']
    s_dict2 = dict()
    for k, v in s_dict.items():
        if k in ['fc.0.weight','fc.0.bias']:
            s_dict2[k.replace("0.","")] = v
        else:
            s_dict2[k] = v
        
    model.load_state_dict(s_dict2, strict=False)
    return model


# In[2]:



# load a model
# load balanced dataset
# define dataset size (hyperparameter)
# finetune model on balanced dataset
# report metrics (worst_group_*, best_group_*, acc, loss)
# create table with data at the method/dataset/spur/seed level then aggregate metho/dataset/spur

def run_experiment(exp_params):
    print(exp_params)
    all_metrics = {'task_env1': dict(), 'eval': dict()}
    args = create_args(exp_params)
    for k in all_metrics.keys():
        for split in ["train", "val", "test"]:
            for m in args.metrics:
                all_metrics[k][f"{split}_{m}"] = []
    
    # define args for dataloader
    model=create_model(args).cuda()
    model = load_model(model, args.pretrained_path).cuda()
    opt = Adam(model.parameters(), lr=0.001)#),momentum=0.9,weight_decay=0.01)
    # reload datasets
    dls = make_dataloaders(args)
    dl = dls['task']['env1']['test']              #train on balanced version of dataset
    n_samples = len(dl.dataset)
    print(n_samples)
    random_indices = choice(n_samples, exp_params["ft_size"])
    dl = DataLoader(Subset(dl.dataset, indices=random_indices),batch_size=10000,shuffle=True) # Get subset of dataset
    for i in tqdm(range(args.task_args.total_iterations),total=args.task_args.total_iterations):
        model,_,_ = train(model,dl,opt,args)
        metrics = evaluate_splits(model,dls['eval'],args,"task")
        # accumulate metrics
        for ds_name, m in metrics.items():
            all_metrics[ds_name] = update_metrics(all_metrics[ds_name], m)
    
    return args, all_metrics # {"worst_group}


# In[ ]:


from os.path import join
from os import listdir
def choose_experiments(method, model_dir = "models"):
    def make_file_dict(f):
        f = f.split("_")
        return {
                'model': f[0],
                'method': f[1],
                'dataset': f[2],
                'corr': f[3],
                'seed': f[5]
               }
    files = []
    for f in listdir(model_dir):
        if method in f:
            files.append(make_file_dict(f))
    return files

exps = choose_experiments("erm", model_dir="models")
max_iters = 2000
size_of_ft = 1000
for size_of_ft in [10, 50, 100, 200, 500, 1000]:
    for e in tqdm(exps,total=len(exps)):
        e['max_iters'] = max_iters
        e['ft_size'] = size_of_ft
        print(e)
        args, results = run_experiment(e)
        args.base_method +=f"_ft_{size_of_ft}"
        save_stats(args,results,root="stats")