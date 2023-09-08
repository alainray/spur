from comet_ml import Experiment, ExistingExperiment
import torch
from functools import wraps
from time import time
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt
import random

def pretty_print(m, args, mode):
    
    line = f"{args[f'{mode}_iter']-1:04d} |"
    min_split = {'train': "tr", "test": 'ts', 'val': 'vl','id': 'ivl'}
    for ds_name, metrics in m.items():

        line+= f"<{ds_name.upper()}>=[LOSS] "

        for k, v in metrics.items():
    
            split = k.split('_')[0]
            metric = k.split('_')[-1]
            if metric == 'loss':
                line += f"{min_split[split]}: {v:.3f} "
        line += "- [ACC] "
    
        for k, v in metrics.items():
            split = k.split('_')[0]
            metric = k.split('_')[-1]
            if metric == 'acc':
                line += f"{min_split[split]}: {v:.1f} "
        line += '|'
    print(line)
    
def show_data(dls, envs, splits):
    def plot_dataset(dataset, caption=""):
        # Create a grid of images from the training set
        images = [torch.cat((dataset[i],torch.zeros((1,28,28)))) for i in range(100)]
        grid = torchvision.utils.make_grid(torch.stack(images), nrow=10)

        # Display the grid of images
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.title(caption)
        plt.show()

    for env in envs:
        for split in splits:
            train_dataset = dls[env][split].dataset.data
            num=(dls[env][split].dataset.targets == 0).sum()
            num1=(dls[env][split].dataset.targets == 1).sum()
            plot_dataset(train_dataset,caption=f"{env}-{split} 0:{100*num/(num+num1):.2f}% 1:{100*num1/(num+num1):.2f}%")

def set_random_state(args):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    #torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)   

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"Func: {f.__name__} took: {te-ts:0.0f} sec")
        return result
    return wrap

# Comet Experiments
def setup_comet(args, resume_experiment_key=''):
    api_key = args.cometKey # w4JbvdIWlas52xdwict9MwmyH
    workspace = args.cometWs # alainray
    project_name = args.cometName # learnhard
    enabled = bool(api_key) and bool(workspace) and args.use_comet
    disabled = not enabled

    print(f"Setting up comet logging using: {{api_key={api_key}, workspace={workspace}, enabled={enabled}}}")

    if resume_experiment_key:
        experiment = ExistingExperiment(api_key=api_key, previous_experiment=resume_experiment_key)
        return experiment

    experiment = Experiment(api_key=api_key, parse_args=False, project_name=project_name,
                            workspace=workspace, disabled=disabled)
    # TEST
    experiment_name = get_prefix(args)
    if experiment_name:
        experiment.set_name(experiment_name)

    train_data_type = os.environ.get('TRAIN_DATA_TYPE')
    if train_data_type:
        experiment.add_tag(train_data_type)

    tags = os.environ.get('TAGS')
    if tags:
        experiment.add_tags(tags.split(','))

    return experiment

def create_schedule(args):
    step_size = args.total_iterations//args.n_interventions
    schedule = [(i+1)*step_size for i in range(args.n_interventions)] 

    if args.total_iterations not in schedule:
        schedule += [args.total_iterations]
    return schedule


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_grads(args, grads):
    path = f"{args.save_grads_folder}/grads_{args.model}_{args.task_dataset_name}_{args.task_dataset_bg}_{args.task_dataset_p}.pth"
    torch.save(grads, path)

def get_prefix(args):
    return "_".join([str(w) for k,w in vars(args).items() if "comet" not in k])

def make_dataset_id(ds_dict):
    if ds_dict['name'] == 'synmnist':
        return ds_dict['name'] + '_' + str(ds_dict['p']) + '_' + ds_dict['bg'] + '_' + ('bs' if ds_dict['baseline'] else 'nobs')
    elif ds_dict['name'] == 'mnistcifar':
        return ds_dict['name'] + "_" + str(ds_dict['corr']) + "_" + str(ds_dict['binarize'])

def save_best_model(args, model_dict):

    model_id = make_dataset_id(args.task_args.dataset)
    path = f'{args.save_model_folder}/{args.model}_{args.base_method}_{model_id}_{args.seed}_best_{args.save_model_path}'
    torch.save(model_dict, path)

def save_model(args, model, modifier=""):

    model_id = make_dataset_id(args.task_args.dataset)
    frozen = "frz" if args.frozen_features else 'nofrz'
    path = f'{args.save_model_folder}/{args.model}_{model_id}_{args.base_method}_{frozen}_{args.task_iter}_{args.seed}_{args.save_model_path}'
    torch.save(model.state_dict(), path)

def load_model(model, weights_path):
    w = torch.load(weights_path)
    model.load_state_dict(w)
    return model

def freeze_model(args, model): # Freeze all layers except classifier
    # Freeze all the layers in the features module
    if args.model == 'scnn':
        layers = ['features.conv1.weight',
                  'features.conv1.bias',
                  'features.conv2.weight',
                  'features.conv2.bias',
                  'features.conv3.weight',
                  'features.conv3.bias',
                  'fc.0.weight',
                  'fc.0.bias']
        layers = layers[:2*args.n_freeze_layers]
        for n, param in model.features.named_parameters():
            if n in layers:
                param.requires_grad = False

        # Make sure the parameters in the classifier module are trainable
        for param in model.fc.parameters():
            param.requires_grad = True
    
    return model