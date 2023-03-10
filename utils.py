from comet_ml import Experiment, ExistingExperiment
import torch
from functools import wraps
from time import time
import numpy as np
import os
import torchvision
import matplotlib.pyplot as plt


def pretty_print(metrics, args, mode):
    line = f"={args[f'{mode}_iter']-1:04d}=[LOSS] "
    for k,v in metrics.items():
        for k1, v1 in v.items():
            if "loss" in k1:
                line += f"{k.upper()}: {v1:.3f} "
    line += "- [ACC] "
    for k,v in metrics.items():
        for k1, v1 in v.items():
            if 'acc' in k1:
                line += f"{k.upper()}: {v1:.2f}% "
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
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
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
    enabled = bool(api_key) and bool(workspace)
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
    step_size = args.total_iterations//(args.n_interventions+1)
    schedule = [i*step_size for i in range(1,args.n_interventions+1)] 

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

def get_prefix(args):
    return "_".join([str(w) for k,w in vars(args).items() if "comet" not in k])