from comet_ml import Experiment, ExistingExperiment
import torch
from functools import wraps
from time import time
import numpy as np
import os


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