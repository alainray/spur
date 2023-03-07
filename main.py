import comet_ml
import torch
from datasets import dls
from params import *                        # Experiment parameters 
from models import create_model, restart_model
from torch.optim import SGD, Adam
from train import train, evaluate_splits
from utils import *

set_random_state(args)
# Define when to stop training and when to intervene.
training_schedule = create_schedule(train_args)
print(f'Current training schedule is: {training_schedule}. {len(training_schedule)-1} intervention(s) total.')

# Create model/optimizers
model=create_model(args).cuda()
opt = Adam(model.parameters(), lr=0.001)
opt_play  = Adam(model.parameters(), lr=0.001)

# Comet.ml logging
exp = setup_comet(args)
exp.log_parameters({k:w for k,w in vars(args).items() if "comet" not in k})
model.comet_experiment_key = exp.get_key() # To retrieve existing experiment
args.exp = exp
# Training loop
args.train_iter = 0
args.play_iter = 0
envs = ['train'
        , 'play'
        ]
splits = ['train', 'test']


if args.showData:
    show_data(dls, envs, splits)

# Evaluation before training
evaluate_splits(model, dls, envs, splits, args, "train")

# Training starts
args.train_iter = 1
args.play_iter = 1
for i, iters in enumerate(training_schedule):
    args.max_cur_iter = iters
    dl = dls['train']['train']
    print(f'TRAINING UP TO ITERATION {iters}')
    while args.train_iter < args.max_cur_iter:
        model, args, train_metrics = train(model, dl, opt, args,'train')
        # Evaluate on all environments/splits!
        evaluate_splits(model, dls, envs, splits, args, "train")
    print('TESTING')

    
    if 'forget' in train_args.mode and args.train_iter <= train_args.total_iterations:
        # Forget last layers before proceeding
        model = restart_model(args, model)


    if 'play' in train_args.mode and args.train_iter <= train_args.total_iterations:
        args.max_cur_iter = play_args.total_iterations
        # Replace classifier for play task
        #model = replaceModelClassifier(model, n_tasks)
        args.play_iter = 1
        dl = dls['play']['train']
        print(f"COMMENCING PLAY #{i+1}!")
        while args.play_iter < args.max_cur_iter:
            model, args, _ = train(model, dl, opt_play, args,'play')
            evaluate_splits(model, dls, envs, splits, args, "play")
        # Evaluate on all environments/splits!
        # Reattach original classifier head


