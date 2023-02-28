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
cur_iter = 1
for i, iters in enumerate(training_schedule):
    args.max_cur_iter = iters
    dl = dls['train']['train']
    print(f'TRAINING UP TO ITERATION {iters}')
    while cur_iter < args.max_cur_iter:
        model, cur_iter = train(model, dl, cur_iter, opt, args,'train_train')
    print('TESTING')
    # Evaluate on all environments/splits!
    evaluate_splits(model, dls, ['train','play'], ['train', 'test'], args)
    
    if 'forget' in train_args.mode and cur_iter <= train_args.total_iterations:
        # Forget last layers before proceeding
        model = restart_model(args, model)


    if 'play' in train_args.mode and cur_iter <= train_args.total_iterations:
        args.max_cur_iter = play_args.total_iterations
        # Replace classifier for play task
        #model = replaceModelClassifier(model, n_tasks)
        play_iter = 1
        dl = dls['play']['train']
        print(f"COMMENCING PLAY #{i+1}!")
        model, play_iter = train(model, dl, play_iter, opt_play, args,'play_train')
        
        # Evaluate on all environments/splits!
        print(f'TESTING AFTER PLAY #{i+1}!')
        evaluate_splits(model, dls, ['train','play'], ['train', 'test'], args)
        # Reattach original classifier head


