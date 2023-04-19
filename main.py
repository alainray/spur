import comet_ml
import torch
from dataset import make_dataloaders
from params import *                        # Experiment parameters 
from models import create_model, restart_model
from torch.optim import SGD, Adam
from train import train, evaluate_splits
from utils import *
from time import time
def main(dls):
    global args
    global task_args
    print(f"===Using Seed={args.seed}===")
    set_random_state(args)
    # Define when to stop training and when to intervene.
    training_schedule = create_schedule(task_args)
    print(f'Current training schedule is: {training_schedule}. {len(training_schedule)-1} intervention(s) total.')

    # Create model/optimizers
 
    model=create_model(args).cuda()
    if args.load_pretrained:
        model = load_model(model, args.pretrained_path).cuda()
    
    if args.frozen_features:
        model = freeze_model(args, model).cuda()

    opt = Adam(model.parameters(), lr=0.001)
    opt_play  = Adam(model.parameters(), lr=0.001)

    # Comet.ml logging
    exp = setup_comet(args)
    exp.log_parameters({k:w for k,w in vars(args).items() if "comet" not in k})
    
    model.comet_experiment_key = exp.get_key() # To retrieve existing experiment
    args.exp = exp
    # Training loop
    args.task_iter = 0
    args.play_iter = 0

    #if args.showData:
    #    show_data(dls, envs, splits)

    # Evaluation before training
    evaluate_splits(model, dls['eval'], args, "task")

    # Training starts
    args.task_iter = 1
    args.play_iter = 1

    for i, iters in enumerate(training_schedule):
        args.max_cur_iter = iters
        dl = dls['task']['train']
        print(f'TRAINING UP TO ITERATION {iters}')
        while args.task_iter < args.max_cur_iter:
            model, args, train_metrics = train(model, dl, opt, args,'task')
            # Evaluate on all environments/splits!
            evaluate_splits(model, dls['eval'], args, "task")
        
        if 'forget' in task_args.mode and args.task_iter <= task_args.total_iterations:
            # Forget last layers before proceeding
            model = restart_model(args, model)

        if 'play' in task_args.mode and args.task_iter <= task_args.total_iterations:
            args.max_cur_iter = play_args.total_iterations
            # Replace classifier for play task
            #model = replaceModelClassifier(model, n_tasks)
            args.play_iter = 1
            dl = dls['play']['train']
            print(f"COMMENCING PLAY #{i+1}!")
            while args.play_iter < args.max_cur_iter:
                model, args, _ = train(model, dl, opt_play, args,'play')
                #evaluate_splits(model, dls['eval'], args, "play")
            # Evaluate on all environments/splits!
            # Reattach original classifier head
    
    return model

# BATCH CODE
seeds = [222
        # ,222,333
         #,444,555,666,777,888,999,123
         ]

envs = ['nobg','gradient','images']
for seed in seeds:
    for env in envs:
        for spur in [0.5, 0.625, 0.75, 0.875, 0.95, 1]: # --> 0, 0.25, 0.5, 0.75, 0.9, 1 Pearson Correlation

            start = time()
            
            #args.spurious_probability = spur
            #args.pretrained_path = f'models/scnn_{spur}_cmnist_baseline.pth'
            args.seed = seed
            args.task_args.dataset['p'] = spur
            args.eval_datasets['task'] = args.task_dataset
            args = update_args(args)
            #
            set_random_state(args)
            # reload datasets
            dls = make_dataloaders(args)
            model = main(dls)
            #
            if args.save_model:
                save_model(args, model, spur)
            end = time()
        
            print(f"Full training iteration took {end-start:.1f}s")
        