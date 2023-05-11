import comet_ml
import torch
import argparse
from dataset import make_dataloaders
from params import *                        # Experiment parameters 
from models import create_model, restart_model
from torch.optim import SGD, Adam
from train import train, evaluate_splits
from utils import *
from time import time
from masks import generate_mask, forget_model
from functools import reduce

def parameters_to_vec(params): # input params is dictionary
    result = []
    layer = []
    for k, v in params.items():
        #
        result.append(v.view(-1).abs())
        n_params = reduce(lambda x, y: x * y,v.shape)
        layer.extend(n_params*[k])
    return torch.cat(result), layer

def average_grads(gradients):
    _, grads = zip(*gradients)
    example_dict = grads[0]
    vec_grads = [parameters_to_vec(g)[0] for g in grads]
    vec_grads = torch.stack(vec_grads)
    avg_grads = vec_grads.mean(dim=0)
    std_grads = vec_grads.std(dim=0)
    start_idx = 0
    avg_grad_dict =  {}
    std_grad_dict = {}
    for name, tensor in example_dict.items():
        end_idx = start_idx + tensor.numel()
        avg_grad_dict[name] = avg_grads[start_idx:end_idx].view_as(tensor)
        std_grad_dict[name] = std_grads[start_idx:end_idx].view_as(tensor)
        start_idx = end_idx
    
    return avg_grad_dict, std_grad_dict

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
        model = restart_model(args, model)

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

    # Evaluation before traicd sto  ning
    evaluate_splits(model, dls['eval'], args, "task")

    # Training starts
    args.task_iter = 1
    args.play_iter = 1
    grads = []
    n_session = 0
    for i, iters in enumerate(training_schedule):
        args.max_cur_iter = iters
        dl = dls['task']['train']
        n_session +=1
        print(f'TRAINING UP TO ITERATION {iters} - Training Session {n_session}')
        while args.task_iter < args.max_cur_iter:
            model, args, train_metrics = train(model, dl, opt, args,'task',args.save_grads)
            # Evaluate on all environments/splits!
            grads += train_metrics['grads']
            evaluate_splits(model, dls['eval'], args, "task")
        
        if 'forget' in task_args.mode and args.task_iter <= task_args.total_iterations:
            # Forget last layers before proceeding
            # Get mask
            print(f"Currently Forgetting using method: {args.forget_method.upper()} - %: {100*args.forget_threshold} - Criteria: {args.forget_criteria.upper()}_{args.forget_asc}")
            avg_grads, std_grads = average_grads(grads)
            if args.forget_method == 'random':
                input_for_mask = model
            elif args.forget_criteria == 'gradients':
                input_for_mask = avg_grads
            elif args.forget_criteria == 'stds':
                input_for_mask = std_grads
            
            forget_mask = generate_mask(input_for_mask, method=args.forget_method, t=args.forget_threshold, asc=args.forget_asc)
            model = forget_model(model, forget_mask)
            #model = restart_model(args, model)

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
    
    if args.save_grads:
        save_grads(args, grads)
    return model



if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--seed', type=int, help='Seed value', required=True)
    parser.add_argument('--method', type=str, help='Forget Method', required=True)
    parser.add_argument('--forget_crit', type=str, help='Forget Criteria', required=True)
    parser.add_argument('--spur', type=float, help='Spurious Probability', required=True)
    parser.add_argument('--forget_asc', type=int, help='Value of forget_asc', required=True)
    parser.add_argument('--forget_t', type=float, help='Value of forget_t', required=True)
    parser.add_argument('--env', type=str, help='Environment variable', required=True)

    # Parse command-line arguments
    input_args = parser.parse_args()

    # Access parsed arguments
    args.forget_method = input_args.method
    seed = input_args.seed
    args.forget_criteria = input_args.forget_crit
    args.forget_asc = bool(input_args.forget_asc)
    args.forget_threshold = input_args.forget_t
    env = input_args.env
    spur = input_args.spur

    start = time()

    if args.load_pretrained:
        args.pretrained_path = f'models/scnn_synmnist_{spur}_{env}_{args.pretrained_model_type}_cmnist_baseline.pth'
    args.seed = seed

    if not args.frozen_features:
        args.task_args.dataset['p'] = spur

    args.task_args.dataset['bg'] = env

    args.eval_datasets['task'] = task_args.dataset


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
        