import comet_ml
import torch
import argparse
from dataset import make_dataloaders, get_spurious_samples
from params import *                        # Experiment parameters 
from models import create_model, restart_model, load_model
from torch.optim import SGD, Adam
from train import train, evaluate_splits, get_reps
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

    if args.after_training:
        training_schedule.append(training_schedule[-1]+args.task_total_iterations)
    print(f'Current training schedule is: {training_schedule}. {len(training_schedule)-1} intervention(s) total.')

    # Create model/optimizers
 
    model=create_model(args).cuda()
    if args.load_pretrained:
        model = load_model(model, args.pretrained_path).cuda()

    if args.frozen_features:
        model = freeze_model(args, model).cuda()
        model = restart_model(args, model)

    opt = Adam(model.parameters(), lr=0.001)#),momentum=0.9,weight_decay=0.01)
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
    # Initialize stored metrics
    all_metrics = {'task_env1': dict(), 'eval': dict()}

    for k in all_metrics.keys():
        for split in ["train", "val","test"]:
            for m in args.metrics:
                all_metrics[k][f"{split}_{m}"] = []
    

    # Evaluation before training
    print("Evaluating untrained model!")
    metrics = evaluate_splits(model, dls['eval'], args, "task")
    for ds_name, m in metrics.items():
        all_metrics[ds_name] = update_metrics(all_metrics[ds_name], m)
    # Training starts

    if "svdrop" in args.base_method:
        all_metrics['task_env1']['singular'] = []
        # get representations R
        # calculate representations
        dl = dls['task']['env1']['train']
        R = get_reps(model, dl)
        # apply to model
        model.fc.set_n_dirs(1000)
        S,_ = model.fc.set_singular(R)
        # dropout first dimension
        #model.fc.dropout_dim([0]) # dropout first singular vector!
        all_metrics['task_env1']['singular'].append(S)
        args.base_method = f"svdrop_{args.svdropout_p}"
    elif "gdro" in args.base_method:
        args.base_method += f"_{args.gdro_adj}"
    args.task_iter = 1
    args.play_iter = 1
    grads = []
    n_session = 0
    last_session_iterations = 0
    min_val_acc= 0.0
    best_model = None
    metric_for_best = "worst_group"

    for i, iters in enumerate(training_schedule):
        args.max_cur_iter = iters

        n_session +=1
        print(f'TRAINING UP TO ITERATION {iters} - Training Session {n_session}')
        while args.task_iter <= args.max_cur_iter:

            dl = dls['task']['env1']['train']
            model, args, train_metrics = train(model, dl, opt, args,'task',args.save_grads)


            # Evaluate on all environments/splits!
            #if args.save_model:
            #    save_model(args, model)
            #grads += train_metrics['grads']
            metrics = evaluate_splits(model, dls['eval'], args, "task")
            for ds_name, m in metrics.items():
                all_metrics[ds_name] = update_metrics(all_metrics[ds_name], m)
            
            current_loss = metrics['eval'][f'val_{metric_for_best}_loss']
            current_acc = metrics['eval'][f'val_{metric_for_best}_acc']
           
            if args.save_best:

                if current_acc > min_val_acc:
                    min_val_acc = current_acc
                    print(f"New best model! Best val {metric_for_best} acc is now: {100*min_val_acc:.2f}% and loss: {current_loss:.3f}")
                    print("",flush=True,end="")
                    best_model = {'iter': args.task_iter-1, 
                                  'loss': current_loss,
                                  'acc':  min_val_acc,
                                  #"args": args,
                                  'model': model.state_dict()}
            if args.base_method == "svdrop":
                # get representations R
                # calculate representations
                R = get_reps(model, dl)
                # apply to model
                model.fc.set_n_dirs(1000)
                S,_ = model.fc.set_singular(R)
                # dropout first dimension
                model.fc.dropout_dim([0], p =args.svdropout_p) # dropout first singular vector!

                all_metrics['task_env1']['singular'].append(S)
            # best_model = model.clone()

        if 'forget' in args.mode and args.task_iter <= training_schedule[-1]:
            # Forget last layers before proceeding
            # Get mask
            print(f"Currently Forgetting using method: {args.forget_method.upper()} - %: {100*args.forget_threshold} - Criteria: {args.forget_criteria.upper()}_{args.forget_asc}")
            print(f"Iters delta: {iters-last_session_iterations}")

            avg_grads, std_grads = average_grads(grads[-(iters-last_session_iterations):])
            last_session_iterations = iters
            if args.forget_method == 'random':
                input_for_mask = model
            elif args.forget_method == 'spur_grads':
                # Get gradients for spurious
                print("Getting Spurious and Non Spurious gradients!")
                spur_data, non_spur_data, y = get_spurious_samples()
                spur_grads_red = get_gradients_from_data(model, spur_data[:30000], y[:30000])
                non_spur_grads_red = get_gradients_from_data(model, non_spur_data[:30000], y[:30000])
                spur_grads_green = get_gradients_from_data(model, spur_data[30000:], y[30000:])
                non_spur_grads_green = get_gradients_from_data(model, non_spur_data[30000:], y[30000:])
                input_for_mask = [spur_grads_red, non_spur_grads_red,spur_grads_green, non_spur_grads_green]
            
            elif args.forget_criteria == 'gradients':
                input_for_mask = avg_grads
            elif args.forget_criteria == 'stds':
                input_for_mask = std_grads

            forget_mask = generate_mask(input_for_mask, method=args.forget_method, t=args.forget_threshold, asc=args.forget_asc)
            print()
            model = forget_model(model, forget_mask)
            #model = restart_model(args, model)

        if 'play' in args.mode and args.task_iter <= training_schedule[-1]:
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

    if args.save_stats:
        save_stats(args, all_metrics)
    return model, best_model



if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--seed', type=int, help='Seed value', required=True)
    parser.add_argument('--forget_method', type=str, help='Forget Method', required=False, default="xxx")
    parser.add_argument('--forget_criteria', type=str, help='Forget Criteria', required=False, default="xxx")
    parser.add_argument('--base_method', type=str, help='Baseline Method', required=False, default="")
    parser.add_argument('--n_interventions', type=int, help='Number of interventions', required=False, default=1)
    parser.add_argument('--spur', type=float, help='Spurious Probability', required=False, default=0.5)
    parser.add_argument('--forget_asc', type=int, help='Value of forget_asc', required=False, default=0)
    parser.add_argument('--forget_threshold', type=float, help='Value of forget_t', required=False,default=0)
    parser.add_argument('--env', type=str, help='Environment variable', required=False, default='nobg')
    parser.add_argument('--svdropout_p', type=float, help='Probability of SVDrop', required=False, default=1.0)
    parser.add_argument('--gdro_adj', type=float, help='Generalization adjustment for gdro', required=False, default=0.0)
    parser.add_argument('--total_iterations', type=int, help='Number of interventions', required=False, default=2000)

    # Parse command-line arguments
    input_args = parser.parse_args()

    # Access parsed arguments
    for arg in vars(input_args):
        args[arg] = getattr(input_args, arg)
        
    args.forget_asc = bool(args.forget_asc)
    task_args.n_interventions = args.n_interventions
    task_args.total_iterations = args.total_iterations
    env = input_args.env
    spur = input_args.spur

    start = time()

    if args.load_pretrained:
        args.pretrained_path = f'models/scnn_synmnist_{spur}_{env}_{args.pretrained_model_type}_cmnist_baseline.pth'

    if not args.frozen_features:
        args.task_datasets['env1']['p'] = spur
        args.task_datasets['env1']['corr']  = round(2*spur -1,2) 

        args.eval_datasets['eval'].corr = 0.0#  round(2*spur -1,2) 
    #args.task_args.dataset['bg'] = env

    #args.eval_datasets['task'] = task_args.dataset


    args = update_args(args)
    #
    set_random_state(args)
    # reload datasets
    dls = make_dataloaders(args)
    print("======================================")
    print(f"Training on: {args.task_datasets['env1']}")
    print(f"Evaluating on: {args.eval_datasets['eval']}")
    print(f"Training method: {args.base_method.upper()}")
    print("======================================")
    model, best_model = main(dls)
    
    if args.save_best:
        save_model(args, best_model)
    if args.save_model:
        save_model(args, {'iter': args.task_iter-1, 
                          'loss': "N/A",
                          'acc':  "N/A",
                          'model': model.state_dict()})
    end = time()

    print(f"Full training iteration took {end-start:.1f}s")
        