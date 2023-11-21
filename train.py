import torch.nn as nn 
import torch
from utils import pretty_print
#from torch.nn.utils import vector_to_parameters, parameters_to_vector
#from os import mkdir
from metrics import *
import os

#from sklearn.metrics import confusion_matrix
from torch.nn.functional import softmax
#from torch import autograd

# Functions for using group based methods
def get_grouped_loss(args, losses, groups):
    bs = losses.shape[0]
    g_losses, g_counts = group_data(losses, groups)

    if args.base_method == "gdro":
        loss = group_dro(g_losses, 0.4) # change for something like base_method_params[base_method]
        loss /= bs
    elif args.base_method == "rw":
        loss = reweight(g_losses, g_counts)
        loss /= bs
    else:
        loss = g_losses/bs
    return loss


def group_dro(g_losses, temp = 1.0):
    p = softmax(*g_losses, dim=0).cuda()
    return (p * g_losses).sum()

def reweight(g_losses, g_counts):
    p = (1/g_counts).cuda()
    return (p * g_losses).sum()

def run_train_iteration(data, model, opt, args):

    # Send data to GPU
    x = data['input'].cuda()
    y = data['labels'].cuda() if args.output_dims > 1 else data['labels'].float().cuda()
    g = data['groups'].cuda()

    # Output metrics definition
    metrics = {'group_loss': None, 
               'logits': None
               }

    logits = model(x)
    
    # Calculate loss
    l_f = get_loss_fn(args)
    losses = l_f(logits.squeeze(), y)
    loss = get_grouped_loss(args, losses, g) # Apply any changes to the loss based on method
    mean_loss = loss.mean()
    # Backpropagation Update
    opt.zero_grad()
    mean_loss.backward()
    opt.step()
    
    metrics['logits'] = logits.clone().cpu()
    metrics['group_loss'] = loss.clone().cpu()
    return model, metrics # logits/loss per group

def run_eval_iteration(data, model, args):

    # Send data to GPU
    x = data['input'].cuda()
    y = data['labels'].cuda() if args.output_dims > 1 else data['labels'].float().cuda()
    g = data['groups'].cuda()

    # Output metrics definition
    metrics = dict()
    with torch.no_grad():
        logits = model(x)
    metrics['logits'] = logits.clone().cpu()
    return metrics # logits

# Metrics
    
def train(model, dl, opt, args, caption='', return_grads=False):
    mode = 'play' if 'play' in caption else 'task'

    #metrics = {'grads': None}#'loss': None, 'acc': None}
    metrics = create_metric_meters(args)
    model.train()

    total_batches = len(dl)

    for n_batch, (x, y, g) in enumerate(dl):
        bs = x.shape[0]
        data = {'input': x, 'labels': y, 'groups': g}
        model, result = run_train_iteration(data, model, opt, args)
        # Update on progress
        print(f"\r{n_batch+1}/{total_batches} ({100*(n_batch+1)/(total_batches):.2f}%)", end="")    
        # Do we need to stop prematurely?
        if args.max_cur_iter == args[f'{mode}_iter']:
            args[f'{mode}_iter'] +=1
            return model, args, metrics
        args[f'{mode}_iter'] +=1
    print("",flush=True)
    return model, args, metrics 
        # Calculate metrics


#@timing
'''
 evaluate splits(...) :

 Function for evaluating performance on a set of evaluation datasets

 Args:
- model: PyTorch model where you want to evaluate.
- dls: a dict() where every key is a split and every value is a dataloader.
- args: dictionary with all experiment arguments.
- stage: ['task', 'play]: whether we are evaluating after training the task or playing.
'''

def evaluate_splits(model, dls, args, stage):
    
    all_results = dict()
    for ds_name, ds in dls.items():
        results = dict()

        for split, dl in ds.items(): 
            metrics = evaluate(args, model, dl, split)
            for k, v in metrics.items():
                results[k] = v
        all_results[ds_name] = results

    pretty_print(all_results,args,stage)
    for ds_name, results in all_results.items():
        args.exp.log_metrics(results, prefix=ds_name, step=args[f'task_iter'], epoch=args[f'task_iter'])
    return all_results

#@timing
def evaluate(args, model, dl, caption='train', show=True):
    metrics = create_metric_meters(args)
    model.train()

    total_batches = len(dl)
    evaluate_result = {'logits': []}
    evaluate_data= {'labels': [], 'groups': []}

    for n_batch, (x, y, g) in enumerate(dl):
        data = {'input': x, 'labels': y, 'groups': g}
        result = run_eval_iteration(data, model, args)
        evaluate_result['logits'] += result['logits']
        evaluate_data['labels'] += y.clone().cpu() 
        evaluate_data['groups'] += g.clone().cpu() 
    
    evaluate_result['logits'] =  torch.stack(evaluate_result['logits'])
    evaluate_data['labels'] = torch.stack(evaluate_data['labels'])
    evaluate_data['groups'] =torch.stack(evaluate_data['groups'])
    #print(evaluate_result['logits'].shape)
    #print(evaluate_data['labels'].shape)
    #print(evaluate_data['groups'].shape)
    metrics = calculate_metrics(evaluate_data, evaluate_result, args)

    return {f'{caption}_{k}': float(v) for k,v in metrics.items()} 