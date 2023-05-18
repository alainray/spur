import torch.nn as nn 
import torch
from utils import AverageMeter, pretty_print
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from os import mkdir
import os
from torch.optim import SGD

def get_grads(model):
    grads = dict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.cpu()
    return grads

def get_gradients_from_data(model, x, y):
    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)
    opt = SGD(model.parameters(), lr=0.001)

    logits = model(x) # With Correlation
    loss = mean_nll(logits.squeeze(), y)
    opt.zero_grad()
    loss.backward()
    grads = get_grads(model)
    opt.zero_grad()
    return grads

def add_grads(grad, new_grad):
    for n, g in grad.items():
        grad[n] += g
    
    return grad

#@timing
def train(model, dl, opt, args, caption='', return_grads=False):
    mode = 'play' if 'play' in caption else 'task'
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_batches = len(dl)
    metrics = {'grads': None}#'loss': None, 'acc': None}
    model.train()
    grads_list = []
    
    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits.squeeze() > 0.).float()
        return  ((preds - y).abs() < 1e-2).float().mean()
    
    for n_batch, (x, y) in enumerate(dl):
        
        x = x.cuda()
        y = y.cuda()
        bs = x.shape[0]
        logits = model(x)
        
        # Calculate metrics
        loss = mean_nll(logits.squeeze(), y.float())
        acc = mean_accuracy(logits, y)
        cur_loss = loss.detach().cpu()
        loss_meter.update(cur_loss, bs)
        acc_meter.update(acc, bs) 

        metrics[f'{caption}_loss'] = float(cur_loss)
        metrics[f'{caption}_acc'] = float(100*acc_meter.avg)
        metrics[f'{mode}_iter'] = args[f'{mode}_iter']
        # Backpropagation Update
        opt.zero_grad()
        loss.backward()
        if return_grads: # Store gradients       
            new_grads = get_grads(model)
            grads_list.append((args[f'{mode}_iter'], new_grads ))
           # grads = add_grads(grads, new_grads) if grads is not None else new_grads
        opt.step()
        # Report results
        # print(f'[{caption.upper():11s}] [{(n_batch+1)*bs:05d}/{total_batches*bs}] {args.cur_iter:03d} Loss: {loss:.3f} Acc: {100*acc:.2f}%')
        #args.exp.log_metrics(metrics, prefix=caption, step=args[f'{mode}_iter'], epoch=args[f'{mode}_iter'])
        # Exit training if we know there's an intervention coming!
        metrics['grads'] = grads_list
      
        if args.max_cur_iter == args[f'{mode}_iter']:
           
            args[f'{mode}_iter'] +=1
            return model, args, metrics
        args[f'{mode}_iter'] +=1

    return model, args, metrics

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
            metrics = evaluate(model, dl, split)
            for k, v in metrics.items():
                results[k] = v
        all_results[ds_name] = results

    #print(results)
    pretty_print(all_results,args,stage)
    for ds_name, results in all_results.items():
        args.exp.log_metrics(results, prefix=ds_name, step=args[f'task_iter'], epoch=args[f'task_iter'])
    return results

#@timing
def evaluate(model, dl, caption='train'):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    metrics = {}
    
    model.eval()
    
    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits.squeeze() > 0.).float()
        return  ((preds - y).abs() < 1e-2).float().mean()

    for n_batch, (x, y) in enumerate(dl):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda()
            bs = x.shape[0]
            logits = model(x)
            
            # Calculate metrics
            loss = mean_nll(logits.squeeze(), y.float())
            acc = mean_accuracy(logits, y)
            cur_loss = loss.detach().cpu()
            loss_meter.update(cur_loss, bs)
            acc_meter.update(acc, bs) 

    # Report results
    metrics[f'{caption}_loss'] = float(loss_meter.avg)
    metrics[f'{caption}_acc'] = float(100*acc_meter.avg)
       
    return metrics