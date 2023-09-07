import torch.nn as nn 
import torch
from utils import AverageMeter, pretty_print
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from os import mkdir
import os
from torch.optim import SGD
from sklearn.metrics import confusion_matrix
from torch.nn.functional import softmax


def calculate_mean_accuracy(logits, targets): # For multiple output dimensions
    _, predicted_labels = torch.max(logits, dim=1)
    correct_predictions = (predicted_labels == targets).sum().item()
    total_predictions = targets.size(0)
    accuracy = correct_predictions / total_predictions
    return accuracy

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y,reduction="none")

def mean_accuracy(logits, y): # When using a single output dim
    preds = (logits.squeeze() > 0.).float()
    return  ((preds - y).abs() < 1e-2).float().mean()

def get_grads(model):
    grads = dict()
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.cpu()
    return grads

def get_gradients_from_data(model, x, y):
    opt = SGD(model.parameters(), lr=0.001)
    logits = model(x) # With Correlation
    loss = mean_nll(logits.squeeze(), y)
    opt.zero_grad()
    loss.backward()
    grads = get_grads(model)
    #opt.zero_grad()
    return grads

def add_grads(grad, new_grad):
    for n, g in grad.items():
        grad[n] += g
    
    return grad


def get_grouped_loss(args, losses, groups):
    bs = losses.shape[0]
    g_losses, g_counts = group_losses(losses, groups)

    if args.base_method == "gdro":
        loss = group_dro(g_losses, 1.0)
        loss /= bs
    elif args.base_method == "rw":
        loss = reweight(g_losses, g_counts)
        loss /= bs

    return loss


def group_losses(losses, groups):
    unique_values, inverse_indices = torch.unique(groups, return_inverse=True)       # Remap group values to the 0 to N_groups - 1 range
    mapping_tensor = torch.arange(len(unique_values))                                # Create a mapping tensor from unique values to indices
    groups = mapping_tensor[inverse_indices]                                         # Map the original tensor to indices using the inverse_indices
    num_classes = groups.max() + 1                                                   # Determine the number of classes or categories (assuming indices are 0-based)                                    # Create an empty tensor to store the grouped losses
    _, group_counts = groups.unique(return_counts=True)                              # Calculate the unique group values and their counts
    one_hot_matrix = torch.eye(num_classes)[groups]                                  # Create mask for losses
    grouped_losses = torch.mm(losses.unsqueeze(0).cuda(),one_hot_matrix.cuda())                    
    return grouped_losses, group_counts

def group_dro(g_losses, temp = 1.0):
    p = softmax(temp*g_losses, dim=0).cuda()
    return (p * g_losses).sum()

def reweight(g_losses, g_counts):
    p = (1/g_counts).cuda()
    return (p * g_losses).sum()

#@timing
def train(model, dl, opt, args, caption='', return_grads=False):
    mode = 'play' if 'play' in caption else 'task'
    
    group_loss = True if args.base_method in ["rw", "gdro"] else False # Do we group losses?
    
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_batches = len(dl)
    metrics = {'grads': None}#'loss': None, 'acc': None}
    model.train()
    grads_list = []
    loss_function = {True: nn.CrossEntropyLoss(reduction="none"), False: mean_nll }
    acc_function = {True: calculate_mean_accuracy, False: mean_accuracy}
    nonbinary = args.output_dims > 1
    l_f = loss_function[nonbinary]
    acc_f = acc_function[nonbinary]


    
    for n_batch, (x, y, g) in enumerate(dl):
        
        x = x.cuda()
        y = y.cuda() if nonbinary else y.float().cuda()
        g = g.cuda()
        bs = x.shape[0]
        logits = model(x)
        
        # Calculate metrics

        # TODO: 0. For ARM, you have to choose group first and sample from it.
        #       It also requires changing the architecture to support adaptation.
        #       1. Group losses by group (group comes from dataset)
        #       2. Add functions that recalculate loss given group losses:
        #           This works for: Reweighting, Group DRO and IRM.


        losses = l_f(logits.squeeze(), y)
        if group_loss:
            loss = get_grouped_loss(args, losses, g)
        else:
            loss = losses.mean()

        acc = acc_f(logits, y)
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
        if n_batch % 20 == 0:
            print(f'\r[{caption.upper():11s}] [{(n_batch+1)*bs:05d}/{total_batches*bs}] {args[f"{mode}_iter"]:03d} Loss: {loss:.3f} Acc: {100*acc:.2f}%', end="")
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
            metrics = evaluate(args, model, dl, split)
            for k, v in metrics.items():
                results[k] = v
        all_results[ds_name] = results
    #print(all_results)
    #print(results)
    pretty_print(all_results,args,stage)
    for ds_name, results in all_results.items():
        args.exp.log_metrics(results, prefix=ds_name, step=args[f'task_iter'], epoch=args[f'task_iter'])
    return all_results

#@timing
def evaluate(args, model, dl, caption='train', show=True):
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    metrics = {}
    all_predictions = []
    all_labels = []
    model.eval()
    group_loss = True if args.base_method in ["rw", "gdro"] else False # Do we group losses?
   
    loss_function = {True: nn.CrossEntropyLoss(reduction="none"), False: mean_nll }
    acc_function = {True: calculate_mean_accuracy, False: mean_accuracy}
    nonbinary = args.output_dims > 1
    l_f = loss_function[nonbinary]

    acc_f = acc_function[nonbinary]
    n_samples = len(dl.dataset)
    for n_batch, (x, y, g) in enumerate(dl):
        with torch.no_grad():
            x = x.cuda()
            y = y.cuda() if nonbinary else y.float().cuda()
            g = g.cuda()
            bs = x.shape[0]
            logits = model(x)
            
            # Calculate metrics
            #print(logits.squeeze(), y)
            losses = l_f(logits.squeeze(), y)
            if group_loss:
                loss = get_grouped_loss(args, losses, g)
            else:
                loss = losses.mean()

            acc = acc_f(logits, y)
            cur_loss = loss.detach().cpu()
            loss_meter.update(cur_loss, bs)
            acc_meter.update(acc, bs)

            # Accumulate predictions and labels
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        if show:
            print(f'\r[{caption.upper():11s}] [{(n_batch+1)*bs:05d}/{n_samples} ({100*((n_batch+1)*bs)/n_samples:.0f}%)] Loss: {loss_meter.avg:.3f} Acc: {100*acc_meter.avg:.2f}%', end="")
        #if args.output_dims > 1:
            # Calculate confusion matrix
         #   confusion_mat = confusion_matrix(all_labels, all_predictions)
         #   print(f"\nConfusion Matrix:\n{confusion_mat}")

    # Report results
    metrics[f'{caption}_loss'] = float(loss_meter.avg)
    metrics[f'{caption}_acc'] = float(100*acc_meter.avg)
       
    return metrics