import torch.nn as nn 
import torch
from utils import timing, AverageMeter, pretty_print


#@timing
def train(model, dl, opt, args, caption=''):
    
    mode = 'play' if 'play' in caption else 'train'
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_batches = len(dl)
    metrics = {}#'loss': None, 'acc': None}
    model.train()
    
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
        opt.step()
        # Report results
        # print(f'[{caption.upper():11s}] [{(n_batch+1)*bs:05d}/{total_batches*bs}] {args.cur_iter:03d} Loss: {loss:.3f} Acc: {100*acc:.2f}%')
        #args.exp.log_metrics(metrics, prefix=caption, step=args[f'{mode}_iter'], epoch=args[f'{mode}_iter'])
        # Exit training if we know there's an intervention coming!
        if args.max_cur_iter == args[f'{mode}_iter']:
            args[f'{mode}_iter'] +=1
            return model, args, metrics
        args[f'{mode}_iter'] +=1
    return model, args , metrics

#@timing
def evaluate_splits(model, dls, envs, splits, args, mode):
    results = dict()

    for env in envs:
        for split in splits:
            dl = dls[env][split]
            _, _, metrics = evaluate(model, dl, args, env + '-' + split)
            results[env + '-' + split] = metrics
    
    pretty_print(results, args, mode)

    return results

#@timing
def evaluate(model, dl, args, caption='train_test'):
    mode = 'play' if 'play' in caption else 'train'
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    total_batches = len(dl)
    metrics = {}#'loss': None, 'acc': None}
    
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

            metrics[f'{caption}_loss'] = float(cur_loss)
            metrics[f'{caption}_acc'] = float(100*acc_meter.avg)
            metrics[f'{mode}_iter'] = args[f'{mode}_iter']
        # Report results
        #print(f'[{caption.upper():11s}] [{(n_batch+1)*bs:05d}/{total_batches*bs}] {cur_iter:03d} Loss: {loss:.3f} Acc: {100*acc:.2f}%')
        args.exp.log_metrics(metrics, prefix=mode, step=args[f'train_iter'], epoch=args[f'play_iter'])
       
    return model, args, metrics