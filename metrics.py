import torch
from utils import AverageMeter
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
'''
 group data(...) :

 Function for evaluating performance on a set of evaluation datasets

 Args:
- data: arbitrary numerical data at the sample level. Could be individual losses or predictions
- groups: a group identifier for each element in 'data'.

'''
def group_data(data, groups=None):
    if groups is None:                                                              # If no groups, all data in the same group
        groups = torch.zeros_like(data)
    unique_values, inverse_indices = torch.unique(groups, return_inverse=True)       # Remap group values to the 0 to N_groups - 1 range
    mapping_tensor = torch.arange(len(unique_values))                                # Create a mapping tensor from unique values to indices
    groups = mapping_tensor[inverse_indices.cpu()]                                   # Map the original tensor to indices using the inverse_indices
    num_classes = groups.max() + 1                                                   # Determine the number of classes or categories (assuming indices are 0-based)                                    # Create an empty tensor to store the grouped losses
    _, group_counts = groups.unique(return_counts=True)                              # Calculate the unique group values and their counts
    one_hot_matrix = torch.eye(num_classes)[groups]                                  # Create mask for losses
    if data.device != one_hot_matrix.device:
        one_hot_matrix = one_hot_matrix.cuda()
    grouped_data = torch.mm(data.unsqueeze(0),one_hot_matrix)
    return grouped_data, group_counts

'''
accuracy(...) :

 Function for evaluating accuracy per group

 Args:
- data: dictionary that contains dataset (inputs, labels, groups)
- metrics: dictionary with key 'logits' with predicted logits for 'labels'
- args: dictionary with all experiment arguments.

'''
def accuracy(data, metrics, args, mode="total"):

    groups = None if mode == "total" else data['groups']
    if args.output_dims > 1:
        group_acc, group_counts = mean_accuracy_N(metrics['logits'], data['labels'], groups)
    else:
        group_acc, group_counts = mean_accuracy_1(metrics['logits'], data['labels'], groups)
    return group_acc/group_counts

'''
 mean accuracy_N(...) :

 Function for evaluating accuracy per group when output dimensions of model > 1

 Args:
- logits: a tensor with outputs from a model
- targets: a tensor with ground truth labels for data from 'logits'
- groups: a tensor with group labels for data in 'logits'

'''       
def mean_accuracy_N(logits, targets, groups): # For multiple output dimensions
    _, predicted_labels = torch.max(logits, dim=1)
    correct_predictions = (predicted_labels == targets).float()
    r = group_data(correct_predictions, groups)
    return r
'''
 mean accuracy_1(...) :

 Function for evaluating accuracy per group when output dimensions of model = 1 (binary classification)

 Args:
- logits: a tensor with outputs from a model
- targets: a tensor with ground truth labels for data from 'logits'
- groups: a tensor with group labels for data in 'logits'

'''    
def mean_accuracy_1(logits, targets, groups): # When using a single output dim
    preds = (logits.squeeze() > 0.).float()
    correct = ((preds - targets).abs() < 1e-2).float()
    return group_data(correct, groups)#.mean(dim=1)

# Wrappers for accuracy
def group_accuracy(data, metrics, args):
    return accuracy(data, metrics, args, mode="group")

def mean_accuracy(data, metrics, args):
    return accuracy(data, metrics, args, mode="total")

def worst_group_accuracy(data, metrics, args):
    accs = group_accuracy(data,metrics,args)
    return accs.min().item()

def best_group_accuracy(data, metrics, args):
    accs = group_accuracy(data,metrics,args)
    return accs.max().item()

def loss(data, metrics, args, mode="total"):

    groups = None if mode == "total" else data['groups']
    if args.output_dims > 1:
        group_loss, group_counts = mean_loss_N(metrics['logits'], data['labels'], groups)
    else:
        group_loss, group_counts = mean_loss_1(metrics['logits'], data['labels'], groups)
    
    return group_loss/group_counts

def get_loss_fn(args):
    return CrossEntropyLoss(reduction="none") if args.output_dims > 1 else BCEWithLogitsLoss(reduction="none")

def mean_loss_1(logits, targets, groups):
    losses = binary_cross_entropy_with_logits(logits.squeeze(), targets.float(), reduction="none")
    return group_data(losses, groups)

def mean_loss_N(logits, targets, groups):
    losses = cross_entropy(logits, targets, reduction="none")
    return group_data(losses, groups)

# Wrappers for loss
def group_loss(data, metrics, args):
    return loss(data, metrics, args, mode="group")

def mean_loss(data, metrics, args):
    return loss(data, metrics, args, mode="total")
 
def worst_group_loss(data, metrics, args):
    losses = group_loss(data, metrics, args)
    return losses.max().item()

def best_group_loss(data, metrics, args):
    losses = group_loss(data, metrics, args)
    return losses.min().item()

def calculate_metrics(data, metrics, args):
    m = {}
    metric_functions = {'acc': mean_accuracy,
                        'worst_group_acc': worst_group_accuracy,
                        'best_group_acc': best_group_accuracy,
                        'loss': mean_loss,
                        'worst_group_loss': worst_group_loss,
                        'best_group_loss': best_group_loss
                        }
    metrics_to_calculate = args.metrics # acc, worst_group_acc, loss, worst_group_loss
    for metric in metrics_to_calculate:
        m[metric]= metric_functions[metric](data, metrics, args)
    return m

def update_metrics(old, new):

    return old

def create_metric_meters(args):
    return {m: AverageMeter() for m in args.metrics}

if __name__ == '__main__':
    import torch
    print(torch.cuda.is_available())