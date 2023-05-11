import torch
import torch.nn as nn
from models import simplecnn
import copy
args = []


# These were all generated using ChatGPT 3.5


def generate_mask(weights, method='absolute', t=0.5, asc=True):
    """
    Generate a binary mask for each weight tensor, based on the specified pruning method.

    Args:
        weights (dict): A dictionary mapping weight tensor names to weight tensor values.
        method (str): The pruning method to use. Possible values are 'absolute', 'full', and 'partial'.
        t (float): The threshold value for selecting elements based on magnitude, used only if method is 'absolute'.
        threshold_ratio (float): A value between 0 and 1 specifying the percentile of the largest magnitude values to use as the threshold, used only if method is 'full' or 'partial'.

    Returns:
        dict: A dictionary mapping weight tensor names to binary mask tensors of the same shape.
    """
    mask = {}
    if method == 'absolute':
        mask = generate_mask_magnitude(weights, t, asc = asc)
    elif method == 'random':
        mask = generate_random_mask(weights, t)
    elif method == 'full':
        mask = generate_mask_percentile_full(weights, t, asc = asc)
    elif method == 'partial':
        mask = generate_mask_percentile_partial(weights, t, asc = asc)
    else:
        raise ValueError(f"Invalid pruning method: {method}")
    return mask

def generate_random_mask(model, p=0.5):
    """
    Generate a binary mask for each parameter in the model, where each element
    is randomly set to 0 or 1 with probability p.

    Args:
        model (nn.Module): The PyTorch model to generate the masks for.
        p (float): The probability of setting each element of the mask to 1.

    Returns:
        dict: A dictionary mapping parameter names to binary mask tensors of the same shape.
    """
    mask = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            mask[name] = torch.zeros_like(param).bernoulli_(p).bool()
    return mask

def generate_mask_magnitude(weights, t=0.5, asc=True):
    """
    Generate a binary mask for each weight tensor, where each element is set to 1 if its
    magnitude is greater than t, and 0 otherwise.

    Args:
        weights (dict): A dictionary mapping weight tensor names to weight tensor values.
        t (float): The threshold value for selecting elements based on magnitude.

    Returns:
        dict: A dictionary mapping weight tensor names to binary mask tensors of the same shape.
    """
    mask = {}
    for name, param in weights.items():
        if asc:
            mask[name] = param < t
        else:
            mask[name] = param > t
    return mask

def generate_mask_percentile_full(tensors_dict, threshold_ratio, asc = True):
    """
    Generate a binary mask for each tensor in the input dictionary, where each element is set to 1 if its magnitude is greater than a threshold value computed as a percentile of the largest magnitude values.

    Args:
        tensors_dict (dict): A dictionary mapping tensor names to tensor values.
        threshold_percentile (float): A value between 0 and 1 specifying the percentile of the largest magnitude values to use as the threshold.

    Returns:
        dict: A dictionary mapping tensor names to binary mask tensors of the same shape as the input tensors.
    """
   

    # Concatenate all tensors in the dictionary along the first dimension
    concatenated_tensor = torch.cat([tensor.view(-1) for tensor in tensors_dict.values()], dim=0)
    
    # Compute the absolute values of the concatenated tensor
    abs_tensor = torch.abs(concatenated_tensor)
    mask_tensor = torch.zeros_like(concatenated_tensor)
    # Compute the threshold value for the specified percentile
    if asc:
        threshold = torch.kthvalue(abs_tensor, int((threshold_ratio) * abs_tensor.numel()))[0]
        mask_tensor[abs_tensor < threshold] = 1
    
    else:
        threshold = torch.kthvalue(abs_tensor, int((1 - threshold_ratio) * abs_tensor.numel()))[0]
        mask_tensor[abs_tensor > threshold] = 1
    

    # Create the mask tensor based on the threshold value

    mask_tensor[abs_tensor > threshold] = 1
    
    # Split the mask tensor back into a dictionary of tensors with the same shapes as the input tensors
    mask_dict = {}
    start_idx = 0
    for name, tensor in tensors_dict.items():
        end_idx = start_idx + tensor.numel()
        mask_dict[name] = mask_tensor[start_idx:end_idx].view_as(tensor).bool()
        start_idx = end_idx
    
    return mask_dict
    
def generate_mask_percentile_partial(tensors_dict, threshold_ratio, asc = True):
    """
    Generate a dictionary of binary masks for a given dictionary of tensors, where each mask indicates which
    values in the corresponding tensor should be pruned based on a specified threshold percentile.
    
    Args:
        tensors_dict (dict): A dictionary of PyTorch tensors.
        threshold_percentile (float): The threshold percentile used to generate the masks, expressed as a float 
            between 0 and 1.
            
    Returns:
        dict: A dictionary of binary masks, where each mask has the same shape as the corresponding tensor 
        in tensors_dict.
    """
    # Create an empty dictionary to hold the mask tensors
    mask_dict = {}

    # Loop over each tensor in the dictionary
    for name, tensor in tensors_dict.items():
        # Compute the absolute values of the tensor
        abs_tensor = torch.abs(tensor)
        mask_tensor = torch.zeros_like(tensor)
        # Compute the threshold value for the specified percentile
        if asc:
            k = min(int((threshold_ratio) * abs_tensor.numel()),1)
            if abs_tensor.view(-1).shape[0]>1 and k > 0:
                threshold = torch.kthvalue(abs_tensor.view(-1), k)[0]
                mask_tensor[abs_tensor < threshold] = 1
        else:
            k = min(int((threshold_ratio) * abs_tensor.numel()),1)
            if abs_tensor.view(-1).shape[0]>1 and k > 0 :
                threshold = torch.kthvalue(abs_tensor.view(-1), k)[0]
                mask_tensor[abs_tensor > threshold] = 1

        # Add the mask tensor to the dictionary of mask tensors
        mask_dict[name] = mask_tensor.bool()

    return mask_dict


# def forget_model(model, mask):
#     """
#     Reinitialize the masked weights in a PyTorch model with random values drawn from a normal distribution.
    
#     Args:
#         model (torch.nn.Module): The PyTorch model to be pruned.
#         mask (dict): A dictionary of binary masks indicating which values in each parameter tensor should be 
#             pruned.
            
#     Returns:
#         torch.nn.Module: The pruned PyTorch model with reinitialized weights.
#     """
#     for name, param in model.named_parameters():
#         if name in mask:
#             param.data[mask[name]] = torch.randn_like(param)[mask[name]]
#     return model

def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)

    return model

def forget_model(model, mask):
    """
    Reinitialize the masked weights in a PyTorch model with random values drawn from a normal distribution.
    
    Args:
        model (torch.nn.Module): The PyTorch model to be pruned.
        mask (dict): A dictionary of binary masks indicating which values in each parameter tensor should be 
            pruned.
            
    Returns:
        torch.nn.Module: The pruned PyTorch model with reinitialized weights.
    """

    reinit_model =  copy.deepcopy(model)
    reinit_model = init_weights(reinit_model)

    for (name, param), (_, r_param) in zip(model.named_parameters(), reinit_model.named_parameters()):
        if name in mask:
            param.data[mask[name]] = 0 
            r_param.data[~mask[name]] = 0
            param.data += r_param.data
    return model
