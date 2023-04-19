import torch
import numpy as np
from torchvision.datasets import MNIST
from typing import Any, Callable, Dict, List, Optional, Tuple
from torch.utils.data import DataLoader
from params import args, task_args, play_args
from copy import deepcopy

n_classes = {'cmnist': 2, 'cfmnist': 2}

batch_sizes = {
    'mlp': {'train': 10000, 'test': 50000},
    'scnn': {'train': 10000, 'test': 50000},
    'resnet18': {'train': 512, 'test': 4096}
}

class TMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    
class TFashionMNIST(TMNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.
        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Divide into train and test sets
# Define colors to use: [red, blue, green]
# For training sets:
# Assign Colors with probability p_train to each set.
# For training sets:
# Assign Opposite Colors with probability p_test to each set. (Recommended use p_test = 1 - p_train)

mnist = TMNIST("../datasets", train=True, download=True)
fmnist = TFashionMNIST("../datasets", train=True, download=True)
mnist_train = (mnist.data[:50000], mnist.targets[:50000])
mnist_val = (mnist.data[50000:], mnist.targets[50000:])
fmnist_train = (fmnist.data[:50000], fmnist.targets[:50000])
fmnist_val = (fmnist.data[50000:], fmnist.targets[50000:])


def make_dataloaders(args):
    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(fmnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(fmnist_train[1].numpy())
    # Build environments

    def make_environment(images, labels, e, baseline=False, duplicate=False):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        
        # Assign a binary label based on the digit
        
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.0, len(labels)))
        images = torch.stack([images, images], dim=1)
        if not baseline:
            # Assign a color based on the label; flip the color with probability e
            colors = torch_xor(labels, torch_bernoulli(1-e, len(labels)))
            # Apply the color to the image by zeroing out the other color channel
            imgs2 = deepcopy(images)
            images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0 # copy 1

            if duplicate:
                imgs2[torch.tensor(range(len(images))), (colors).long(), :, :] *= 0 # copy 2
                n_els, *_ = imgs2.shape
                cutoff_index = int(n_els*args.aid_perc)
                images = torch.cat((images, imgs2[:cutoff_index]))
                labels = torch.cat((labels, labels[:cutoff_index]))
            print(images.shape, labels.shape)
        return {
        'images': (images.float() / 255.),
        'labels': labels[:, None]
        }

    p = args.spurious_probability

    if task_args.dataset == 'cmnist':
        train_train = mnist_train
        train_test = mnist_val
    else:
        train_train = fmnist_train
        train_test = fmnist_val

    if play_args.dataset == 'cmnist':
        play_train = mnist_train
        play_test= mnist_val
    else:
        play_train = fmnist_train
        play_test = fmnist_val

    train_env = make_environment(train_train[0], train_train[1], p, baseline=args.train_baseline, duplicate=False)
    test_env = make_environment(train_test[0], train_test[1], 1-p, baseline=args.train_baseline, duplicate=False)

    play_train_env = make_environment(play_train[0], play_train[1], 1-p, baseline=args.play_baseline)
    play_test_env = make_environment(play_test[0], play_test[1], 1-p, baseline=args.play_baseline)

    train_ds = TMNIST("../datasets")
    test_ds = TMNIST("../datasets")
    play_train_ds = TFashionMNIST("../datasets")
    play_test_ds = TFashionMNIST("../datasets")
    train_ds.data = train_env['images']
    train_ds.targets = train_env['labels']
    test_ds.data = test_env['images']
    test_ds.targets = test_env['labels']
    play_train_ds.data = play_train_env['images']
    play_train_ds.targets = play_train_env['labels']
    play_test_ds.data = play_test_env['images']
    play_test_ds.targets = play_test_env['labels']

    dls = {
        'train': {'train': train_ds, 'test': test_ds} , 
        'play': {'train': play_train_ds, 'test': play_test_ds}
    }

    for k, v in dls.items():
        for k1,  v1 in v.items():
            dls[k][k1] = DataLoader(v1,batch_size=batch_sizes[args.model][k1],shuffle=True)
    return dls