
from params import args
from easydict import EasyDict  as edict
import torch
import torch.nn as nn

def create_model(args):
    archs = edict()
    archs.resnet18 = resnet18
    archs.scnn = simplecnn
    archs.mlp = MLP
    return archs[args.model](args)

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
      
        self.fc = nn.Linear(512*block.expansion, 1)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def resnet18(num_classes=1, **kwargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        lin1 = nn.Linear(2 * 28 * 28, args.hidden_dim)
        lin2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        lin3 = nn.Linear(args.hidden_dim, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
    
    def forward(self, input):
        out = input.view(input.shape[0], 2 * 28 * 28)
        out = self._main(out)
        return out

class SimpleCNN(nn.Module):
    """
    Convolutional Neural Network
    """
    def __init__(self, num_channels, N, num_classes=1, add_pooling=False):
        super(SimpleCNN, self).__init__()

        if add_pooling:
            stride=1
        else:
            stride=2

        layer = nn.Sequential()
        layer.add_module('conv1',nn.Conv2d(2, num_channels[0]*N, kernel_size=3, stride=stride))
        layer.add_module('relu1',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool1',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv2',nn.Conv2d(num_channels[0]*N, num_channels[1]*N, kernel_size=3, stride=stride))
        layer.add_module('relu2',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool2',nn.MaxPool2d(kernel_size=2, stride=2))
        layer.add_module('conv3',nn.Conv2d(num_channels[1]*N, num_channels[2]*N, kernel_size=3, stride=stride))
        layer.add_module('relu3',nn.ReLU(inplace=True))
        if add_pooling:
            layer.add_module('pool3',nn.MaxPool2d(kernel_size=2, stride=1))
        layer.add_module('flatten', nn.Flatten())
        self.features = layer

        self.fc = nn.Sequential(nn.Linear(512*N, num_classes))
        '''for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)'''

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        return x


def simplecnn(args,**kwargs):
    return SimpleCNN([32,64,128],1,add_pooling=False)


def restart_model(args, model):
    layers = []
    
    def make_resnet_layers(resnet):
        l = []
        layer_structure = {"resnet18": [2,2,2,2],
                            "resnet34": [3, 4, 6, 3],
                            "resnet50": [3, 4, 6, 3],
                            "resnet101": [3, 4, 23, 3]
                            }
        l.append("conv1.0")

        for j,n_blocks in enumerate(layer_structure[resnet]):
            for i in range(n_blocks):
                l.append(f"conv{j+2}_x.{i}")
        
        l.append("fc")
        return l
    def initialize(m):
        if type(m) in [nn.Linear, nn.Conv2d]:
            nn.init.kaiming_normal_(m.weight)
            #if m.bias is not None:
            #    nn.init.kaiming_normal_(m.bias)

    if args.arch == "simplecnn":
        layers = ["features.conv1", "features.conv2", "features.conv3","fc"]
    elif "resnet" in args.arch:
        layers = make_resnet_layers(args.arch)
    else:
        print("Model Not Supported!")

    #print(f"Affecting layers: {layers[-args.n_layers:]}")
    for name, module in model.named_modules():
        if name in layers[-args.n_layers:]:
            module.apply(initialize)
    return model



if __name__ == '__main__':
    N = 2
    x = torch.rand(10,3,299,299)
    args = edict()
    args.arch = "resnet50"
    args.n_layers = 3       
    model = resnet18()
    #model.fc = nn.Identity()

    #model = SimpleCNN([32,64,128],4,add_pooling=False)
    print(model.conv1[0].weight[0][0][0])
    print(model.conv5_x[1].residual_function[0].weight[0][0][0])
    print(model.fc.weight[0])
    model = restart_model(args, model)
    print(model.conv1[0].weight[0][0][0])
    print(model.conv5_x[1].residual_function[0].weight[0][0][0])
    print(model.fc.weight[0])
    #for name, module in model.named_modules():
    #    print(name)#, module)