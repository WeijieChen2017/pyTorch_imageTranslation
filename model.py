from torch.nn import Conv3d, ConvTranspose3d
from torch.nn import BatchNorm3d, GroupNorm, InstanceNorm3d
from torch.nn import ELU, LeakyReLU, ReLU, Linear
from torch.nn import Dropout3d, MaxPool3d, AdaptiveMaxPool3d
from torch import nn

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.net_list = unet3d()

    def forward(self, x):
        for item in self.net_list:
            name, layer = item
            color_layer(layer)
            print(x.size())
            x = layer(x)
        return x

def color_layer(item):
    print(item)
    name = item[0]
    layer = item[1]
    if name == "Conv3d":
        print("\33[34m", layer)
    elif "Norm" in name:
        print("\33[33m", layer)
    elif "LU" in name:
        print("\33[32m", layer)
    elif "Pool" in name or "Trans" in name:
        print("\33[31m", layer)
    else:
        print("\33[35m", layer)

def network_visualization(network):
    for item in network:
        print(item)
        color_layer(item)


def convBlock(in_channels, out_channels, num_groups, norm_type, acti_type):

    convBlock = []

    # add conv layer
    # input size (N,C,D,H,W)
    convBlock.append(["Conv3d", Conv3d(in_channels=in_channels,
                                       out_channels=out_channels,
                                       kernel_size=3,
                                       groups=num_groups)])
    # add norm layer
    if norm_type == "batch":
        # num_features
        convBlock.append(["BatchNorm3d", BatchNorm3d(num_features=out_channels)])
    elif norm_type == "group":
        convBlock.append(["GroupNorm", GroupNorm(num_groups=num_groups,
                                                   num_features=out_channels)])
    elif norm_type == "instance":
        convBlock.append(["InstanceNorm3d", InstanceNorm3d(num_features=out_channels)])
    elif norm_type == "none":
        pass

    # add activation layer
    if acti_type == "ReLU":
        convBlock.append(["ReLU", ReLU()])
    elif acti_type == "ELU":
        convBlock.append(["ELU", ELU()])
    elif acti_type == "LeakyReLU":
        convBlock.append(["LeakyReLU", LeakyReLU()])

    return convBlock

def unet3d(num_filters=16, num_level=2, num_groups=1):

    unet3d = []

    block = convBlock(in_channels = 1,
                      out_channels = num_filters,
                      num_groups = num_groups,
                      norm_type = "batch",
                      acti_type = "LeakyReLU")
    unet3d.append(block)

    # encoder
    for idx in range(num_level):
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(block)
        unet3d.append(block)
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters * 2,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(block)
        unet3d.append(["MaxPool3d", MaxPool3d(kernel_size=3, stride=2)])
        num_filters = num_filters * 2

    # lowest learning
    block = convBlock(in_channels = num_filters,
                      out_channels = num_filters,
                      num_groups = num_groups,
                      norm_type = "batch",
                      acti_type = "LeakyReLU")
    unet3d.append(block)
    unet3d.append(block)
    unet3d.append(["Dropout3d", Dropout3d()])
    unet3d.append(block)
    unet3d.append(block)

    # decoder
    for idx in range(num_level):
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(["ConvTrans3d", ConvTranspose3d(in_channels=num_filters,
                                                      out_channels=num_filters,
                                                      kernel_size=3,
                                                      groups=num_groups,
                                                      stride=2)])
        unet3d.append(block)
        unet3d.append(block)
        block = convBlock(in_channels = num_filters,
                          out_channels = num_filters // 2,
                          num_groups = num_groups,
                          norm_type = "batch",
                          acti_type = "LeakyReLU")
        unet3d.append(block)
        num_filters = num_filters // 2
    unet3d.append(["Conv3d", Conv3d(in_channels=num_filters,
                                    out_channels=num_filters,
                                    kernel_size=3,
                                    groups=num_groups)])
    unet3d.append(["Linear", Linear(in_features = num_filters,
                                    out_features = 1)])

    # flatten the list of layer
    unet3d_flatten = []
    for item in unet3d:
        if isinstance(item[0], list):
            for elem in item:
                unet3d_flatten.append(elem)
        else:
            unet3d_flatten.append(item)
    # network_visualization(unet3d_flatten)

    return unet3d_flatten


