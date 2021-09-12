from torch.nn import Conv3d, ConvTranspose3d
from torch.nn import BatchNorm3d, GroupNorm, InstanceNorm3d
from torch.nn import ELU, LeakyReLU, ReLU, Linear
from torch.nn import Dropout3d, MaxPool3d, AdaptiveMaxPool3d

def convBlock(in_channels, out_channels, num_groups, conv_type, norm_type, acti_type):

    convBlock = []

    # add conv layer
    if conv_type == "conv":
        # input size (N,C,D,H,W)
        convBlock.append(["Conv3d", Conv3d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=3,
                                           groups=num_groups)])
    elif conv_type == "convT":
        # input size (N,C,D,H,W)
        convBlock.append(["ConvTrans3d", ConvTranspose3d(in_channels=in_channels,
                                                         out_channels=out_channels,
                                                         kernel_size=3,
                                                         groups=num_groups,
                                                         strides=2)])
    # add norm layer
    if norm_type == "batch":
        # num_features
        convBlock.append(["BatchNorm3d", BatchNorm3d(num_features=out_channels)])
    elif norm_type == "group":
        convBlock.append(["BatchNorm3d", GroupNorm(num_groups=num_groups,
                                                   num_features=out_channels)])
    elif norm_type == "instance":
        convBlock.append(["InstanceNorm3d", InstanceNorm3d(num_features=num_features)])
    elif norm_type == "none":
        pass

    # add activation layer
    if acti_type == "ReLU":
        convBlock.append(["ReLU", ReLU()])
    elif acti_type == "ELU":
        convBlock.append(["ELU", ELU()])
    elif acti_type == "LeakyReLU":
        convBlock.append(["LeakyReLU", LeakyReLU()])

    # debug print
    for item in convBlock:
        name = item[0]
        layer = item[1]
        print(name)











