import torch
import torch.nn as nn
from inplace_abn import ABN


def calc_activation(ABN_layer):
    activation = None
    if isinstance(ABN_layer, ABN):
        if ABN_layer.activation == "relu":
            activation = nn.ReLU(inplace=True)
        elif ABN_layer.activation == "leaky_relu":
            activation = nn.LeakyReLU(negative_slope=ABN_layer.activation_param, inplace=True)
        elif ABN_layer.activation == "elu":
            activation = nn.ELU(alpha=ABN_layer.activation_param, inplace=True)
    return activation


def fuse_bn_to_conv(bn_layer, conv_layer):
    # print('bn fuse')
    bn_st_dict = bn_layer.state_dict()
    conv_st_dict = conv_layer.state_dict()

    # BatchNorm params
    eps = bn_layer.eps
    mu = bn_st_dict['running_mean']
    var = bn_st_dict['running_var']
    gamma = bn_st_dict['weight']

    if 'bias' in bn_st_dict:
        beta = bn_st_dict['bias']
    else:
        beta = torch.zeros(gamma.size(0)).float().to(gamma.device)

    # Conv params
    W = conv_st_dict['weight']
    if 'bias' in conv_st_dict:
        bias = conv_st_dict['bias']
    else:
        bias = torch.zeros(W.size(0)).float().to(gamma.device)

    denom = torch.sqrt(var + eps)
    b = beta - gamma.mul(mu).div(denom)
    A = gamma.div(denom)
    bias *= A
    A = A.expand_as(W.transpose(0, -1)).transpose(0, -1)

    W.mul_(A)
    bias.add_(b)

    conv_layer.weight.data.copy_(W)
    if conv_layer.bias is None:
        conv_layer.bias = torch.nn.Parameter(bias)
    else:
        conv_layer.bias.data.copy_(bias)


def fuse_bn_sequential(block):
    """
    This function takes a sequential block and fuses the batch normalization with convolution
    :param model: nn.Sequential. Source resnet model
    :return: nn.Sequential. Converted block
    """

    if not isinstance(block, nn.Sequential) and not hasattr(block, 'bn'):
        return block
    stack = []
    if isinstance(block, nn.Sequential) and len(block) == 1 and isinstance(block[0], nn.Sequential):  # 'downsample' layer in tresnet
        block = block[0]
    for m in block.children():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN):
            if isinstance(stack[-1], nn.Conv2d):
                fuse_bn_to_conv(m, stack[-1])
                if isinstance(m, ABN):
                    activation = calc_activation(m)
                    if activation is not None:
                        stack.append(activation)
        elif isinstance(m, nn.BatchNorm1d) and len(stack) > 0 and isinstance(stack[-1], torch.nn.Linear):
            fuse_bn_to_conv(m, stack[-1])
        else:
            stack.append(m)

    if len(stack) > 1:
        return nn.Sequential(*stack)
    else:
        return stack[0]


def fuse_bn_recursively(model):
    for module_name in model._modules:
        model._modules[module_name] = fuse_bn_sequential(model._modules[module_name])
        if len(model._modules[module_name]._modules) > 0:
            fuse_bn_recursively(model._modules[module_name])

    return model
