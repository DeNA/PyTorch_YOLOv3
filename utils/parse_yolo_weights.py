
from __future__ import division
import torch
import numpy as np


def parse_conv_block(m, weights, offset, initflag):
    """
    Initialization of conv layers with batchnorm
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m[0]
    bn_model = m[1]
    param_length = m[1].bias.numel()

    # batchnorm
    for pname in ['bias', 'weight', 'running_mean', 'running_var']:
        layerparam = getattr(bn_model, pname)

        if initflag: # yolo initialization - scale to one, bias to zero
            if pname == 'weight':
                weights = np.append(weights, np.ones(param_length))
            else:
                weights = np.append(weights, np.zeros(param_length))

        param = torch.from_numpy(weights[offset:offset + param_length]).view_as(layerparam)
        layerparam.data.copy_(param)
        offset += param_length

    param_length = conv_model.weight.numel()

    # conv
    if initflag: # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights

def parse_yolo_block(m, weights, offset, initflag):
    """
    YOLO Layer (one conv with bias) Initialization
    Args:
        m (Sequential): sequence of layers
        weights (numpy.ndarray): pretrained weights data
        offset (int): current position in the weights file
        initflag (bool): if True, the layers are not covered by the weights file. \
            They are initialized using darknet-style initialization.
    Returns:
        offset (int): current position in the weights file
        weights (numpy.ndarray): pretrained weights data
    """
    conv_model = m._modules['conv']
    param_length = conv_model.bias.numel()

    if initflag: # yolo initialization - bias to zero
        weights = np.append(weights, np.zeros(param_length))

    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.bias)
    conv_model.bias.data.copy_(param)
    offset += param_length

    param_length = conv_model.weight.numel()

    if initflag: # yolo initialization
        n, c, k, _ = conv_model.weight.shape
        scale = np.sqrt(2 / (k * k * c))
        weights = np.append(weights, scale * np.random.normal(size=param_length))
 
    param = torch.from_numpy(
        weights[offset:offset + param_length]).view_as(conv_model.weight)
    conv_model.weight.data.copy_(param)
    offset += param_length

    return offset, weights

def parse_yolo_weights(model, weights_path):
    """
    Parse YOLO (darknet) pre-trained weights data onto the pytorch model
    Args:
        model : pytorch model object
        weights_path (str): path to the YOLO (darknet) pre-trained weights file
    """
    fp = open(weights_path, "rb")

    # skip the header
    header = np.fromfile(fp, dtype=np.int32, count=5) # not used
    # read weights 
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    offset = 0 
    initflag = False #whole yolo weights : False, darknet weights : True

    for m in model.module_list:

        if m._get_name() == 'Sequential':
            # normal conv block
            offset, weights = parse_conv_block(m, weights, offset, initflag)

        elif m._get_name() == 'resblock':
            # residual block
            for modu in m._modules['module_list']:
                for blk in modu:
                    offset, weights = parse_conv_block(blk, weights, offset, initflag)

        elif m._get_name() == 'YOLOLayer':
            # YOLO Layer (one conv with bias) Initialization
            offset, weights = parse_yolo_block(m, weights, offset, initflag)

        initflag = (offset >= len(weights)) # the end of the weights file. turn the flag on
