import torch
import torch.nn as nn

from collections import defaultdict
from models.yolo_layer import YOLOLayer

def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage


class resblock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """
    def __init__(self, ch, nblocks=1, shortcut=True):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(add_conv(ch, ch//2, 1, 1))
            resblock_one.append(add_conv(ch//2, ch, 3, 1))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


def create_yolov3_modules(config_model, ignore_thre):
    """
    Build yolov3 layer modules.
    Args:
        config_model (dict): model configuration.
            See YOLOLayer class for details.
        ignore_thre (float): used in YOLOLayer.
    Returns:
        mlist (ModuleList): YOLOv3 module list.
    """

    # DarkNet53
    mlist = nn.ModuleList()
    mlist.append(add_conv(in_ch=3, out_ch=32, ksize=3, stride=1))
    mlist.append(add_conv(in_ch=32, out_ch=64, ksize=3, stride=2))
    mlist.append(resblock(ch=64))
    mlist.append(add_conv(in_ch=64, out_ch=128, ksize=3, stride=2))
    mlist.append(resblock(ch=128, nblocks=2))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=2))
    mlist.append(resblock(ch=256, nblocks=8))    # shortcut 1 from here
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=2))
    mlist.append(resblock(ch=512, nblocks=8))    # shortcut 2 from here
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=2))
    mlist.append(resblock(ch=1024, nblocks=4))

    # YOLOv3
    mlist.append(resblock(ch=1024, nblocks=2, shortcut=False))
    mlist.append(add_conv(in_ch=1024, out_ch=512, ksize=1, stride=1))
    # 1st yolo branch
    mlist.append(add_conv(in_ch=512, out_ch=1024, ksize=3, stride=1))
    mlist.append(
         YOLOLayer(config_model, layer_no=0, in_ch=1024, ignore_thre=ignore_thre))

    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
    mlist.append(add_conv(in_ch=768, out_ch=256, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(resblock(ch=512, nblocks=1, shortcut=False))
    mlist.append(add_conv(in_ch=512, out_ch=256, ksize=1, stride=1))
    # 2nd yolo branch
    mlist.append(add_conv(in_ch=256, out_ch=512, ksize=3, stride=1))
    mlist.append(
        YOLOLayer(config_model, layer_no=1, in_ch=512, ignore_thre=ignore_thre))

    mlist.append(add_conv(in_ch=256, out_ch=128, ksize=1, stride=1))
    mlist.append(nn.Upsample(scale_factor=2, mode='nearest'))
    mlist.append(add_conv(in_ch=384, out_ch=128, ksize=1, stride=1))
    mlist.append(add_conv(in_ch=128, out_ch=256, ksize=3, stride=1))
    mlist.append(resblock(ch=256, nblocks=2, shortcut=False))
    mlist.append(
         YOLOLayer(config_model, layer_no=2, in_ch=256, ignore_thre=ignore_thre))

    return mlist


class YOLOv3(nn.Module):
    """
    YOLOv3 model module. The module list is defined by create_yolov3_modules function. \
    The network returns loss values from three YOLO layers during training \
    and detection results during test.
    """
    def __init__(self, config_model, ignore_thre=0.7):
        """
        Initialization of YOLOv3 class.
        Args:
            config_model (dict): used in YOLOLayer.
            ignore_thre (float): used in YOLOLayer.
        """
        super(YOLOv3, self).__init__()

        if config_model['TYPE'] == 'YOLOv3':
            self.module_list = create_yolov3_modules(config_model, ignore_thre)
        else:
            raise Exception('Model name {} is not available'.format(config_model['TYPE']))

    def forward(self, x, targets=None):
        """
        Forward path of YOLOv3.
        Args:
            x (torch.Tensor) : input data whose shape is :math:`(N, C, H, W)`, \
                where N, C are batchsize and num. of channels.
            targets (torch.Tensor) : label array whose shape is :math:`(N, 50, 5)`

        Returns:
            training:
                output (torch.Tensor): loss tensor for backpropagation.
            test:
                output (torch.Tensor): concatenated detection results.
        """
        train = targets is not None
        output = []
        self.loss_dict = defaultdict(float)
        route_layers = []
        for i, module in enumerate(self.module_list):
            # yolo layers
            if i in [14, 22, 28]:
                if train:
                    x, *loss_dict = module(x, targets)
                    for name, loss in zip(['xy', 'wh', 'conf', 'cls', 'l2'] , loss_dict):
                        self.loss_dict[name] += loss
                else:
                    x = module(x)
                output.append(x)
            else:
                x = module(x)

            # route layers
            if i in [6, 8, 12, 20]:
                route_layers.append(x)
            if i == 14:
                x = route_layers[2]
            if i == 22:  # yolo 2nd
                x = route_layers[3]
            if i == 16:
                x = torch.cat((x, route_layers[1]), 1)
            if i == 24:
                x = torch.cat((x, route_layers[0]), 1)
        if train:
            return sum(output)
        else:
            return torch.cat(output, 1)

