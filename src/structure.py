"""Internal structures used in the network."""

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F


class DirFullDil(nn.Module):
    """
    Performs horizontal, vertical and full convolutions, concatenate them, then perform the same number of horizontal, vertical and full convolutions.
    In parallel, performs horizontal, vertical and full convolutions once, but with a dilation factor of 2.
    Returns the concatenated results
    number of parametres: (2*n_dir + n_full)*(1 + 2*n_dir + n_full + channels_in) + (channels_in + 1)*(2*n_dir_dil + n_full_dil)
    """
    def __init__(self, channels_in, *n_convolutions):
        super(DirFullDil, self).__init__()
        n_dir, n_full, n_dir_dil, n_full_dil = n_convolutions
        self.h1 = nn.Conv2d(channels_in, n_dir, (1, 3))
        self.h2 = nn.Conv2d(2*n_dir+n_full, n_dir, (1, 3))
        self.v1 = nn.Conv2d(channels_in, n_dir, (3, 1))
        self.v2 = nn.Conv2d(2*n_dir+n_full, n_dir, (3, 1))
        self.f1 = nn.Conv2d(channels_in, n_full, 3)
        self.f2 = nn.Conv2d(2*n_dir+n_full, n_full, 3)
        self.hd = nn.Conv2d(channels_in, n_dir_dil, (1, 3), dilation=2)
        self.vd = nn.Conv2d(channels_in, n_dir_dil, (3, 1), dilation=2)
        self.fd = nn.Conv2d(channels_in, n_full_dil, 3, dilation=2)
        self.channels_out = 2*n_dir + n_full + 2*n_dir_dil + n_full_dil
        
    def forward(self, x):
        h_d = self.hd(x)[:, :, 2:-2]
        v_d = self.vd(x)[:, :, :, 2:-2]
        f_d = self.fd(x)
        h = self.h1(x)[:, :, 1:-1]
        v = self.v1(x)[:, :, :, 1:-1]
        f = self.f1(x)
        x = F.softplus(torch.cat((h, v, f), 1))
        h = self.h2(x)[:, :, 1:-1]
        v = self.v2(x)[:, :, :, 1:-1]
        f = self.f2(x)
        return torch.cat((h_d, v_d, f_d, h, v, f), 1)



class SkipDoubleDirFullDir(nn.Module):
    """Uses a first DirFullDir module, skips the input to the results of the first module, and finally send everything through a second DirFullDir module."""
    def __init__(self, channels_in, convolutions_1, convolutions_2):
        super(SkipDoubleDirFullDir, self).__init__()
        self.conv1 = DirFullDil(3, *convolutions_1)
        self.conv2 = DirFullDil(3 + self.conv1.channels_out, *convolutions_2)
        self.channels_out = self.conv2.channels_out
        self.padding = 4
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = torch.cat((x[:, :, 2:-2, 2:-2], x1), 1)
        x2 = self.conv2(x1)
        x2 = torch.cat((x1[:, :, 2:-2, 2:-2], x2), 1)
        return x2


class SeparateAndPermutate(nn.Module):
    def forward(self, x):
        n, C, Y, X = x.shape
        assert n==1
        assert Y%2==0
        assert X%2==0
        x_00 = x[:, :, ::2, ::2]
        x_01 = x[:, :, ::2, 1::2]
        x_10 = x[:, :, 1::2, ::2]
        x_11 = x[:, :, 1::2, 1::2]
        
        ind = [k+C*i for k in range(C) for i in range(4)]
        
        xx_00 = torch.cat((x_00, x_01, x_10, x_11), 1)[:, ind]
        xx_01 = torch.cat((x_01, x_00, x_11, x_10), 1)[:, ind]
        xx_10 = torch.cat((x_10, x_11, x_00, x_01), 1)[:, ind]
        xx_11 = torch.cat((x_11, x_10, x_01, x_00), 1)[:, ind]
        
        x = torch.cat((xx_00, xx_01, xx_10, xx_11), 0)
        return x



class Pixelwise(nn.Module):
    def __init__(self):
        super(Pixelwise, self).__init__()
        self.conv1 = nn.Conv2d(103, 30, 1)
        self.conv2 = nn.Conv2d(30, 15, 1)
        self.conv3 = nn.Conv2d(45, 15, 1)
        self.conv4 = nn.Conv2d(60, 30, 1)
    
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(x))
        x3 = F.leaky_relu(self.conv3(torch.cat((x, x2), 1)))
        x4 = self.conv4(torch.cat((x, x2, x3), 1))
        return x4
        

class FullNet(nn.Module):
    """
    Full network
    Input image: (1, 3, 2Y, 2X)
    ⮟First module: Spatial convolutions, without pooling
    Pixel-wise features: (1, 30 , 2Y-8, 2X-8) (⮞Pixelwise auxiliary training)
    ⮟Grid separation and permutations
    : (1, 120, Y-4, X-4)
    ⮟Permutations of the grid pixels
    Pixel-wise features, with the pixels separated by position in the grid in different channels, permutated in the four possible ways in image numbers: (4, 120, Y-4, X-4)
    ⮟ 1×1 convolutions: pixel-wise causality
    ⮟Average Pooling to get the mean in each block
    Mean in each block, channel and permutation of the pixels' features: (4N, 120, (Y-4)//block_size, (X-4)//block_size
    ⮟1×1 convolutions: block-wise causality
    Features in each block, channel and permutation of the pixels's features: (4N, 4, (Y-4)//block_size, (X-4)//block_size)
    ⮟LogSoftMax
    Out
    """
    def __init__(self):
        """
        :param spatial: nn.Module, must have spatial.padding defined
        :param causal: CausalNet instance
        output channels of spatial must match with input channels of causal
        """
        super(FullNet, self).__init__()
        self.spatial = SkipDoubleDirFullDir(3, (10, 5, 10, 5), (10, 5, 10, 5))
        self.pixelwise = Pixelwise()
        self.blockwise = nn.Sequential(nn.Conv2d(120, 180, 1, groups=30), nn.Softplus(), nn.Conv2d(180, 90, 1, groups=30), nn.Softplus(), nn.Conv2d(90, 90, 1, groups=30), nn.Softplus(), nn.Conv2d(90, 45, 1), nn.Softplus(), nn.Conv2d(45, 45, 1), nn.Softplus(), nn.Conv2d(45, 4, 1), nn.LogSoftmax(dim=1))
        self.pixelwise = Pixelwise()
        self.auxiliary = nn.Sequential(self.spatial, self.pixelwise, nn.Conv2d(30, 4, 1), nn.LogSoftmax(dim=1))
        self.grids = SeparateAndPermutate()
        self.padding = self.spatial.padding
    
    def forward(self, x, block_size=32):
        x = self.spatial(x)
        x = self.pixelwise(x)
        x = self.grids(x)
        x = F.avg_pool2d(x, block_size//2)
        x = self.blockwise(x)
        return x


class SelfNLLLoss(nn.Module):
    """Modified version of nn.NLLLoss for blockwise training, taking into account the fact that the target can be deduced from the output."""
    def forward(self, o, global_best=False):
        N, C, Y, X = o.shape
        if global_best:
            scores = torch.mean(o, axis=(2, 3)).detach().cpu().numpy()
            score_00 = scores[0, 0] + scores[1, 1] + scores[2, 2] + scores[3, 3]
            score_01 = scores[0, 1] + scores[1, 0] + scores[2, 3] + scores[3, 2]
            score_10 = scores[0, 2] + scores[1, 3] + scores[2, 0] + scores[3, 1]
            score_11 = scores[0, 3] + scores[1, 2] + scores[2, 1] + scores[3, 0]
            best = np.argmax((score_00, score_01, score_10, score_11))
        else:
            best = 0
        target = torch.zeros((4, Y, X), dtype=torch.long)
        target[0] = [0, 1, 2, 3][best]
        target[1] = [1, 0, 3, 2][best]
        target[2] = [2, 3, 0, 1][best]
        target[3] = [3, 2, 1, 0][best]
        target = target.cuda()
        loss = F.nll_loss(o, target)
        return loss


class SelfPixelwiseNLLLoss(nn.Module):
    """Modified version of nn.NLLLoss, for pixelwise auxiliary training."""
    def forward(self, o, δy=0, δx=0, global_best=False):
        N, C, Y, X = o.shape
        assert N==1
        if global_best:
            score_00 = torch.sum(o[0, 0, ::2, ::2] + o[0, 1, ::2, 1::2] + o[0, 2, 1::2, ::2] + o[0, 3, 1::2, 1::2]).item()
            score_01 = torch.sum(o[0, 0, ::2, 1::2] + o[0, 1, ::2, ::2] + o[0, 2, 1::2, 1::2] + o[0, 3, 1::2, ::2]).item()
            score_10 = torch.sum(o[0, 0, 1::2, ::2] + o[0, 1, 1::2, 1::2] + o[0, 2, ::2, ::2] + o[0, 3, ::2, 1::2]).item()
            score_11 = torch.sum(o[0, 0, 1::2, 1::2] + o[0, 1, 1::2, ::2] + o[0, 2, ::2, 1::2] + o[0, 3, ::2, ::2]).item()
            best = np.argmax((score_00, score_01, score_10, score_11))
            δy = best//2
            δx = best%2
        target = torch.zeros((N, Y, X), dtype=torch.long)
        #target[:, δy::2, δx::2] = 0
        target[:, δy::2, 1-δx::2] = 1
        target[:, 1-δy::2, δx::2] = 2
        target[:, 1-δy::2, 1-δx::2] = 3
        target = target.cuda()
        loss = F.nll_loss(o, target)
        return loss
        


