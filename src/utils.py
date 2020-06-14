"""Various utilitary functions."""

import io


import PIL
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

def n_params(net):
    """Computes the number of trainable parametres of a nn.Module instance."""
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def img_to_tensor(img):
    """
    Turns a numpy image into a torch tensor
    :param img: np.ndarray, shape (Y, X, C)
    :output t: torch.tensor, shape (1, C, Y, X)
    """
    t = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
    return t


def jpeg_compress(img, quality):
    """Applies JPEG compression to an image."""
    img = img.copy()
    if img.max() <= 1.:
        img *= 255
    pimg = PIL.Image.fromarray(img.astype(np.uint8))
    out = io.BytesIO()
    pimg.save(out, format='JPEG', quality=quality)
    out.seek(0)
    result = np.array(PIL.Image.open(out)).astype(float)/255
    return result
