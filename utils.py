import torch
from torch import nn
import numpy as np
import cv2
import pickle
import subprocess
import re
import os


class GANLoss(nn.Module):
    def __init__(self, label_smoothing: float = 1.0):
        super(GANLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, inp, real: bool):
        if real:
            label = torch.full(inp.size(), self.label_smoothing,
                               requires_grad=True, device=inp.device)
        else:
            label = torch.full(inp.size(), 1 - self.label_smoothing,
                               requires_grad=True, device=inp.device)
        return self.criterion(inp, label)


def cvtLAB2RGBFromList(images):
    return [cv2.cvtColor(x, cv2.COLOR_LAB2RGB) for x in images]


def cvtRGB2BWFromList(images):
    return [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)[..., None].repeat(3, axis=-1) for x in images]


def load_file(path, mode='rb'):
    with open(path, mode=mode) as f:
        file = pickle.load(f)
    return file


def write_logging(text, fn):
    with open(fn, mode='w') as f:
        f.write(text)


def setEval(model, device=None, state_dicts=None):
    for k, v in model.items():
        if state_dicts != None:
            v.load_state_dict(state_dicts[k])
        for params in v.parameters():
            params.requires_grad = False
        v.eval()
        if device != None:
            v.to(device)


def download_image(url, path, pattern, rename=None):
    retrieved = subprocess.run(
        f'wget -nv -P {path}/ {url}'.split(' '), capture_output=True, text=True).stderr
    fp = re.search(r'"(.+)"', retrieved)
    fp = fp.group()[1:-1]
    fp_split = fp.split('/')
    if rename != None:
        os.rename(fp, '/'.join(fp_split[:-1]) + '/' + rename)
        fp = '/'.join(fp_split[:-1]) + '/' + rename
    return fp


def imagenet_norm(x):
    return (x - torch.tensor([0.485, 0.456, 0.406], dtype=x.dtype, device=x.device)[None, :, None, None])/torch.tensor([0.229, 0.224, 0.225], dtype=x.dtype, device=x.device)[None, :, None, None]


def calculate_perceptual_loss(x, y, loss_fn, model=None):
    if model != None:
        return loss_fn(model(x), model(y))
    return loss_fn(x, y)


class rgb2ycbcr():
    """rgb to ycbcr
    """

    def __init__(self, device='cpu', dtype=torch.float32):
        KR, KG, KB = 0.299, 0.587, 0.114
        self.rgb2ycbcr = torch.from_numpy(np.array(
            [
                [KR, KG, KB],
                [-KR/(1 - KB)/2, -KG/(1 - KB)/2, 0.5],
                [0.5, -KG/(1 - KR)/2, -KB/(1 - KR)/2],
            ]
        ).T).to(device).type(dtype)
        self.shift = torch.from_numpy(np.array([0., 0.5, 0.5])[
                                      None, None]).to(device).type(dtype)

    def __call__(self, x):
        return ((torch.matmul(x, self.rgb2ycbcr)/255. + self.shift - 0.5)/0.5)[..., [0]].permute(2, 0, 1)


class ycbcr2rgb():
    """ycbcr to rgb
    """

    def __init__(self, device='cpu', dtype=torch.float32):
        KR, KG, KB = 0.299, 0.587, 0.114
        self.ycbcr2rgb = torch.from_numpy(np.array(
            [
                [1., 0., 2*(1 - KR)],
                [1., -2*(KB/KG)*(1 - KB), -2*(KR/KG)*(1 - KR)],
                [1., 2*(1 - KB), 0.]
            ]
        ).T).to(device).type(dtype)
        self.shift = torch.from_numpy(np.array([0., 0.5, 0.5])[
                                      None, :, None, None]).to(device).type(dtype)

    def __call__(self, x):
        x = (x*0.5 + 0.5 - self.shift).permute(0, 2, 3, 1)
        return torch.matmul(x, self.ycbcr2rgb).permute(0, 3, 1, 2)
