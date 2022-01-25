from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import *


class Data(Dataset):
    def __init__(self, fileList, size, training: bool = True):
        """torch dataset object

        Args:
            fileList (list): a list containing paths of images
            size (tuple(int, int)): image size  
            training (bool, optional): is training or not. Defaults to True.
        """
        super().__init__()
        self.fileList = fileList
        self.size = (size, size)
        KR, KG, KB = 0.299, 0.587, 0.114
        # the matrix to convert rgb to ycbcr colorspace
        self.rgb2ycbcr = np.array(
            [
                [KR, KG, KB],
                [-KR/(1 - KB)/2, -KG/(1 - KB)/2, 0.5],
                [0.5, -KG/(1 - KR)/2, -KB/(1 - KR)/2],
            ]
        ).T
        self.shift = np.array([0., 0.5, 0.5])[None, None]
        if training:
            self.transform = Compose([
                Resize((size, size)),
            ])
        else:
            self.transform = Compose([
                Resize((size, size)),
            ])

    @property
    def length(self):
        return len(self.fileList)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """

        Args:
            ...

        Returns:
            dict: {'y': grayscale images, 'ycbcr': real images}
        """
        rgb = np.array(self.transform(
            Image.open(self.fileList[idx]).convert('RGB')))
        ycbcr = (rgb@self.rgb2ycbcr/255. + self.shift - 0.5)/0.5
        return {
            'y': torch.Tensor(ycbcr[..., [0]]).permute(2, 0, 1).repeat_interleave(3, dim=0),
            'ycbcr': torch.Tensor(ycbcr).permute(2, 0, 1),
        }
