import pandas as pd
from collections import defaultdict
import glob
from model import *
from utils import *
from dataWrapper import Data
from torchinfo import summary
from IPython.display import display
from tqdm.notebook import tqdm
from PIL import Image
import numpy as np
from typing import Union, List, Tuple
from glob import glob
import shutil
import os
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
print(f'GPU is available.\nNum:{torch.cuda.device_count():>3}' if torch.cuda.is_available(
) else 'GPU is not available.')


fileDict = load_file('coco_fileDict.p')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using \'{device}\'')
state_dicts = torch.load('result/weight_140000_.pt', map_location=device)
models = {'gen': Unet(UnetEncoder, UnetDecoder)}
setEval(models, device=device, state_dicts=state_dicts)
cvt = {
    'rgb2ycbcr': rgb2ycbcr(device=device),
    'ycbcr2rgb': ycbcr2rgb(device=device),
}
path = 'images_coco_evaluation'
try:
    shutil.rmtree(path)
except Exception as e:
    pass
os.mkdir(path)
size = 512
bs = 16

phase = ('train', 'val', 'test')
ds = Data(fileDict[phase[0]], size, False)
train = DataLoader(ds, batch_size=bs)
ds = Data(fileDict[phase[1]], size, False)
val = DataLoader(ds, batch_size=bs)
ds = Data(fileDict[phase[2]], size, False)
test = DataLoader(ds, batch_size=bs)
dl = dict(zip(phase, (train, val, test)))
phase = ('test', )

size = str(size)
os.mkdir(os.path.join(path, size))
for p in phase:
    print(p)
    os.mkdir(os.path.join(path, size, f'{p}_generated'))
    os.mkdir(os.path.join(path, size, f'{p}_target'))
    for i, batch in tqdm(enumerate(dl[p])):
        with torch.no_grad():
            y, ycbcr = batch['y'].to(device), batch['ycbcr'].to(device)
            generated = models['gen'](torch.clip(y, -1., 1.))
            fake = cvt['ycbcr2rgb'](torch.cat([y[:, [0], ...], generated[:, [
                                    1, 2], ...]], dim=1)).cpu().permute(0, 2, 3, 1).numpy()
            real = cvt['ycbcr2rgb'](ycbcr).cpu().permute(0, 2, 3, 1).numpy()
            fake = (np.clip(fake, 0., 1.)*255.).astype(np.uint8)
            real = (np.clip(real, 0., 1.)*255.).astype(np.uint8)
            for j in range(len(real)):
                Image.fromarray(fake[j]).save(os.path.join(
                    path, size, f'{p}_generated', f'{i*bs + j}.jpg'))
                Image.fromarray(real[j]).save(os.path.join(
                    path, size, f'{p}_target', f'{i*bs + j}.jpg'))
            #display_result_real_fake(real, fake)


def calculate_metric_values(source_img_folder, target_img_folder, p=1.0):
    result = defaultdict(list)
    files_num = len(glob.glob(os.path.join(source_img_folder, '*.jpg')))
    for i in tqdm(range(files_num)):
        x_img = mpimg.imread(os.path.join(source_img_folder, f'{i}.jpg'))
        y_img = mpimg.imread(os.path.join(target_img_folder, f'{i}.jpg'))
        mse = metrics.mean_squared_error(y_img / 255., x_img / 255.)
        #psnr = metrics.peak_signal_noise_ratio(y_img.mean(-1), x_img.mean(-1), data_range=255.)
        psnr = 20.*np.log10(1/mse**0.5)
        ssim = metrics.structural_similarity(
            y_img, x_img, data_range=255., multichannel=True)
        result['mse'].append(mse)
        result['psnr'].append(psnr)
        result['ssim'].append(ssim)
    result['mse'] = np.array(result['mse'])
    result['psnr'] = np.array(result['psnr'])
    result['ssim'] = np.array(result['ssim'])
    if p < 1.0:
        for i in result.keys():
            low = np.percentile(result[i], (1 - p)/2*100)
            high = np.percentile(result[i], (1 - (1 - p)/2)*100)
            filt = (result[i] >= low) & (result[i] < high)
            result[i] = result[i][filt]
    print({key: value.mean() for key, value in result.items()})
    return result


phase = {'train': None, 'val': None, 'test': None}
phase = {'test': None}
for p in phase.keys():
    print(p)
    phase[p] = calculate_metric_values(
        f'./images_coco_evaluation/160/{p}_generated',
        f'./images_coco_evaluation/160/{p}_target'
    )
