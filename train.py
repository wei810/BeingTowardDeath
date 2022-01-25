import argparse
from utils import *
from model import *
from dataWrapper import Data
from tqdm import tqdm
from functools import partial
import pickle
import cv2
from PIL import Image
import numpy as np
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers import tensorboard_logger as tb_logger
from ignite.metrics import Average, RunningAverage
from ignite.engine import Engine, Events
from torchinfo import summary
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torchvision.transforms import *
from torchvision.utils import make_grid
from torch import DeviceObjType, nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
torch.cuda.device_count()
torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('log_dir', type=str)
    parser.add_argument('file_dict_path', type=str)
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--size', default=160, type=int)
    parser.add_argument('--generator_lr', default=0.00015, type=float)
    parser.add_argument('--critic_lr', default=0.00015, type=float)
    parser.add_argument('--content_loss_constant', default=50., type=float)
    parser.add_argument('--device', default='cuda:0', type=str)
    args = parser.parse_args()

    print(args)
    LOG_DIR = args.log_dir
    FILE_DICT_PATH = args.file_dict_path
    EPOCH = args.epoch
    BATCH_SIZE = args.batch_size
    SIZE = args.size
    GENERATOR_LR = args.generator_lr
    CRITIC_LR = args.critic_lr
    CONTENT_LOSS_CONSTANT = args.content_loss_constant
    DEVICE = args.device
    LOG_IMAGE_FREQ = 4000
    WEIGHT_SAVING_FREQ = 50000
    NUM_LOG_IMAGES = 32
    LABEL_SMOOTHING = 0.98

    fileDict = load_file(FILE_DICT_PATH)

    gen = Unet(UnetEncoder, UnetDecoder)
    txt = 'GENERATOR ARCHITECTURE:\n' + str(summary(gen))
    write_logging(txt, f'{LOG_DIR}/GENERATOR.txt')
    gen.to(DEVICE)
    critic = PatchCritic(3, CriticBasicBlock, nn.ReLU, nn.BatchNorm2d)
    txt = 'CRITIC ARCHITECTURE:\n' + str(summary(critic))
    write_logging(txt, f'{LOG_DIR}/CRITIC.txt')
    critic.to(DEVICE)

    vgg = models.vgg19(pretrained=True).features[:26]
    vgg.eval()
    vgg.to(DEVICE)

    opts = {
        'gen': optim.Adam(gen.parameters(), lr=GENERATOR_LR, eps=1e-3),
        'critic': optim.Adam(critic.parameters(), lr=CRITIC_LR, eps=1e-3)
    }

    trainDS = Data(fileDict['train'], SIZE, True)
    valDS = Data(fileDict['val'], SIZE, False)
    trainDL = DataLoader(trainDS, batch_size=BATCH_SIZE,
                         num_workers=48, pin_memory=True, shuffle=True)
    valDL = DataLoader(valDS, batch_size=BATCH_SIZE,
                       num_workers=48, pin_memory=True)

    criterionContent = nn.MSELoss()
    criterionGAN = GANLoss(label_smoothing=LABEL_SMOOTHING)

    cvt = ycbcr2rgb(device=DEVICE)

    scaler = GradScaler()

    def train_step(engine, batch):
        gen.train()
        critic.train()
        opts['gen'].zero_grad()
        opts['critic'].zero_grad()

        y, ycbcr = batch['y'].to(DEVICE), batch['ycbcr'].to(DEVICE)
        with autocast():
            generated = gen(y)
            fake_critic_input = generated.detach()
            fake_outs = critic(fake_critic_input)
            real_critic_input = ycbcr.detach()
            real_outs = critic(real_critic_input)
            d_loss = criterionGAN(fake_outs, False)*0.5 + \
                criterionGAN(real_outs, True)*0.5

        # critic step
        scaler.scale(d_loss).backward()
        scaler.step(opts['critic'])

        critic.eval()
        with autocast():
            fake_outs = critic(generated)
            fake = cvt(generated)
            real = cvt(ycbcr)
            perceptual_loss = calculate_perceptual_loss(
                imagenet_norm(fake),
                imagenet_norm(real),
                criterionContent,
                model=vgg
            )
            # generator steps
            gan_loss = criterionGAN(fake_outs, True)
            g_loss = CONTENT_LOSS_CONSTANT*perceptual_loss + gan_loss

        scaler.scale(g_loss).backward()
        scaler.step(opts['gen'])
        scaler.update()
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'gan_loss': gan_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
        }

    tb = tb_logger.TensorboardLogger(log_dir=LOG_DIR)

    trainEngine = Engine(train_step)
    names = ['g_loss', 'd_loss', 'gan_loss', 'perceptual_loss']

    def ot_func(output, name):
        return output[name]
    [RunningAverage(output_transform=partial(ot_func, name=name),
                    alpha=0.9).attach(trainEngine, name) for name in names]
    ProgressBar().attach(trainEngine)

    def val_step(engine, batch):
        with torch.no_grad():
            y, ycbcr = batch['y'].to(DEVICE), batch['ycbcr'].to(DEVICE)
            generated = gen(y).detach()
            fake_outs = critic(generated).detach()
            real_critic_input = ycbcr.detach()
            real_outs = critic(real_critic_input).detach()
            d_loss = criterionGAN(fake_outs, False)*0.5 + \
                criterionGAN(real_outs, True)*0.5
            fake = cvt(generated)
            real = cvt(ycbcr)
            perceptual_loss = calculate_perceptual_loss(imagenet_norm(
                fake), imagenet_norm(real), criterionContent, model=vgg).detach()
            gan_loss = criterionGAN(fake_outs, True)
            g_loss = CONTENT_LOSS_CONSTANT*perceptual_loss + gan_loss
            return {
                'g_loss': g_loss.item(),
                'd_loss': d_loss.item(),
                'gan_loss': gan_loss.item(),
                'perceptual_loss': perceptual_loss.item(),
            }

    valEngine = Engine(val_step)
    [Average(output_transform=partial(ot_func, name=name)
             ).attach(valEngine, name) for name in names]

    @trainEngine.on(Events.ITERATION_COMPLETED)
    def log(engine):

        if engine.state.iteration % LOG_IMAGE_FREQ == 0:
            gen.eval()

            print('Logging images to Tensorboard ...')
            np.random.seed(engine.state.iteration)
            train_picked = np.random.choice(
                fileDict['train'], NUM_LOG_IMAGES, replace=False)
            ds = Data(train_picked, SIZE, False)
            train = DataLoader(ds, batch_size=NUM_LOG_IMAGES)
            val_picked = np.random.choice(
                fileDict['val'], NUM_LOG_IMAGES, replace=False)
            ds = Data(val_picked, SIZE, False)
            val = DataLoader(ds, batch_size=NUM_LOG_IMAGES)
            dl = {'Train': train, 'Val': val}

            for phase in ['Train', 'Val']:
                for batch in dl[phase]:
                    with torch.no_grad():
                        y, ycbcr = batch['y'].to(
                            DEVICE), batch['ycbcr'].to(DEVICE)
                        generated = gen(y)
                    fake = torch.clamp(cvt(generated).cpu(), min=0., max=1.)
                    real = torch.clamp(cvt(ycbcr).cpu(), min=0., max=1.)
                    bw = (y*0.5 + 0.5).cpu()

                tb.writer.add_image(
                    f'{phase}/X', make_grid(bw), engine.state.iteration)
                tb.writer.add_image(
                    f'{phase}/Real', make_grid(real), engine.state.iteration)
                tb.writer.add_image(
                    f'{phase}/Fake', make_grid(fake), engine.state.iteration)

        if engine.state.iteration % WEIGHT_SAVING_FREQ == 0:
            state_dict = {
                'gen': gen.state_dict(),
                'critic': critic.state_dict()
            }
            path = LOG_DIR + f'/weight_{engine.state.iteration}_.pt'
            torch.save(state_dict, path)

        tb.writer.add_scalar(
            'G_Loss/Train', engine.state.metrics['g_loss'], engine.state.iteration)
        tb.writer.add_scalar(
            'D_Loss/Train', engine.state.metrics['d_loss'], engine.state.iteration)
        tb.writer.add_scalar(
            'G_Gan_Loss/Train', engine.state.metrics['gan_loss'], engine.state.iteration)
        tb.writer.add_scalar(
            'G_Perceptual_Loss/Train', engine.state.metrics['perceptual_loss'], engine.state.iteration)

    @trainEngine.on(Events.EPOCH_COMPLETED)
    def complete(engine):
        gen.eval()
        critic.eval()

        names = ['g_loss', 'd_loss', 'gan_loss', 'perceptual_loss']
        print('Training Results - Epoch[%d]' % (engine.state.epoch))
        for x in names:
            print(f'{x}={round(engine.state.metrics[x], 5)}')

        valEngine.run(valDL)
        print('Validating Results - Epoch[%d]' % (engine.state.epoch))
        for x in names:
            print(f'{x}={round(valEngine.state.metrics[x], 5)}')

        tb.writer.add_scalar(
            'G_Loss/Val', valEngine.state.metrics['g_loss'], engine.state.epoch)
        tb.writer.add_scalar(
            'D_Loss/Val', valEngine.state.metrics['d_loss'], engine.state.epoch)
        tb.writer.add_scalar(
            'G_Gan_Loss/Val', valEngine.state.metrics['gan_loss'], engine.state.epoch)
        tb.writer.add_scalar(
            'G_Perceptual_Loss/Val', valEngine.state.metrics['perceptual_loss'], engine.state.epoch)

    trainEngine.run(trainDL,  max_epochs=EPOCH)
    state_dict = {'gen': gen.state_dict(), 'critic': critic.state_dict()}
    torch.save(state_dict, LOG_DIR + '/weight.pt')


if __name__ == '__main__':
    main()
