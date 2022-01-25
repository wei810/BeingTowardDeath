from functools import partial
from typing import Type, Any, Callable, Union, List, Optional
from utils import *
from torchvision import models
from torch import nn
import torch
torch.cuda.is_available()


def conv1x1(inplanes: int, planes: int, stride: int = 1):
    return nn.Conv2d(inplanes, planes, 1, stride=stride)


def conv3x3(inplanes: int, planes: int, stride: int, padding: int = 1):
    return nn.Conv2d(inplanes, planes, 3, stride=stride, padding=padding)


class ResidualBasicBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int,
        activation_function: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        num_inner_conv_layers: int = 2,
    ):
        """the basic block for ResNet

        Args:
            kernel_size (int)
            inplanes (int): the number of the input channel
            planes (int): the number of the output channel
            stride (int)
            activation_function (Callable[..., nn.Module]): activation (e.g: ReLU)
            norm_layer (Callable[..., nn.Module]): normalization (e.g: BatchNorm)
            num_inner_conv_layers (int, optional): the number of convolution layers in a block. Defaults to 2.
        """
        super(ResidualBasicBlock, self).__init__()
        self.stride = stride
        self.process = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv2d(inplanes, inplanes, kernel_size,
                              padding=(kernel_size - 1)//2, stride=1),
                    norm_layer(inplanes),
                    activation_function(),
                ) for i in range(num_inner_conv_layers)
            ]
        )
        self.merge = nn.Sequential(
            conv1x1(inplanes*2, inplanes),
        )
        self.downscale = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes),
            activation_function(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.process(x)
        out = self.merge(torch.cat([x, out], dim=1))
        out = self.downscale(out)
        return out


class CriticBasicBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        inplanes: int,
        planes: int,
        stride: int,
        activation_function: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
    ):
        """the basic block for the discriminator network

        Args:
            kernel_size (int)
            inplanes (int): the number of the input channel
            planes (int): the number of the output channel
            stride (int)
            activation_function (Callable[..., nn.Module]): activation (e.g: ReLU)
            norm_layer (Callable[..., nn.Module]): normalization (e.g: BatchNorm)
        """
        super(CriticBasicBlock, self).__init__()
        self.stride = stride
        self.process = nn.Sequential(
            nn.Conv2d(inplanes, inplanes, kernel_size),
            norm_layer(inplanes),
            activation_function(),
        )
        self.downscale = nn.Sequential(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes),
            activation_function(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.process(x)
        out = self.downscale(out)
        return out


class UnetEncoder(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
        inplane_base: Optional[int] = 64,
    ):
        """Encoder part of the generator (Unet)

        Args:
            block: basic block
            layers (List[int]): the number of blocks in each layer (limit 3 elements) (e.g: [4, 3, 2])
            norm_layer (Callable[..., nn.Module]): normalization (e.g: BatchNorm)
            activation_function (Callable[..., nn.Module]): activation (e.g: ReLU)
            inplane_base (Optional[int], optional): the number of the input channel (basis). Scale up two time between layers. Defaults to 64.
        """
        super(UnetEncoder, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer
        self.block = nn.Sequential(
            nn.Conv2d(3, inplane_base, 5, stride=1, padding=2),
            norm_layer(inplane_base),
            self.activ(),
        )
        self.layer1 = self.__make_layer(block, 5, inplane_base*1, layers[0])
        self.bottle1 = nn.Sequential(
            conv1x1(inplane_base*1, inplane_base*1), self.norm_layer(inplane_base*1))
        self.down1 = block(3, inplane_base*1, inplane_base *
                           2, 2, self.activ, self.norm_layer)

        self.layer2 = self.__make_layer(block, 3, inplane_base*2, layers[1])
        self.bottle2 = nn.Sequential(
            conv1x1(inplane_base*2, inplane_base*2), self.norm_layer(inplane_base*2))
        self.down2 = block(3, inplane_base*2, inplane_base *
                           4, 2, self.activ, self.norm_layer)

        self.layer3 = self.__make_layer(block, 3, inplane_base*4, layers[2])
        self.bottle3 = nn.Sequential(
            conv1x1(inplane_base*4, inplane_base*4), self.norm_layer(inplane_base*4))
        self.down3 = block(3, inplane_base*4, inplane_base *
                           8, 2, self.activ, self.norm_layer)

    def __make_layer(
        self,
        block,
        kernel_size: int,
        planes: int,
        blocks: int,
    ):
        """a single layer

        Args:
            ...
            blocks (int): the number of blocks in a single layer

        Returns:
            nn.ModuleList 
        """
        layers = []
        for i in range(blocks):
            layers.append(block(kernel_size, planes, planes,
                          1, self.activ, self.norm_layer))
        return nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = {'x': x}

        out = self.block(x)
        outs['block'] = out

        out = self.__merge(out, self.layer1, self.bottle1)
        out = self.down1(out)
        outs['down1'] = out

        out = self.__merge(out, self.layer2, self.bottle2)
        out = self.down2(out)
        outs['down2'] = out

        out = self.__merge(out, self.layer3, self.bottle3)
        out = self.down3(out)
        outs['down3'] = out

        return outs

    def __merge(self, x: torch.Tensor, layers: nn.ModuleList, bottle: nn.Module):
        """feature extractor module
        As tensors flow through each block in a layer, the output of each block will be preserved and added together at last.
        Args:
            x (torch.Tensor): input
            ...

        Returns:
            torch.Tensor: output
        """
        out = 0.
        for l in layers:
            x = l(x)
            out = out + x
        out = bottle(out)
        return out


class Fusion(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
    ):
        """This module acts as "long connections" between the encoder and the decoder of a UNet.

        Args:
            inplanes (int): the number of the input channel
            planes (int): the number of the output channel
            activation_function (Callable[..., nn.Module]): activation (e.g: ReLU)
            norm_layer (Callable[..., nn.Module]): normalization (e.g: BatchNorm)
        """
        super(Fusion, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.norm_layer = norm_layer
        self.bottle = nn.Sequential(
            conv1x1(inplanes*2, planes), self.norm_layer(planes))
        self.activ = activation_function()

    def forward(self, a, b) -> torch.Tensor:
        out = self.bottle(torch.cat([a, b], dim=1))
        out = self.activ(out)
        return out


class UnetDecoder(nn.Module):
    def __init__(
        self,
        block,
        layers: List[int],
        norm_layer: Callable[..., nn.Module],
        activation_function: Callable[..., nn.Module],
        inplane_base: Optional[int] = 64,
    ):
        """Decoder part of the generator (UNet)

        Args:
            block: basic block
            layers (List[int]): the number of blocks in each layer (limit 3 elements) (e.g: [2, 3, 4])
            norm_layer (Callable[..., nn.Module]): normalization (e.g: BatchNorm)
            activation_function (Callable[..., nn.Module]): activation (e.g: ReLU)
            inplane_base (Optional[int], optional): the number of the input channel (basis). Scale up two time between layers. Defaults to 64.
        """
        super(UnetDecoder, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer

        self.up1 = self.__make_upsample_block(block, inplane_base*8)
        self.layer1 = self.__make_layer(block, 3, inplane_base*4, layers[0])
        self.bottle1 = nn.Sequential(
            conv1x1(inplane_base*4, inplane_base*4), norm_layer(inplane_base*4))
        self.fusion1 = Fusion(inplane_base*4, inplane_base*4,
                              norm_layer, activation_function)

        self.up2 = self.__make_upsample_block(block, inplane_base*4)
        self.layer2 = self.__make_layer(block, 3, inplane_base*2, layers[1])
        self.bottle2 = nn.Sequential(
            conv1x1(inplane_base*2, inplane_base*2), norm_layer(inplane_base*2))
        self.fusion2 = Fusion(inplane_base*2, inplane_base*2,
                              norm_layer, activation_function)

        self.up3 = self.__make_upsample_block(block, inplane_base*2)
        self.layer3 = self.__make_layer(block, 3, inplane_base*1, layers[2])
        self.bottle3 = nn.Sequential(
            conv1x1(inplane_base*1, inplane_base*1), norm_layer(inplane_base*1))

        self.block = nn.Sequential(
            block(3, inplane_base, inplane_base//2,
                  1, self.activ, self.norm_layer),
            block(3, inplane_base//2, inplane_base //
                  2, 1, self.activ, self.norm_layer),
        )

        self.output_block = nn.Sequential(
            conv1x1(inplane_base//2, 3),
            nn.Tanh(),
        )

    def forward(self, inp) -> torch.Tensor:
        x, block, down1, down2, down3 = inp['x'], inp['block'], inp['down1'], inp['down2'], inp['down3']

        down_output = self.__merge(self.up1(down3), self.layer1, self.bottle1)
        out = self.fusion1(down2, down_output)

        down_output = self.__merge(self.up2(out), self.layer2, self.bottle2)
        out = self.fusion2(down1, down_output)

        down_output = self.__merge(self.up3(out), self.layer3, self.bottle3)

        out = self.block(down_output)
        out = self.output_block(out)
        return out

    def __merge(self, x: torch.Tensor, layers: nn.ModuleList, bottle: nn.Module):
        """feature extractor module
        As tensors flow through each block in a layer, the output of each block will be preserved and added together at last.
        Args:
            x (torch.Tensor): input
            ...

        Returns:
            torch.Tensor: output
        """
        out = 0.
        for l in layers:
            x = l(x)
            out = out + x
        out = bottle(out)
        return out

    def __make_upsample_block(
        self,
        block,
        inplanes,
        dropout_rate: Union[None, float] = None,
        scale: int = 2,
        scale_factor: int = 2,
    ):
        """upsample block (to scale up the image) (e.g: (64, 128, 128) -> (32, 256, 256))

        Args:
            block ([type]): [description]
            inplanes ([type]): [description]
            dropout_rate (Union[None, float], optional): dropout. Defaults to None.
            scale (int, optional): to scale the number of channels (channels//scale). Defaults to 2.
            scale_factor (int, optional): to scale the size of the tensor. Defaults to 2.

        Returns:
            nn.Sequential
        """
        if scale_factor > 1:
            return nn.Sequential(
                block(3, inplanes, inplanes//scale,
                      1, self.activ, self.norm_layer),
                nn.Upsample(scale_factor=scale_factor),
            )
        else:
            return nn.Sequential(
                block(3, inplanes, inplanes//scale,
                      1, self.activ, self.norm_layer),
            )

    def __make_layer(
        self,
        block,
        kernel_size: int,
        planes: int,
        blocks: int,
    ):
        """a single layer

        Args:
            ...
            blocks (int): the number of blocks in a single layer

        Returns:
            nn.ModuleList 
        """
        layers = []
        for i in range(blocks):
            layers.append(block(kernel_size, planes, planes,
                          1, self.activ, self.norm_layer))
        return nn.ModuleList(layers)


def custom_leaky_relu(rate: float = 0.):
    return partial(nn.LeakyReLU, negative_slope=rate)


class Unet(nn.Module):
    def __init__(
        self,
        encoder: Callable[..., nn.Module],
        decoder: Callable[..., nn.Module],
        **kwargs,
    ):
        """UNet

        Args:
            encoder (Callable[..., nn.Module])
            decoder (Callable[..., nn.Module])
            kwargs (dict): keyword arguments
                inplane_base (int)
                encoder_block
                encoder_layers (list: three integers)
                encoder_norm_layer
                encoder_activation_function
                decoder_block
                decoder_layers (list: three integers)
                decoder_norm_layer
                decoder_activation_function
                -----
                Example:
                {
                    'inplane_base': 32,
                    'encoder_block': ResidualBasicBlock,
                    'encoder_layers': [4, 4, 4],
                    'encoder_norm_layer': nn.BatchNorm2d,
                    'encoder_activation_function': nn.ReLU,
                    'decoder_block': ResidualBasicBlock,
                    'decoder_layers': [4, 4, 4],
                    'decoder_norm_layer': nn.BatchNorm2d,
                    'decoder_activation_function': nn.ReLU,
                }

        """
        super(Unet, self).__init__()

        self.args = {
            'inplane_base': 32,
            'encoder_block': ResidualBasicBlock,
            'encoder_layers': [4, 4, 4],
            'encoder_norm_layer': nn.BatchNorm2d,
            'encoder_activation_function': nn.ReLU,
            'decoder_block': ResidualBasicBlock,
            'decoder_layers': [4, 4, 4],
            'decoder_norm_layer': nn.BatchNorm2d,
            'decoder_activation_function': nn.ReLU,
        }
        self.args.update(kwargs)
        self.encoder = encoder(
            self.args['encoder_block'],
            self.args['encoder_layers'],
            self.args['encoder_norm_layer'],
            self.args['encoder_activation_function'],
            inplane_base=self.args['inplane_base'],
        )
        self.decoder = decoder(
            self.args['decoder_block'],
            self.args['decoder_layers'],
            self.args['decoder_norm_layer'],
            self.args['decoder_activation_function'],
            inplane_base=self.args['inplane_base'],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.encoder(x)
        y = self.decoder(out)
        return y


class PatchCritic(nn.Module):
    def __init__(
        self,
        input_dims: int,
        layer_block,
        activation_function: Callable[..., nn.Module],
        norm_layer: nn.BatchNorm2d,
        inplane_base: Optional[int] = 64,
        out_inplanes: Optional[int] = 16,
        dropout_rate: Optional[Union[None, float]] = 0.5,
    ):
        """Patch GAN Critic

        Args:
            input_dims (int): color channel
            layer_block ([type]): basic block
            activation_function (Callable[..., nn.Module]): activation (e.g: ReLU)
            norm_layer: normalization (e.g: BatchNorm)
            inplane_base (Optional[int], optional): the number of the input channel (basis). Scale up two time between layers. Defaults to 64.
            out_inplanes (Optional[int], optional): the number of channels of the output. Defaults to 16.
            dropout_rate (Optional[Union[None, float]], optional): dropout. Defaults to 0.5.
        """
        super(PatchCritic, self).__init__()
        self.activ = activation_function
        self.norm_layer = norm_layer

        self.block = nn.Sequential(
            nn.Conv2d(input_dims, inplane_base, 5, stride=1),
            norm_layer(inplane_base),
            self.activ(),
        )

        self.layer1 = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            layer_block(5, inplane_base, inplane_base*2, 2,
                        activation_function, norm_layer),
        )

        self.layer2 = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            layer_block(3, inplane_base*2, inplane_base*4,
                        2, activation_function, norm_layer),
        )

        self.squeeze = nn.Sequential(
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(inplane_base*4, inplane_base*8, 3),
            norm_layer(inplane_base*8),
            self.activ(),
            nn.Conv2d(inplane_base*8, inplane_base*16, 3),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(inplane_base*16, out_inplanes, 1),
        )

    def forward(self, x: torch.Tensor):
        out = self.layer2(self.layer1(self.block(x)))
        out = self.squeeze(out)
        return out
