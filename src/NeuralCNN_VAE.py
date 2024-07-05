import sys
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.init as torch_init
import torch.optim as optim

# Data utils and dataloader
import torchvision
from torchvision import transforms, utils
import torchvision.models as models

from src.CNNVAE import CNNVAE

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample_shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        self.downsample = downsample

    def forward(self, input):
        # print("input_shape:",input.shape)
        if self.downsample:
            shortcut = self.downsample_shortcut(input)
        else:
            shortcut = self.shortcut(input)
        # print(self.shortcut(input).shape)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        # print("input resulting shape:", input.shape)
        # print("shortcut:", self.shortcut)
        # print("shortcut shape:", shortcut.shape)
        input = input + shortcut
        return nn.ReLU()(input)

class NeuralCNNEncoder(nn.Module):
    def __init__(self, resblock=ResBlock):
        self.in_channels = 1
        self.latent_space_dim =  128

        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(self.in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = torch.nn.Linear(128, self.latent_space_dim)
        # self.ac = torch.nn.Sigmoid()

    def forward(self, input):
        # print("original input size",input.shape)
        input = self.layer0(input)
        #print(input.shape)
        input = self.layer1(input)
        #print(input.shape)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        #print(input.shape)
        input = self.gap(input)
        # input = self.ac(input)
        input = input.view(input.size(0), -1)
        # #print(input.shape)
        #print(input.shape)
        return input
    
class UnFlatten(nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)

class Decoder_Block(nn.Module):
    def __init__(self,in_channels,out_channels,upsample,batch_norm=False):
        super(Decoder_Block,self).__init__()
        self.upsample = upsample
        self.batch_norm = batch_norm
        #reverse of encoder block without shortuct
        # self.conv1 = nn.ConvTranspose2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=4,stride=2 if self.upsample else 1,padding=1)
        self.conv2 = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv3 = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv4 = nn.ConvTranspose2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        if self.batch_norm:
            input = self.bn1(self.conv1(input))
            input = self.relu(input)
            input = self.bn2(self.conv2(input))
            input = self.relu(input)
        else:
            input = self.conv1(input)
            input = self.relu(input)
            input = self.conv2(input)
            input = self.relu(input)
        input = self.conv3(input)
        input = self.relu(input)
        input = self.conv4(input)
        input = self.relu(input)
        return input

# class Decoder_Block_With_Shortcut(nn.Module):
#     def __init__(self,in_channels,)

class Neural_Decoder(nn.Module):
    def __init__(self,batch_norm=False, decoder_block = Decoder_Block,latent_space_size=128):
        super(Neural_Decoder,self).__init__()
        self.batch_norm = batch_norm
        self.layer0 = nn.Linear(latent_space_size,512)
        self.UnFlatten = UnFlatten((512,1,1))
        self.inverse_pool = nn.ConvTranspose2d(512,512,kernel_size=4,stride=1,padding=0)
        self.layer1 =  nn.Sequential(
            Decoder_Block(512,256,upsample=True,batch_norm=self.batch_norm),
            Decoder_Block(256,128,upsample=True,batch_norm=self.batch_norm),
            )
        self.layer2 = nn.Sequential(
            Decoder_Block(128,64,upsample=True,batch_norm=self.batch_norm),
            Decoder_Block(64,32,upsample=True,batch_norm=self.batch_norm))

        self.final_layer = nn.Sequential(
            # nn.ConvTranspose2d(64,64,kernel_size=7,stride=2,padding=3),
            nn.ConvTranspose2d(32,1,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid())

    def forward(self,input):
        # print("decoding")
        input = self.layer0(input)
        #print(input.shape)
        input = self.UnFlatten(input)
        #print(input.shape)
        input = self.inverse_pool(input)
        #print(input.shape)
        input = self.layer1(input)
        #print(input.shape)
        input = self.layer2(input)
        #print(input.shape)
        input = self.final_layer(input)
        # #print(input.shape)
        return input
    
class NeuralVAE(CNNVAE):
    def __init__(self, image_channels=1, latentSpace=128, criterion = nn.MSELoss(),
                 kernel_sizes=[3,3,3,3,4],decoder_channels=[64,32,16,8,4,1],stride=[2,2,2,2,2],padding=[0,0,0,0,0],
                 sigma=1,
                 sigma_learnable=True,beta=1,beta_vae=False,activation_func=nn.Sigmoid(),generator=None,reshape_shape=(64,1,1),
                 kld_sigma=1):
        super(CNNVAE, self).__init__()
        
        self.encoder = NeuralCNNEncoder()

        self.fc1 = nn.Linear(512, latentSpace)
        self.fc2 = nn.Linear(512, latentSpace)
        # self.fc3 = nn.Linear(latentSpace, 128)

        decoder_layers = [
            nn.Linear(latentSpace,reshape_shape[0]),
            UnFlatten(reshape_shape)]
        
        assert reshape_shape[0] == decoder_channels[0], "reshape_shape[0] != decoder_channels[0]"
        assert len(kernel_sizes) == len(decoder_channels)-1, "len(kernel_sizes) != len(decoder_channels)-1"
        assert len(stride) == len(decoder_channels)-1, "len(stride) != len(decoder_channels)-1"
        assert len(padding) == len(decoder_channels)-1, "len(padding) != len(decoder_channels)-1"


        # for i in range(len(kernel_sizes)):
        #     decoder_layers.append(nn.Sequential(
        #         nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1], kernel_sizes[i], stride=stride[i], padding=padding[i]),
        #         #nn.BatchNorm2d(output_channels),
        #         nn.ReLU()
        #     ))

        # decoder_layers.append(activation_func)
        # self.decoder=nn.Sequential(*decoder_layers)
        
        self.decoder = Neural_Decoder(latent_space_size=latentSpace)
        self.beta=beta
        self.beta_vae=beta_vae
        self.sigma_learable=sigma_learnable
        if sigma_learnable:
            self.log_sigma=torch.nn.Parameter(torch.full((1,), np.log(sigma))[0],requires_grad=sigma_learnable)
        else:
            self.log_sigma=np.log(sigma)
        self.kld_sigma=kld_sigma
        self.generator=generator


if __name__ == "__main__":
    params = dict()
    VAE = NeuralVAE(**params)
    device = "cuda:0"
    # # VAE.to(device)
    # model = torch.compile(VAE)
    x = torch.randn(600, 1, 64, 64)
    print(VAE(x)[0].shape)
