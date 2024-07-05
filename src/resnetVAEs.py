import sys
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import torch
from torch import Tensor
#from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from src.resnetDecoderUtils import Decoder_Block_With_Shortcut
from src.resnetUtils import resnetEncoder18, thinResnetEncoder
from src.CNNVAE import CNNVAE

class UnFlatten(nn.Module):
    def __init__(self, shape):
        super(UnFlatten, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(input.size(0), *self.shape)

class decoderBlock(nn.Module):
    def __init__(self, input_channels,output_channels,kernel_size,stride=2, padding=0, relu= True,batch_norm=True):
        super(decoderBlock, self).__init__()
        self.kernel_size= kernel_size
        self.stride=stride
        if relu:
            if batch_norm:
                self.module=nn.Sequential(
                    nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride=stride,padding=padding),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU()
                )
            else:
                self.module=nn.Sequential(
                    nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride=stride,padding=padding),
                    # nn.BatchNorm2d(output_channels),
                    nn.ReLU()
                )
        else:
            self.module=nn.Sequential(
                nn.ConvTranspose2d(input_channels,output_channels,kernel_size,stride=stride,padding=padding),
                #nn.BatchNorm2d(output_channels),
                #nn.ReLU()
            )
    def forward(self,x):
        return self.module(x)

class resnetVAE(CNNVAE):
    def __init__(self,resent="resnet18", image_channels=1, latentSpace=64, criterion = nn.MSELoss(),
                 kernel_sizes=[3,3,3,3,4],decoder_channels=[128,64,32,16],stride=[2,2,2,2,2],padding=[1,1,1,1,1],
                 sigma=1,sigma_learnable=True,beta=1,beta_vae=False,activation_func=nn.Sigmoid(),generator=None,reshape_shape=(64,1,1),
                 decoder_block_type="decoderBlock",kld_sigma=1,batch_norm=True):
        super(CNNVAE, self).__init__()
        
        self.criterion=criterion
        
        if resent=="resnet18":
            self.encoder=resnetEncoder18()
        elif resent=="thinResnet":
            self.encoder=thinResnetEncoder()
        
        self.fc1 = nn.Linear(self.encoder.output_size, latentSpace)
        self.fc2 = nn.Linear(self.encoder.output_size, latentSpace)

        decoder_modules=[nn.Linear(latentSpace,reshape_shape[0]),
            UnFlatten(reshape_shape)]
        input_channels=reshape_shape[0]
        if decoder_block_type=="decoderBlock":
            for i in range(len(decoder_channels)):
                decoder_modules.append(decoderBlock(input_channels=input_channels,
                                                    output_channels=decoder_channels[i],
                                                    kernel_size=kernel_sizes[i],
                                                    padding=padding[i],
                                                    stride=stride[i],batch_norm=batch_norm))
                input_channels=decoder_channels[i]
            decoder_modules.append(decoderBlock(input_channels,image_channels,
                                            kernel_size=kernel_sizes[-1], stride=stride[-1],padding=padding[-1], relu=False))

        elif decoder_block_type=="decoderBlockWithShortcut":
            decoder_modules.append(nn.ConvTranspose2d(input_channels,input_channels,-reshape_shape[-1]+1+64//np.prod(stride)))
            for i in range(len(decoder_channels)):
                decoder_modules.append(Decoder_Block_With_Shortcut(in_channels=input_channels,
                                                    out_channels=decoder_channels[i],
                                                    stride=stride[i]))
                input_channels=decoder_channels[i]
            decoder_modules.append(nn.ConvTranspose2d(input_channels,image_channels,kernel_size=3,padding=1))

        decoder_modules.append(activation_func)
        self.decoder=nn.Sequential(*decoder_modules)
        self.beta=beta
        self.beta_vae=beta_vae
        self.sigma_learable=sigma_learnable
        if sigma_learnable:
            self.log_sigma=torch.nn.Parameter(torch.full((1,), np.log(sigma))[0],requires_grad=sigma_learnable)
        else:
            self.log_sigma=np.log(sigma)
        self.generator=generator
        self.kld_sigma=kld_sigma
        

if __name__=="__main__":
    test_img=torch.rand([1,1,64,64])
    model=resnetVAE(decoder_channels=[512,512,256,256,128,128,64,64],
                                stride=[2,1,2,1,2,1,1,1],
                                reshape_shape=(512,1,1),
                                decoder_block_type="decoderBlockWithShortcut")
    print(model(test_img)[0].shape)
        
        

