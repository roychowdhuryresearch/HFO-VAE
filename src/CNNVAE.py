import torch
from torch import Tensor
#from models import BaseVAE
from torch import nn
from torch.nn import functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import numpy as np

class UnFlatten(nn.Module):
    def forward(self, input, size=1024):
        print("UnFlatten")
        print(input.shape)
        print(input.view(input.size(0), size, 1, 1).shape)
        return input.view(input.size(0), size, 1, 1)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
def softclip(tensor, min):
    """ Clips the tensor values at the minimum value max in a softway. Taken from Handful of Trials """
    result_tensor = min - F.softplus(tensor - min)

    return result_tensor

class CNNVAE(nn.Module):    
    def __init__(self, image_channels=1, h_dim=1024, z_dim=32, criterion = nn.MSELoss(),sigma=1,generator=None,beta=1,beta_vae=False,sigma_learable=False,
                 kld_sigma=1):
        super(CNNVAE, self).__init__()
        self.criterion=criterion
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        )
        
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)
        
        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )
        self.log_sigma=torch.nn.Parameter(torch.full((1,), np.log(sigma))[0])
        print("log_sigma",self.log_sigma)
        self.generator=generator
        self.beta=beta
        self.beta_vae=beta_vae
        self.sigma_learable=sigma_learable
        self.kld_sigma=kld_sigma


    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc1(h), self.fc2(h)
        x_dist = Normal(mu, logvar.mul(.5).exp())
        z = x_dist.rsample()
        return z, mu, logvar, x_dist

    def decode(self, z):
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar,x_dist  = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar, x_dist
    
    def get_sigma(self):
        if not self.sigma_learable:
            return np.exp(self.log_sigma)
        return self.log_sigma.detach().cpu().exp().item()
    
if __name__ == "__main__":
    test_net=CNNVAE()
    for name, param in test_net.named_parameters():
        if param.requires_grad:
            print(name)
            
    print(test_net.get_sigma())