import torch
from torch import Tensor
#from models import BaseVAE
from torch import nn
from torch.nn import functional as F

class Decoder_Block_With_Shortcut(nn.Module):
    def __init__(self,in_channels=1, out_channels=1, stride=1,):
        super(Decoder_Block_With_Shortcut, self).__init__()
        if stride==1:
            self.conv_path = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                                        nn.ReLU(),
                                            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),)
        else:
            self.conv_path = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3+(stride-1), stride=stride, padding=1, bias=False),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                        # nn.Upsample(scale_factor=stride, mode='nearest'),
                        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=stride, stride=stride, padding=0, bias=False))
            #self.shortcut=nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False,output_padding=stride-1)
        else:
            self.shortcut = nn.Sequential()
        self.non_linearity = nn.ReLU()
    
    def forward(self, x):
        return self.non_linearity(self.conv_path(x) + self.shortcut(x))

if __name__=="__main__":
    test_img=torch.rand([1,1,8,8])
    test_block=Decoder_Block_With_Shortcut(1,1)
    print(test_block(test_img).shape)
    test_block=Decoder_Block_With_Shortcut(1,1,stride=2)
    print(test_block(test_img).shape)
