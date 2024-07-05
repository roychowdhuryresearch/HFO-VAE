import torch
import torch.nn as nn
import torchvision

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[:4].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[4:9].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[9:16].eval())
        blocks.append(torchvision.models.vgg16(weights='DEFAULT').features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            #print(y.shape)
            if i in feature_layers:
                loss += torch.mean(torch.nn.functional.mse_loss(x, y,reduction="none"),dim=[1,2,3])
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.mean(torch.nn.functional.mse_loss(gram_x, gram_y,reduction="none"),dim=[1,2,3])
        return loss
    
class VGGand(nn.Module):
    def __init__(self, other_loss=nn.BCELoss(reduction='none'),other_loss_weight=0.25,thresh=0.5):
        super(VGGand, self).__init__()
        self.other_loss=other_loss
        self.other_loss_weight=other_loss_weight
        self.thresh=thresh
        self.VGGLoss=VGGPerceptualLoss()
        self.VGG_weight=(1-other_loss_weight)
        
    def forward(self,output,target):
        other_loss=((output>self.thresh)*self.other_loss(output, target)).view(len(output), -1).T
        VGGLoss=self.VGGLoss(output, target)
        #print(f"BCELoss.shape={BCELoss.shape}|VGGLoss.shape={VGGLoss.shape}")
        return (self.other_loss_weight*other_loss + self.VGG_weight*VGGLoss).T
    
class VGGandBCE(nn.Module):
    def __init__(self, BCE_weight=0.75,VGG_weight=0.25):
        super(VGGandBCE, self).__init__()
        self.VGGLoss=VGGPerceptualLoss()
        self.BCELoss=nn.BCELoss(reduction='none')
        self.BCE_weight=BCE_weight
        self.VGG_weight=VGG_weight
    
    def forward(self,output,target):
        BCELoss=self.BCELoss(output, target).view(len(output), -1).T
        VGGLoss=self.VGGLoss(output, target)
        #print(f"BCELoss.shape={BCELoss.shape}|VGGLoss.shape={VGGLoss.shape}")
        return (self.BCE_weight*BCELoss + self.VGG_weight*VGGLoss).T
        