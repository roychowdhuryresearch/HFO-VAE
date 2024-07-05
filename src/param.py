import torch.nn as nn
import sklearn.decomposition
import sklearn.manifold

class exponential_beta_scheduler:
    def __init__(self, beta_start, beta_end, time_constant=300, num_epochs=300):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.time_constant = time_constant
        self.num_epochs = num_epochs
    def __call__(self, epoch):
        return self.beta_start + (self.beta_end - self.beta_start) * (1 - np.exp(-epoch / self.time_constant))/(1-np.exp(-self.num_epochs/self.time_constant))

args = {
"data_dir": "data",
"meta_fn":"data/meta.csv",
"device": "cuda:2",
"lr": 3e-4,
"use_torch2":False,
"num_epochs": 80,
"batch_size": 512, 
"seed": 4,
"validation_freq":10,
"loss_name":"vggloss",
"model_name":"resnet18VAE",
"reduction_args":{"n_components":2},
"disable_tqdm":True,
"res_folder":"res",
"K":5,
"dataset_size":2500, 
"time_augmentation":True,
"beta_schedule":False,
"beta_scheduler":"linear",
"beta_learnable":True,
"beta_lr":1e-4,
"beta_start":0.5,
}
def get_args():
    """
    adds the model args
    """
    if args["model_name"]=="resnet18VAE":
        print("resnet 18 VAE model")
        args["model_args"]=dict(resent = "thinResnet",
                                latentSpace=8, 
                                decoder_channels=[256]*2+[128]*2+[64]*2+[32]*2,
                                kernel_sizes=[4]+[4,3,]*4,
                                stride=[1]+4*[2,1],
                                padding=[0]+12*[1,1,],
                                reshape_shape=(512,1,1),
                                beta_vae=True,
                                beta=0.1,
                                sigma_learnable=False, 
                                kld_sigma = 1,
                                batch_norm=True,
                                )
    if args["model_name"]=="NeuralCNN_VAE":
        print("NeuralCNN_VAE model")
        args["model_args"]= dict(
                                # decoder_channels=[512]+[256]*4+[128]*4+[64]*4+[32]*4,
                                # kernel_sizes=[4]+[4,3,3,3]*4,
                                # stride=[1]+4*[2,1,1,1],
                                # padding=[0]+12*[1,1,1,1],
                                # reshape_shape=(512,1,1)
                                )
    return args
