import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from resnetVAEs import resnetVAE
import torch
import os
import itertools
import numpy as np
import matplotlib.pyplot as plt


def get_points(path,dims_of_interest = np.arange(8)):
    data = np.load(path)
    predictions = data["pred_with_artifact"]
    mu = data["mu"]
    reconstruction = data['reconstruct']
    predictions = predictions[predictions!=-1]
    mu = mu[data["pred_with_artifact"]!=-1]
    out = {}
    for p in np.unique(predictions):
        mu_p = mu[predictions==p]
        example_mu = np.mean(mu_p,axis=0)
        out[p] = example_mu
    ranges = {}
    for dim in dims_of_interest:
        # left_bound = 75% and right bound = 25% of the data
        left_bound = np.quantile(mu[:,dim],0.99)
        right_bound = np.quantile(mu[:,dim],0.01)
        ranges[dim] = [left_bound,right_bound]
    return out,ranges
    # return out

def perturbe(path,n=10):
    fold = int(path.split("/")[-3].split("_")[-1])
    out,ranges = get_points(path)
    checkpoint = int(path.split("/")[-2].split("_")[-1])
    
    model = resnetVAE(**dict(resent = "thinResnet",
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
                                ))
    print("loading from :",f"res/{date}/fold_{fold}/ckpt/model_{checkpoint}.pth")
    model.load_state_dict(torch.load(f"res/{date}/fold_{fold}/ckpt/model_{checkpoint}.pth")["state_dict"])
    model.to("cpu")
    model.eval()
    
    
    save_path = f"res/{date}/{path.split('/')[-2]}/perturbed/"
    pathological, physiological = [],[]
    os.makedirs(save_path, exist_ok=True)
    for dim in ranges.keys():
        min_,max_ = ranges[dim]
        fig,axs = plt.subplots(len(out.keys()),n, figsize=(5*n,5*len(out.keys())))
        pa, ph = [], []
        for i,key in enumerate(out.keys()):
            mu = out[key]
            #linspace
            linspace = mu.reshape(1,-1).repeat(n+1,axis=0) 
            mid_val = np.mean(linspace[:,dim])
            linspace[:5,dim] = np.linspace(min_,mid_val,n//2)
            linspace[5:,dim] = np.linspace(mid_val,max_,n//2+1)
            print(linspace.shape)
            # print(linspace)
            print("min",np.min(linspace[:,dim])),print("max",np.max(linspace[:,dim])),print("mu[:,dim],", linspace[:,dim])
            imgs_lin = model.decode(torch.tensor(linspace).float()).detach().cpu().numpy()
            
            for j in range(n):
                axs[i,j].imshow(imgs_lin[j][0])
            axs[i,0].set_ylabel({-1:"artifact",0:"non-spkHFO",1:"spkHFO"}[key])
            if key == 1:
                pa.append(imgs_lin)
            if key == 0:
                ph.append(imgs_lin)
        pathological.append(np.concatenate(pa,axis=0))
        physiological.append(np.concatenate(ph,axis=0))
        fig.suptitle(f"Dimension {dim}")
        plt.savefig(f"{save_path}/fold{fold}_dim{dim}.png")
    np.savez_compressed(f"{save_path}/fold{fold}.npz",pathological = np.array(pathological), physiological = np.array(physiological))

if __name__ == "__main__":    
    import glob
    date = sys.argv[1]
    suffix = sys.argv[2]
    device = sys.argv[3]

    for path in glob.glob(f"res/{date}/fold_*/{suffix}/train_overall_.npz"):
        print(path)
        perturbe(path)