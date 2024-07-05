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
from cluster import overall_clustering as cluster
from CustomLossFuncs import VGGPerceptualLoss
import seaborn as sns


def get_points(path,dims_of_interest = np.arange(8)):
    data = np.load(path)
    predictions = data["pred_with_artifact"]
    mu = data["mu"]
    reconstruction = data['reconstruct']
    keep_index = data["pred_with_artifact"]!=-1
    reconstruction = reconstruction[keep_index]
    predictions = predictions[keep_index]
    mu = mu[keep_index]
    labels = data["labels"]
    out = {}
    for p in np.unique(predictions):
        mu_p = mu[predictions==p]
        example_mu = np.mean(mu_p,axis=0)
        out[p] = mu_p
    ranges = {}
    for dim in dims_of_interest:
        # left_bound = 75% and right bound = 25% of the data
        left_bound = np.quantile(mu[:,dim],0.99)
        right_bound = np.quantile(mu[:,dim],0.01)
        ranges[dim] = [left_bound,right_bound]
    return out,ranges, mu,reconstruction.reshape(-1,1), predictions
    # return out

def reconstruction_loss(imgs, out, criterion):
    return torch.mean(criterion(imgs,out).view(len(imgs), -1), -1).detach().cpu().numpy()

def filter_data(all_mu, reconstruction,cluster_algo):
    all_mu = np.concatenate([all_mu,reconstruction.reshape(-1,1)],axis=-1)
    predict = cluster_algo.pathological_predictor.cluster_algo.predict_proba(all_mu)[:,0]
    out = {
        0: all_mu[:,:-1][predict<0.01],
        1: all_mu[:,:-1][predict>0.99]
    }
    reconstruction = {
        0: reconstruction[predict<0.01],
        1: reconstruction[predict>0.99]
    }
    return out, reconstruction

@torch.no_grad()
def perturbe(path,n=10):
    fold = int(path.split("/")[-3].split("_")[-1])
    out,ranges, all_mu, reconstruction, predictions = get_points(path)

    cluster_algo = create_cluster(date, fold)
    out, reconstruction  = filter_data(all_mu, reconstruction, cluster_algo)

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
    ckpt_path = f"res/{date}/fold_{fold}/ckpt/model_{checkpoint}.pth"
    print("loading from :", ckpt_path)
    model.load_state_dict(torch.load(ckpt_path,map_location="cpu")["state_dict"])
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    
    save_path = f"res/{date}/{path.split('/')[-2]}/perturbed1/"
    os.makedirs(save_path, exist_ok=True)
    dev = torch.device(device)
    criterion = VGGPerceptualLoss().to(dev)
    criterion.eval()
    
    # 8 dimensions
    pos = out[0]
    neg = out[1]
    all_in,latents, decodes,outputs, mu_ori, recon = [],[], [],[],[], []
    for i in range(1000):
        for dim in ranges.keys():
            dim = 1
            min_,max_ = ranges[dim]
            # sample one from all mu 
            if i < 500:
                random_index = np.random.choice(len(neg))
                mu = neg[random_index]
                # concate mu with the reconstruction
                oo = np.concatenate([mu,reconstruction[1][random_index]],axis=-1)
                mu_ori.append(oo.reshape(1,-1))
            else:
                random_index = np.random.choice(len(pos))
                mu = pos[random_index]
                oo = np.concatenate([mu,reconstruction[0][random_index]],axis=-1)
                mu_ori.append(oo.reshape(1,-1))

            linspace = mu.reshape(1,-1).repeat(n+1,axis=0)
            mid_val = np.mean(all_mu[:,dim])
            linspace[:5,dim] = np.linspace(min_,mid_val,n//2)
            linspace[5:,dim] = np.linspace(mid_val,max_,n//2+1)
            
            input = torch.tensor(linspace).float()
            decode = model.decode(input.to(dev))
            recon.append(model.decode(torch.from_numpy(mu).reshape(1,-1).to(dev)).detach().cpu().numpy())

            min_val = decode.min(-1,keepdims=True)[0].min(-2,keepdims=True)[0]
            max_val = decode.max(-1,keepdims=True)[0].max(-2,keepdims=True)[0]
            decode = (decode-min_val)/(max_val-min_val)

            out, latent, _, _ = model(decode)
            #out = model.decode(latent)
            latent = latent.detach().cpu().numpy()
            decodes.append(decode.detach().cpu().numpy())
            outputs.append(out.detach().cpu().numpy())
            imgs_lin = decode.detach().cpu().numpy().squeeze()
            all_in.append(imgs_lin)
            latents.append(latent)
            break
    recon = np.concatenate(recon,axis=0)
    outputs = torch.from_numpy(np.concatenate(outputs,axis=0))
    decodes = torch.from_numpy(np.concatenate(decodes,axis=0))
    mu_ori = np.concatenate(mu_ori,axis=0)
    loss_all = []
    for i in range(0, len(outputs), 100):
        end = min(i+100,len(outputs))
        loss = reconstruction_loss(outputs[i:end].to(dev)
                                   ,decodes[i:end].to(dev)
                                   ,criterion)
        loss_all.append(loss)
    loss = np.concatenate(loss_all,axis=0)
    #loss = criterion(decodes.to(dev),outputs.to(dev)).detach().cpu().numpy()
    latents = np.concatenate(latents,axis=0)
    latents = np.concatenate([latents,loss.reshape(-1,1)],axis=-1)
    #latents[:,-1] = 0
   
    predict = cluster_algo.pathological_predictor.cluster_algo.predict_proba(latents)[:,0]
    ori_predict = cluster_algo.pathological_predictor.cluster_algo.predict_proba(mu_ori)[:,0]
    print(np.histogram(ori_predict))
    predict = predict.reshape(1000,-1)
    all_in = np.array(all_in) # (100,11,50,20)
    all_in_c = all_in[:,:,1:48,22:42]
    max_value = np.mean(all_in_c,axis=-1) # (100,11,50,-1)
    max_index = np.argmax(max_value,axis=-1)
    fig,axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].boxplot(max_index)
    #sns.boxplot(data=predict,ax=axs[1])
    # plot the mean of each prediction
    mean = np.mean(predict,axis=0)
    sns.boxplot(data=predict,ax=axs[1],palette="Set2")
    plt.savefig(f"{save_path}/max_value.png")
    # save the data
    np.savez(f"{save_path}/perturbed1.npz", max_index=max_index, predict=predict, all_in=all_in, recon=recon)


def create_cluster(date, fold):
    n_sample_artifact = 10000
    n_sample_pathologic = 2000
    epoch = 81
    fn = f"res/{date}/fold_{fold}/train_{epoch}.npz"
    artifact_predictor_kwargs = {"cluster_algo": "gmm", "n_clusters": 2, "seed": 42, "num_samples":n_sample_artifact}
    pathological_predictor_kwargs = {"cluster_algo": "gmm", "n_clusters": 2, "seed": 42, "num_samples": n_sample_pathologic}
    algo = cluster(fn, artifact_rejection_method="cluster", artifact_predictor_kwargs=artifact_predictor_kwargs, pathological_predictor_kwargs=pathological_predictor_kwargs)
    algo()
    return algo

if __name__ == "__main__":
    import glob
    date = sys.argv[1]
    suffix = sys.argv[2]
    device = sys.argv[3]

    path = f"res/{date}/fold_1/{suffix}/train_overall_.npz"
    perturbe(path)