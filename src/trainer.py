import sys
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import random
import torch
import torch.nn as nn
import numpy as np
import tqdm 
from src.dataloader import create_loader_multi_processing
torch.use_deterministic_algorithms(True)
from src.resnetVAEs import resnetVAE
from src.NeuralCNN_VAE import NeuralVAE
from src.meter import Meter, StatsMeter
from src.utils import get_gradient
from src.CustomLossFuncs import VGGPerceptualLoss
import torch.utils.data as data
from src.utils import to_cpu, to_numpy, pick_best_model
from src.beta_schedulers import exponential_beta_scheduler, linear_beta_scheduler
import src.param as param
import time
import json
from sklearn.metrics import precision_score
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from sklearn.mixture import GaussianMixture
import copy
class Trainer():
    def __init__(self, args):
        self.args = args
        print("initializing trainer")
        self.use_torch2 = args["use_torch2"]
        self.device = args["device"]
        self.lr = args["lr"]
        self.num_epochs = args["num_epochs"]   
        self.seed = args["seed"]    
        self.validation_freq = args["validation_freq"]
        self.res_folder = args["res_folder"]
        #model stuff
        self.model_name = args["model_name"]
        self.model_args=args["model_args"]
        #reduction algorithm stuff
        self.generator=torch.manual_seed(args["seed"])
        self.disable_tqdm=args["disable_tqdm"]
        self.beta_schedule=args["beta_schedule"]
        self.beta_learnable = args["beta_learnable"]
        self.beta_lr = args["beta_lr"]
        self.beta_start = args["beta_start"]
        try:
            beta_scheduler=args["beta_scheduler"]
        except:
            beta_scheduler="linear"
        #initialize beta scheduler
        if beta_scheduler=="linear":
            self.beta_scheduler=linear_beta_scheduler(0.01,1,self.num_epochs)
        elif beta_scheduler=="exponential":
            self.beta_scheduler=exponential_beta_scheduler(0,2,-100,self.num_epochs)
        else:
            assert False, f"beta scheduler {beta_scheduler} not implemented"
        np.random.seed(args["seed"])
        random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.criterion = self._init_criterion()
        self.criterion.eval()

        print("seeded")
        print("self.use_torch2", self.use_torch2)
        
    def _init_criterion(self):
        #return nn.MSELoss(reduction='none')
        if self.use_torch2:
            return VGGPerceptualLoss().to(self.device)
        return VGGPerceptualLoss().to(self.device)
    
    def _init_model(self):
        if self.model_name=="NeuralCNN_VAE":
            return NeuralVAE(generator=self.generator, **self.model_args).to(self.device)
        return resnetVAE(generator=self.generator, **self.model_args).to(self.device)
    def _init_optimzer(self, model):
        return torch.optim.AdamW(model.parameters(), self.lr, weight_decay=1e-5)
        
    def _init_dataloader(self, k, train = True, train_uniform_sample = True):
        return create_loader_multi_processing(k, args, train= train, train_uniform_sample= train_uniform_sample)
    
    def reconstruction_loss(self, imgs, out):
        return torch.mean(self.criterion(imgs,out).view(len(imgs), -1), -1)
    
    def loss_function(self, imgs, out, x_dist, mu ,beta=1, beta_learnable=False, beta_lr=0.001):
        reconstruct_loss=self.reconstruction_loss(imgs,out)
        # kld_loss = -0.5 * torch.sum(1 + logVar - mu.pow(2) - logVar.exp(), dim = 1)
        kld_loss = kl_divergence(x_dist, Normal(torch.zeros_like(mu), torch.ones_like(mu))).sum(1)
        if beta_learnable:
            new_beta = beta+beta_lr*(kld_loss.mean().item()-reconstruct_loss.mean().item())
            # print("beta:", beta, "new_beta:", new_beta, "kld_loss:", kld_loss.mean().item(), "reconstruct_loss:", reconstruct_loss.mean().item())
            #clip beta to be between 0 and 1
            new_beta = min(max(new_beta,0),1)
        else:
            new_beta = beta
            return [kld_loss, reconstruct_loss, beta*kld_loss+reconstruct_loss, new_beta]
        return [kld_loss, reconstruct_loss, beta*kld_loss+(1-beta)*reconstruct_loss, new_beta]

    def _train_once(self,model, optimizer, loader, beta):
        """trains the model over the dataset once"""
        model.train()
        meter = Meter()
        for _, (_, imgs, _, _, _, _,_, _) in enumerate(tqdm.tqdm(loader,disable=self.disable_tqdm), 0):
            optimizer.zero_grad()
            imgs = imgs.to(self.device).float().unsqueeze(1)
            out, mu, logVar, x_dist = model(imgs)
            kl_loss, recon_loss, loss, beta = self.loss_function(imgs, out, x_dist, mu ,beta, self.beta_learnable,self.beta_lr)
            loss_log = loss.clone().detach()
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step() 
            decoder_grad, encoder_grad = get_gradient(model)
            meter.add(to_numpy(loss_log), to_numpy(kl_loss), to_numpy(recon_loss),decoder_grad,encoder_grad)
        return meter.dump_wandb(), model, beta
    
    def inference(self, model, loader, dump_image_folder, get_meter = False, beta = 1):
        with torch.no_grad():
            model.eval()
            stats_meter = StatsMeter()
            for _, (pt_name, imgs, label, channel_names, starts, ends, detector_type, _) in enumerate(loader, 0):
                imgs = imgs.to(self.device).float().unsqueeze(1)
                out, mu, logVar, x_dist = model(imgs)
                kl_loss, recon_loss, _, _ = self.loss_function(imgs, out, x_dist, mu, beta = beta, beta_learnable=False)
                loss = kl_loss + recon_loss
                stats_meter.add(to_numpy(loss), to_numpy(kl_loss), to_numpy(recon_loss))
                stats_meter.add_inout(pt_name, channel_names, starts, ends, label, detector_type, to_numpy(mu), to_numpy(imgs), to_numpy(out))
        stats_meter.plot_results(30, dump_image_folder)
        if get_meter:
            return stats_meter
        else:
            stats = stats_meter.dump_wandb()
            mu = torch.from_numpy(stats_meter.get_mus())
            gmm = GaussianMixture(n_components=2, random_state=0, warm_start= True, init_params="kmeans", tol=0.0001, max_iter=2000)
            pred = gmm.fit_predict(mu)
            spike_labels = stats_meter.labels_[:,-1] > 0.5
            if precision_score(spike_labels, pred) < precision_score(spike_labels, (1-pred)):
                pred = 1-pred
            stats["patient_wise_precision"] = stats_meter.patient_wise_precision(pred)
            stats["precision"] = precision_score(spike_labels, pred)
            return stats
    

    def sample(self, loader, model, device, meter = None):
        with torch.no_grad():
            model.eval()   
            if meter is None:
                meter = StatsMeter()
            for _, (pt_name, imgs, label, channel_names, starts, ends, detector_type, _) in enumerate(loader, 0):
                imgs = imgs.to(device).float().unsqueeze(1)
                _, mu, _, _ = model.encode(imgs)
                imgs_reconstruct = model.decode(mu)
                reconstruct_loss = self.reconstruction_loss(imgs, imgs_reconstruct).cpu().numpy()
                meter.add_inout(pt_name, channel_names, starts, ends ,label,  detector_type, to_numpy(mu), reconstruct_loss= reconstruct_loss,
                                # in_=to_numpy(imgs), out_=to_numpy(imgs_reconstruct)
                                )
            return meter

    def classification(self, train_loader, valid_loader ,test_loader, model, save_dir = None,epoch = None,compile = True):  
        np.random.seed(args["seed"])
        random.seed(args["seed"])
        torch.manual_seed(args["seed"])
        torch.cuda.manual_seed(args["seed"])
        # print("model=",model)
        use_gmm = True
        if self.use_torch2 and compile:
            model = torch.compile(model)
        # print("model=",model)
        meter_val = self.sample(valid_loader, model, self.device)
        meter_val_copy = copy.deepcopy(meter_val)
        # if epoch is None:
        #     np.savez_compressed(os.path.join(save_dir, f"valid_.npz"), **meter_val.dump_csv())
        # else:
        #     np.savez_compressed(os.path.join(save_dir, f"valid_{epoch}.npz"), **meter_val.dump_csv())
        print("val done")
        meter_train = self.sample(train_loader, model, self.device, meter_val_copy)
        if epoch is None:
            np.savez_compressed(os.path.join(save_dir, f"train_.npz"), **meter_train.dump_csv())
        else:
            np.savez_compressed(os.path.join(save_dir, f"train_{epoch}.npz"), **meter_train.dump_csv())
        print("train done")
        # meter_val = self.sample(valid_loader, model, self.device, meter_train)
        meter_test = self.sample(test_loader, model, self.device)
        print("test done")

        # np.savez_compressed(os.path.join(save_dir, f"train_.npz"), **meter_train.dump_csv())
        if epoch is None:
            np.savez_compressed(os.path.join(save_dir, f"test_.npz"), **meter_test.dump_csv())
        else:
            np.savez_compressed(os.path.join(save_dir, f"test_{epoch}.npz"), **meter_test.dump_csv())
        # np.savez_compressed(os.path.join(save_dir, f"valid_.npz"), **meter_val.dump_csv())
        print("done")

    def train(self, train_loader, valid_loader, test_loader, save_dir = None):
        valid_stats = {}
        model_o = self._init_model()
        if self.use_torch2:
            model = torch.compile(model_o)
        else:
            model = model_o
        optimizer = self._init_optimzer(model)
        best_loss = 100000
        beta = self.beta_start
        for e in range(self.num_epochs + 1):
            start = time.time()
            if self.beta_schedule:
                beta = self.beta_scheduler(e)
            # # beta = 0
            train_stats, model, beta = self._train_once(model, optimizer, train_loader, beta) 
            #wandb.log(stats)
            stats = train_stats
            print(f"epoch {e} : time {time.time() - start:.3f}",end=" ")
            for key in stats.keys():
                print(f"{key} {stats[key]:.3f}", end=" ")
            print(f"beta {beta:.3f}") 
            if e % self.validation_freq == 0 and e != 0:
                valid_stats = self.inference(model, valid_loader, os.path.join(save_dir, "valid_imgs" ,str(e)),beta=beta)
                best_loss = pick_best_model(model_o, valid_stats, best_loss, self.args, e)
                print(f"eval {e} : time {time.time() - start:.3f} loss {valid_stats['v_loss']:.3f} kl {valid_stats['v_kl']:.3f} recons {valid_stats['v_recons']:.3f} precision {valid_stats['precision']:.3f} patient_wise_precision {valid_stats['patient_wise_precision']:.3f}")
                stats = {**train_stats, **valid_stats}
            print("--"*20)
    def onefold_crossvalidation(self, k):
        train_loader, valid_loader = self._init_dataloader(k, train = True)
        test_loader = self._init_dataloader(k, train = False, train_uniform_sample=False)
        fold_res_folder = os.path.join(self.res_folder, f"fold_{k}")
        os.makedirs(fold_res_folder, exist_ok=True)
        self.args["checkpoint_folder"] = os.path.join(fold_res_folder, "ckpt")
        os.makedirs(self.args["checkpoint_folder"], exist_ok=True)
        self.train(train_loader, valid_loader, test_loader,save_dir = fold_res_folder)
        
    def onefold_classification(self, k,epoch=None):
        train_loader, valid_loader = self._init_dataloader(k, train = True, train_uniform_sample=False)
        fold_res_folder = os.path.join(self.res_folder, f"fold_{k}")
        model = self._init_model()
        #print(torch.load(os.path.join(fold_res_folder,"ckpt" ,"model_46.pth"), map_location=self.device)["state_dict"].keys())
        checkpont_folders = {
            4:"best", 
            3:"best", 
            2:"best",
            1:"best",
            0:"best"
        }
        if epoch is not None:
            checkpont_folders[k] = epoch
        else:
            checkpont_folders[k] = "81"
        model.load_state_dict(torch.load(os.path.join(fold_res_folder,"ckpt" ,f"model_{checkpont_folders[k]}.pth"), map_location=self.device)["state_dict"], strict=False)
        test_loader = self._init_dataloader(k, train = False, train_uniform_sample=False)
        self.classification(train_loader, valid_loader, test_loader, model, save_dir = fold_res_folder,epoch=checkpont_folders[k])

    def reconstruction(self, loader, model_path, save_dir):
        model = self._init_model()
        model.load_state_dict(torch.load(model_path, map_location=self.device)["state_dict"])
        stats = self.inference(model, loader, save_dir, get_meter=True)
        return stats


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    print("here")
    fold_num = int(sys.argv[1])
    device = sys.argv[2]
    suffix = sys.argv[3]
    K = int(sys.argv[4])
    print("K:", K)
    if len(sys.argv) > 6 and sys.argv[5] == "test":
        epoch = int(sys.argv[6])
        if len(sys.argv) > 7:
            save_dir = sys.argv[7]
        else:
            save_dir = None    
    else:
        path_to_model = None
        epoch = None
        save_dir = None
    device_num = int(device.split(":")[-1])
    print("device:", device, "device_num:", device_num)
    torch20_capable = False 
    print(f"cuda:{device_num} is torch20_capable: {torch20_capable}")
    args = param.get_args()
    args["use_torch2"] = torch20_capable
    args["K"] = K
    args["device"] = device
    args["res_folder"] = os.path.join(args["res_folder"], suffix)
    trainer = Trainer(args)
    print("here")
    if sys.argv[5] == "train":
        os.makedirs(args["res_folder"], exist_ok=True)
        with open(os.path.join(args["res_folder"], "args.json"), "w") as f:
            json.dump(args, f, indent=4)
        print("initializing wandb")
        trainer.onefold_crossvalidation(fold_num)
    elif sys.argv[5] == "test":
        #if there is a 6th argument, then it is the epoch
        if epoch is None:
            trainer.onefold_classification(fold_num)
        else:
            trainer.onefold_classification(fold_num, epoch)
    print("done")
