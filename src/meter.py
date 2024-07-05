import os
import numpy as np
from sklearn.metrics import precision_score
import torch
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def plot_image(fig, ax, im):
    im = np.squeeze(im)
    im =ax.imshow(im)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

def plot_inout(im_, out_, save_fn):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_image(fig, ax1, im_)
    ax1.set_title('input')
    plot_image(fig, ax2, out_)
    ax2.set_title('output')
    fig.suptitle(f'maxOutputVal: {np.max(out_)}')
    plt.savefig(save_fn)
    plt.close()

class Meter():
    def __init__(self):
        self.loss_ = []
        self.kl_ = []
        self.reconstruct_ = []
        self.decoder_grads = None
        self.encoder_grad = None
        self.n = 0
    def copy(self):
        new_meter = Meter()
        for attr in self.__dict__:
            #if attr is none
            if getattr(self, attr) is None:
                continue
            if hasattr(getattr(self, attr), "copy"):
                print(attr, "of type:", type(getattr(self, attr)), "has copy method")
                setattr(new_meter, attr, getattr(self, attr).copy())
            else:
                setattr(new_meter, attr, getattr(self, attr))
        return new_meter
    def add(self, loss, kl, reconstruct,deocder_grad=None, encoder_grad=None):
        self.loss_.append(loss)
        self.kl_.append(kl)
        self.reconstruct_.append(reconstruct)
        if deocder_grad is not None:
            if self.decoder_grads is None:
                self.decoder_grads = deocder_grad
            else:
                self.decoder_grads += deocder_grad
        if encoder_grad is not None:
            if self.encoder_grad is None:
                self.encoder_grad = encoder_grad
            else:
                self.encoder_grad += encoder_grad
            self.n += 1

    def _concate(self):
        self.loss_ = np.concatenate(self.loss_, 0)
        self.kl_ = np.concatenate(self.kl_, 0)
        self.reconstruct_ = np.concatenate(self.reconstruct_, 0)
        
    def dump(self):
        self._concate()
        if  self.decoder_grads is None:
            return np.mean(self.loss_), np.mean(self.kl_), np.mean(self.reconstruct_)
        decoder_grad = torch.linalg.norm(self.decoder_grads/self.n)
        encoder_grad = torch.linalg.norm(self.encoder_grad/self.n)
        return np.mean(self.loss_), np.mean(self.kl_), np.mean(self.reconstruct_), decoder_grad, encoder_grad
        
    def dump_wandb(self):
        if self.decoder_grads is None:
            loss, kl, reconstruct = self.dump()
            res = {"v_loss": loss, "v_kl": kl, "v_recons": reconstruct}
        loss, kl, reconstruct,decoder_grad_norm,encoder_grad_norm = self.dump()
        res = {"train_loss": loss, "train_kl": kl, "train_recons": reconstruct, "train_decoder_grad_norm": decoder_grad_norm, "train_encoder_grad_norm": encoder_grad_norm}
        return res
       
class StatsMeter():
    def __init__(self, type="validation"):
        self.meter_ = Meter()
        self.out_ = []
        self.in_ = []
        self.pt_names_ = []
        self.channel_names_ = []
        self.starts_ = []
        self.ends_ = []
        self.mu_ = []
        self.type = type
        self.pred = []
        self.pred_prob = []
        self.labels_ = []
        self.detector_ = []
    
    def copy(self):
        new_meter = StatsMeter(self.type)
        for attr in self.__dict__:
            if getattr(self, attr) is None:
                continue
            #if attribute has a copy method
            if hasattr(getattr(self, attr), "copy"):
                print(attr, "of type:", type(getattr(self, attr)), "has copy method")
                setattr(new_meter, attr, getattr(self, attr).copy())
            else:
                setattr(new_meter, attr, getattr(self, attr))
        return new_meter


    def add(self, loss, kl, reconstruct):
        self.meter_.add(loss, kl, reconstruct)
    def add_inout(self, pt_names_,channel_names_, starts_, ends_, labels_, detectors_, mu_, in_ = None, out_ = None, reconstruct_loss = None):
        if out_ is not None:    
            self.out_.append(np.transpose(out_, (0, 2, 3, 1)))
        if in_ is not None: 
            self.in_.append(np.transpose(in_, (0, 2, 3, 1)))
        if reconstruct_loss is not None:
            self.meter_.add(reconstruct_loss, 0, reconstruct_loss)
        self.pt_names_.append(pt_names_)
        self.channel_names_.append(channel_names_)
        self.starts_.append(starts_)
        self.ends_.append(ends_)
        self.mu_.append(mu_)
        self.labels_.append(labels_)
        self.detector_.append(detectors_)
    def add_pred(self, pred):
        self.pred.append(pred)
    def add_pred_prob(self, pred_prob):
        self.pred_prob.append(pred_prob)
    def _concate(self):
        self.in_ = np.concatenate(self.in_, 0) if len(self.in_) > 0 else self.in_
        self.out_ = np.concatenate(self.out_, 0) if len(self.out_) > 0 else self.out_
        self.pt_names_ = np.concatenate(self.pt_names_, 0) if len(self.pt_names_) > 0 else self.pt_names_
        self.channel_names_ = np.concatenate(self.channel_names_, 0) if len(self.channel_names_) > 0 else self.channel_names_
        self.starts_ = np.concatenate(self.starts_, 0) if len(self.starts_) > 0 else self.starts_
        self.ends_ = np.concatenate(self.ends_, 0) if len(self.ends_) > 0 else self.ends_
        self.labels_ = np.concatenate(self.labels_, 0) if len(self.labels_) > 0 else self.labels_
        self.detector_ = np.concatenate(self.detector_, 0) if len(self.detector_) > 0 else self.detector_
        self.mu_ = np.concatenate(self.mu_, 0) if len(self.mu_) > 0 else self.mu_
        
    def dump_wandb(self):
        loss, kl, reconstruct = self.meter_.dump()
        if self.type == "validation":
            res = {"v_loss": loss, "v_kl": kl, "v_recons": reconstruct} 
        else:
            res = {"t_loss": loss, "t_kl": kl, "t_recons": reconstruct}
        return res

    def dump_loss(self):
        return np.mean(np.concatenate(self.meter_.loss_, 0))
    
    def patient_wise_precision(self,preds):
        if type(self.pt_names_) == list:
            self._concate()
        pt_names = np.unique(self.pt_names_)
        precisions = []
        spike_labels = self.labels_[:, -1] > 0.5
        for pt_name in pt_names:
            index = np.where(self.pt_names_ == pt_name)[0]
            precisions.append(precision_score(spike_labels[index], preds[index]))
        return np.mean(precisions)

    def dump_image(self, num_samples):
        self._concate()
        index = np.random.choice(np.arange(len(self.in_)), num_samples, replace=False)
        return self.in_[index], self.out_[index], self.pt_names_[index], self.channel_names_[index], self.starts_[index], self.ends_[index]
    
    def plot_results(self, num_samples, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        in_, out_, pt_names_, channel_names_, starts, ends = self.dump_image(num_samples)
        for idx in range(len(in_)):
            plot_inout(in_[idx], out_[idx], save_dir + "/" + f"{pt_names_[idx]}_{channel_names_[idx]}_{starts[idx]}_{ends[idx]}.jpg")
    
    def get_mus(self):
        if type(self.pt_names_) == list:
            self._concate()
        return self.mu_

    def sample_uniform(self, num_samples):
        # sample uniformly from each patient or the max number of samples
        self._concate()
        pt_names = np.unique(self.pt_names_)
        # print(pt_names)
        index = []
        
        for pt_name in pt_names:
            # sample uniformly from each patient or the max number of samples
            index.append(np.random.choice(np.where(self.pt_names_ == pt_name)[0], min(num_samples, len(np.where(self.pt_names_ == pt_name)[0])), replace=False))
        index = np.concatenate(index, 0)
        label = self.labels_[index]
        mu = self.mu_[index]
        # select all variables 
        self.pt_names_ = self.pt_names_[index]
        self.channel_names_ = self.channel_names_[index]
        self.starts_ = self.starts_[index]
        self.ends_ = self.ends_[index]
        self.labels_ = self.labels_[index]
        self.detector_ = self.detector_[index]
        self.mu_ = self.mu_[index]
        
        # filter artifact
        # index = np.where(label[:, 3] > 0.5)[0]
        # return mu[index], label[index, -1] > 0.5
        return mu, label[:, -1] > 0.5

    def dump_cluster(self):
        if type(self.pt_names_) == list:
            self._concate()
        pred = np.concatenate(self.pred, 0)
        pred_prob = np.concatenate(self.pred_prob, 0)
        reconstruct_ = np.concatenate(self.meter_.reconstruct_, 0)
        res = {
            "pt_names": self.pt_names_,
            "channel_names": self.channel_names_,
            "starts": self.starts_,
            "ends": self.ends_,
            "labels": self.labels_,
            "detector": self.detector_,
            "mu": self.mu_,
            "pred": pred,
            "pred_prob": pred_prob,
            "reconstruct": reconstruct_,
        }
        if len(self.in_) > 0:
            res["in"] = self.in_
            res["out"] = self.out_
        return res

    def dump_csv(self):
        if type(self.pt_names_) == list:
            self._concate()
        reconstruct_ = np.concatenate(self.meter_.reconstruct_, 0)
        res = {
            "pt_names": self.pt_names_,
            "channel_names": self.channel_names_,
            "starts": self.starts_,
            "ends": self.ends_,
            "labels": self.labels_,
            "detector": self.detector_,
            "mu": self.mu_,
            "detector": self.detector_,
            "reconstruct": reconstruct_
        }
        return res
            
