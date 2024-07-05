from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import os, shutil
import sys
if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(current)
    sys.path.append(parent)
import torch
from src.meter import Meter, StatsMeter


def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)

def pick_best_model(model, v_stats, best_loss, param, epoch):
    v_loss = v_stats["v_loss"]
    save_checkpoint(
            {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            "param": param,
            "v_stats": v_stats,
            },
            filename= os.path.join(param["checkpoint_folder"], f'model_{epoch + 1}.pth'))
    if v_loss < best_loss:
        best_loss = v_loss 
        save_checkpoint(
                {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                "param": param,
                "v_stats": v_stats,
                },
                filename= os.path.join(param["checkpoint_folder"], f'model_best.pth')) 
        print("best model saved at epoch", epoch + 1)   
    return best_loss

def to_cpu(x):
    return x.detach().cpu()

def to_numpy(x):
    return to_cpu(x).numpy()

def clean_folder(saved_fn):
    if not os.path.exists(saved_fn):
        #os.mkdir(saved_fn)
        os.makedirs(saved_fn)
    else:
        shutil.rmtree(saved_fn)
        os.mkdir(saved_fn)

def calculate_mutual_info(latent_space_mu,latent_space_var,labels):
    """
    find the mutual information between the latent space and the specific label, could be channels, patient, removed, seziure free, etc

    Args:
        latent_space_mu (np array or pytorch tensor): the latent space mus for each patient, so (n_observations x n_latent_dims)
        latent_space_var (np array or pytorch tensor): the latent space variances for each patient, so (n_observations x n_latent_dims)
        labels (the labels for each patient/observation): the labels for each observation could be channels, patient, removed, seziure free, etc
    
    Returns:
        an array or array type of the mutual information for each latent space dimension
    """
    mutual_information=latent_space_mu[:0]
    mutual_information=0
    n_total=labels.shape[0]
    
    for l in np.unique(labels):
        n_l=(labels==l).sum()
        
        p_l=n_l/n_total
        
        mutual_information-=p_l*np.log(p_l)
        
        #because the latent space mus and vars are Independent
        mu_l=latent_space_mu[labels==l].sum(axis=0)
        var_l=latent_space_var[labels==l].sum(axis=0)
        
        mutual_information+=-p_l*0.5*(np.log(var_l)+np.log(2*np.pi)+1)

    return mutual_information
        
        
    


def create_dataset(data_dir, patient_90,transform):
    def read_dataset(data_dir,pt_name, transform):
        print("???",pt_name)
        #return HFODataset(data_dir, pt_name,transform=transform)
    param_list = [{
                "data_dir":data_dir, 
                 "pt_name":patient_90[idx], 
                 "transform": transform
                 } for idx in range(len(patient_90))]
    print(param_list)
    ret = parallel_process(param_list, read_dataset ,n_jobs=2, use_kwargs=True, front_num=3)
    res = []
    for r in ret:
        print(r.patient_name)
        res.append(r)
    return r

def parallel_process(array, function, n_jobs=16, use_kwargs=False, front_num=3):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=16): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    front = []
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
            out.append(e)
    return front + out

from src.CNNVAE import CNNVAE
def get_gradient(model:CNNVAE):

    decoder_grads = []
    encoder_grads = []

    for layer in model.decoder.modules():
        # print(layer)
        for param in layer.parameters():
            # print(param)
            # print(param.grad)
            decoder_grads.append(param.grad.view(-1).detach().clone())
    
    for layer in model.encoder.modules():
        for param in layer.parameters():
            encoder_grads.append(param.grad.view(-1).detach().clone())

    decoder_grad = torch.cat(decoder_grads)
    encoder_grad = torch.cat(encoder_grads)
    return decoder_grad, encoder_grad


# if __name__ == "__main__":
#     model = 