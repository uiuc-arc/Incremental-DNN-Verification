""" This program runs incremental smoothing on a base classifier. 
"""

import pickle
import sys
import torch
import os
import numpy as np
import pandas as pd
from pathlib import Path
from nnverify import util
from nnverify.common.dataset import Dataset
from nnverify.config import NET_HOME
from nnverify.proof_transfer.approximate import QuantizationType
sys.path.append(str(Path(__file__).resolve().parent))
from datasets import get_dataset, get_num_classes
from core import Smooth
from time import time
from scipy import stats
from certify_logger import Logger


def get_device(approximation):
        if 'qt_type' in dir(approximation) and  approximation.qt_type in [QuantizationType.INT32, QuantizationType.INT16, QuantizationType.INT8]:
            # Since CUDA does not support integer quantization, we use CPU for its inference
            return torch.device("cpu")
        else:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_result_dir(args):
        if args.approximation is None:
            dir = os.path.join('results', 'smoothing', get_dataset_name(args.dataset), get_net_name(args.net_location), 'noise'+str(args.sigma), 'N='+str(args.N))
        else:
            dir = os.path.join('results', 'smoothing', get_dataset_name(args.dataset), get_net_name(args.net_location), 'noise'+str(args.sigma), 'N='+str(args.N), str(args.approximation.approx_type), str(args.approximation))

        # Create directory if it doesn't exist
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir

def get_net_name(net_dir):
    # Extract resnet name
    if 'resnet20' in net_dir:
        return 'resnet20'
    elif 'resnet50' in net_dir:
        return 'resnet50'
    elif 'resnet110' in net_dir:
        return 'resnet110'
    else:
        raise ValueError('Invalid network name')

def get_dataset_name(dataset):
    if dataset == Dataset.CIFAR10:
        return 'cifar10'
    elif dataset == Dataset.IMAGENET:
        return 'imagenet'
    else:
        raise ValueError('Invalid dataset name')

def get_fname(args):
    return 'Nchi=' + str(args.N_chi) + '_N2=' + str(args.N2)

def get_radii_chi_dir(args):
    dir_name = get_result_dir(args)
    radii_dir = os.path.join(dir_name, 'radii')
    chi_dir = os.path.join(dir_name, 'chi')
    if not os.path.exists(radii_dir):
        os.makedirs(radii_dir)
    if not os.path.exists(chi_dir):
        os.makedirs(chi_dir)
    return radii_dir, chi_dir

def get_time_dir(args):
    dir_name = get_result_dir(args)
    time_dir = os.path.join(dir_name, 'time')
    if not os.path.exists(time_dir):
        os.makedirs(time_dir)
    return time_dir
    

class IncrementalRSArgs():
    def __init__(self, dataset, sigma, N, N2, N_chi, alpha1, alpha_chi, batch, approx, count=100, net=None, net_path=None):
        if net_path is not None:
            self.net_location = net_path
        else:
            self.net_location = NET_HOME + str.replace(net, 'sigma', '%.2f' % sigma)

        self.dataset = dataset
        self.sigma = sigma
        self.N = N
        self.alpha1 = alpha1
        self.alpha_chi = alpha_chi
        self.batch = batch
        self.N2 = N2
        self.N_chi = N_chi
        self.count = count
        self.approximation = approx
        

class IncrementalRSResults():
    def __init__(self, args):
        self.args = args
        self.original_radii = None
        self.irs_radii = None
        self.base_radii = None
        self.irs_chi = None
        self.original_pA = None
        self.irs_pA = None
        self.base_pA = None
        self.original_time = None
        self.irs_time = None
        self.base_time = None
        self.gamma = None
    
    def add_original(self, original_radii, original_pA, original_time):
        self.original_radii = np.array(original_radii)
        self.original_pA = np.array(original_pA)
        self.original_time = np.array(original_time)
    
    def add_irs(self, irs_radii, irs_chi, irs_pA, irs_time, gamma):
        self.irs_radii = np.array(irs_radii)
        self.irs_chi = np.array(irs_chi)
        self.irs_pA = np.array(irs_pA)
        self.irs_time = np.array(irs_time)
        self.gamma = gamma
    
    def add_base(self, base_radii, base_pA, base_time):
        self.base_radii = np.array(base_radii)
        self.base_pA = np.array(base_pA)
        self.base_time = np.array(base_time)
    
    def get_mean_radius_all(self):
        return np.mean(self.original_radii), np.mean(self.irs_radii), np.mean(self.base_radii)
    
    def get_mean_time_elapsed_all(self):
        return np.mean(self.original_time), np.mean(self.irs_time), np.mean(self.base_time)
    
    def get_tot_cert_time_all(self):
        return np.sum(self.original_time), np.sum(self.irs_time), np.sum(self.base_time)

    def get_cert_accuracy_all(self, radius: float):
        return np.sum(self.original_radii>= radius)/len(self.original_radii), np.sum(self.irs_radii >= radius)/len(self.irs_radii), np.sum(self.base_radii >= radius)/len(self.base_radii)
    
    
    def load(self):
        fname = get_fname(self.args)
        radii_dir, chi_dir = get_radii_chi_dir(self.args)
        time_dir = get_time_dir(self.args)
        self.original_radii = np.load((os.path.join(radii_dir, fname +  '_original_radii.npy')))
        self.irs_radii = np.load((os.path.join(radii_dir, fname +  '_irs_radii.npy')))
        self.base_radii =np.load((os.path.join(radii_dir, fname +  '_base_radii.npy')))
        self.irs_chi = np.load(os.path.join(chi_dir, fname +  '_irs_chi.npy'), allow_pickle=True)
        self.original_time = np.load((os.path.join(time_dir, fname +  '_original_time.npy')))
        self.irs_time = np.load((os.path.join(time_dir, fname +  '_irs_time.npy')))
        self.base_time =np.load((os.path.join(time_dir, fname +  '_base_time.npy')))
        if self.args.dataset == Dataset.CIFAR10:
            self.gamma = 0.99
        elif self.args.dataset == Dataset.IMAGENET:
            self.gamma = 0.995 
    
    def save(self):
        fname = get_fname(self.args)
        dir_name = get_result_dir(self.args)
        radii_dir, chi_dir = get_radii_chi_dir(self.args)
        time_dir = get_time_dir(self.args)
        mean_df = pd.DataFrame(columns= ['original_mean_radii', 'irs_mean_radii', 'base_mean_radii', 'mean_chi', 'original_mean_pA', 'irs_mean_pA', 'base_mean_pA', 'original_mean_time', 'irs_mean_time', 'base_mean_time', 'pA>gamma'])

        # Remove None values and calculate mean
        mean_clear_none = lambda x: np.mean([val for val in x if val is not None])

        mean_df.loc[0] = [mean_clear_none(self.original_radii), mean_clear_none(self.irs_radii), mean_clear_none(self.base_radii), mean_clear_none(self.irs_chi), mean_clear_none(self.original_pA), mean_clear_none(self.irs_pA), mean_clear_none(self.base_pA), mean_clear_none(self.original_time), mean_clear_none(self.irs_time), mean_clear_none(self.base_time), len(self.original_pA[self.original_pA > self.gamma])/len(self.original_pA)]
        mean_df.to_csv(os.path.join(dir_name, fname + '_summary.txt'), sep='\t', index=False)
        np.save(os.path.join(radii_dir, fname +  '_original_radii.npy'), self.original_radii)
        np.save(os.path.join(radii_dir, fname +  '_irs_radii.npy'), self.irs_radii)
        np.save(os.path.join(radii_dir, fname +  '_base_radii.npy'), self.base_radii)
        np.save(os.path.join(chi_dir, fname +  '_irs_chi.npy'), self.irs_chi)
        np.save(os.path.join(time_dir, fname +  '_original_time.npy'), self.original_time)
        np.save(os.path.join(time_dir, fname +  '_irs_time.npy'), self.irs_time)
        np.save(os.path.join(time_dir, fname +  '_base_time.npy'), self.base_time)
        pickle.dump(self, open(os.path.join(get_result_dir(self.args), get_fname(self.args) + '_result.pkl'), 'wb'))
    

class IncrementalRS():
    def __init__(self, args):   
        self.base_classifier = util.get_net(args.net_location, args.dataset).torch_net
        self.args = args
        self.dataset = get_dataset(self.args.dataset, 'test')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.results = IncrementalRSResults(args)
    
    def certify_original(self):
        sys.stdout = Logger(os.path.join(get_result_dir(self.args), get_fname(self.args) + '_results.log'))
        print("Run Certification of Original Network")
        original_network = Smooth(self.base_classifier, self.dataset, get_num_classes(self.args.dataset), self.args.sigma, dataset_name = self.args.dataset)
        original_network.certify(self.args.count, self.args.N, self.args.alpha1, self.args.batch)
        self.results.add_original(original_network.rs_cache.radii, original_network.rs_cache.pA_arr, original_network.rs_cache.time_arr)
        return original_network.rs_cache, self.results.original_radii, self.results.original_pA, self.results.original_time
    
    def perturb_network(self, approximation):
        perturbed_base_classifier = approximation.approximate(self.args.net_location, self.args.dataset, should_check_accuracy=False, conv=True, dummy=False).torch_net
        return perturbed_base_classifier

    
    def certify_perturbed_irs_and_base(self, rs_cache, perturbed_base_classifier, original_radii, original_pA, original_time, gamma = 0.99):
        self.results.add_original(original_radii, original_pA, original_time)

        # #run certification on perturbed network using theorem 2 and cache_map
        print("Run Certification of Perturbed Network with Cache Map")
        device = get_device(self.args.approximation)
        perturbed_network = Smooth(perturbed_base_classifier, self.dataset, get_num_classes(self.args.dataset), self.args.sigma, device=device, dataset_name = self.args.dataset)
        perturbed_network.certify_with_cache(self.args.count, self.args.N2, self.args.N_chi, self.args.alpha1, self.args.alpha_chi, self.args.batch, rs_cache, gamma) 
        self.results.add_irs(perturbed_network.rs_cache.radii, perturbed_network.rs_cache.chi_arr, perturbed_network.rs_cache.pA_arr, perturbed_network.rs_cache.time_arr, gamma)
      
        #run certification on perturbed network as base comparison
        print("Run Certification of Perturbed Network")
        perturbed_network2 = Smooth(perturbed_base_classifier, self.dataset, get_num_classes(self.args.dataset), self.args.sigma, device=device, dataset_name = self.args.dataset)
        perturbed_network2.certify(self.args.count, self.args.N2, self.args.alpha1 + self.args.alpha_chi, self.args.batch)
        self.results.add_base(perturbed_network2.rs_cache.radii, perturbed_network2.rs_cache.pA_arr, perturbed_network2.rs_cache.time_arr)

        self.results.save()
        return self.results 
    
    def certify_perturbed_irs(self, rs_cache, perturbed_base_classifier, original_radii, original_pA, original_time, gamma = 0.99):
        self.results.add_original(original_radii, original_pA, original_time)

        # #run certification on perturbed network using theorem 2 and cache_map
        print("Run Certification of Perturbed Network with Cache Map")
        device = get_device(self.args.approximation)
        perturbed_network = Smooth(perturbed_base_classifier, self.dataset, get_num_classes(self.args.dataset), self.args.sigma, device=device, dataset_name = self.args.dataset)
        perturbed_network.certify_with_cache(self.args.count, self.args.N2, self.args.N_chi, self.args.alpha1, self.args.alpha_chi, self.args.batch, rs_cache, gamma) 
        self.results.add_irs(perturbed_network.rs_cache.radii, perturbed_network.rs_cache.chi_arr, perturbed_network.rs_cache.pA_arr, perturbed_network.rs_cache.time_arr, gamma)
        return self.results 
    