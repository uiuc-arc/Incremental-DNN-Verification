import nnverify.config as config
import nnverify.proof_transfer.approximate as ap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import tabulate

from unittest import TestCase
from nnverify.common.dataset import Dataset
from nnverify.smoothing.code.incremental_rs import IncrementalRS, IncrementalRSArgs, get_result_dir
from nnverify.smoothing.code.irs_analyze import plot_cert_accuracy, plot_mean_radius, plot_radius_time, plot_pA_distribution, plot_radius_tot_time, plot_accuracy_time, plot_radius_time_with_eb, compute_smoothed_accuracy, compute_std_accuracy



class irs_plot(TestCase):
    def certify_for_each_sample_size(self, net, dataset, approximation=None, N=10000,sigma=1.00):
        samples = [(int)(i*N*0.01) for i in range(1, 11)]
        count=500

        if dataset == Dataset.CIFAR10:
            gamma = 0.99
        elif dataset == Dataset.IMAGENET:
            gamma = 0.995 

        if approximation == None:
            approximations = [ap.Quantize(ap.QuantizationType.INT8), ap.Quantize(ap.QuantizationType.FP16), ap.Quantize(ap.QuantizationType.BF16)]
        else:
            approximations = [approximation]

        original_args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=100, sigma=sigma, N=N, N2=None, N_chi=None, approx=None)
        rs_cache, original_radii, original_pA, original_time = IncrementalRS(original_args).certify_original()

        for approximation in approximations:
            perturbed_base_classifier = IncrementalRS(original_args).perturb_network(approximation)
            res_map = {}
            for sample in samples:
                print("Certifying Sample Size ", sample)
                args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=100, sigma=sigma, N=N, N2=sample, N_chi=sample, approx=approximation)
                res = IncrementalRS(args).certify_perturbed_irs_and_base(rs_cache, perturbed_base_classifier, original_radii, original_pA, original_time, gamma=gamma)
                res_map[sample] = res
            
            plot_mean_radius(args, res_map)
            plot_cert_accuracy(args, res_map)
            plot_pA_distribution(args, res_map)
            plot_radius_tot_time(args, res_map)
            plot_radius_time(args, res_map)
            plot_accuracy_time(args, res_map)

    def test_cifar10_resnet110_bf16_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.BF16))

    def test_cifar10_resnet110_bf16_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.BF16))
    
    def test_cifar10_resnet110_fp16_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.FP16))

    def test_cifar10_resnet110_int8_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.INT8))
    
    def test_cifar10_resnet20_bf16_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.BF16))
    
    def test_cifar10_resnet20_fp16_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.FP16))

    def test_cifar10_resnet20_int8_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.INT8))
    
    def test_imagenet_resnet50_bf16_sigma1(self):
        self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET, ap.Quantize(ap.QuantizationType.BF16))
    
    def test_imagenet_resnet50_fp16_sigma1(self):
        self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET, ap.Quantize(ap.QuantizationType.FP16))

    def test_imagenet_resnet50_int8_sigma1(self):
        self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET, ap.Quantize(ap.QuantizationType.INT8))
    
    # tests for multiple approximation methods
    def test_cifar10_resnet110_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10)
    
    def test_cifar10_resnet20_sigma1(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10)
    
    def test_imagenet_resnet50_sigma1(self):
        self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET)

    def test_cifar10_resnet110_sigma05(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, sigma=0.5)
    
    def test_cifar10_resnet20_sigma05(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigma=0.5)
    
    def test_imagenet_resnet50_sigma05(self):
        self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET, sigma=0.5)

    def test_cifar10_resnet110_sigma25(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, sigma=0.25)
    
    def test_cifar10_resnet20_sigma25(self):
        self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigma=0.25)
    
    def test_imagenet_resnet50_sigma2(self):
        self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET, sigma=2.0)

    # Certify with error bars
    def certify_for_each_sample_size_eb(self, net, dataset, approximation=None, N=10000,sigma=1.00, reapeat=3):
        samples = [(int)(i*N*0.01) for i in range(1, 11)]
        count=500

        if dataset == Dataset.CIFAR10:
            gamma = 0.99
        elif dataset == Dataset.IMAGENET:
            gamma = 0.995 

        res_maps = []
        for _ in range(reapeat):
            original_args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=100, sigma=sigma, N=N, N2=None, N_chi=None, approx=None)
            rs_cache, original_radii, original_pA, original_time = IncrementalRS(original_args).certify_original()

            perturbed_base_classifier = IncrementalRS(original_args).perturb_network(approximation)
            res_map = {}
            for sample in samples:
                print("Certifying Sample Size ", sample)
                args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=100, sigma=sigma, N=N, N2=sample, N_chi=sample, approx=approximation)
                res = IncrementalRS(args).certify_perturbed_irs_and_base(rs_cache, perturbed_base_classifier, original_radii, original_pA, original_time, gamma=gamma)
                res_map[sample] = res
            res_maps.append(res_map)
        
        plot_radius_time_with_eb(args, res_maps)

    def test_cifar10_resnet20_int8_sigma05_eb(self):
        self.certify_for_each_sample_size_eb(config.CIFAR_RESNET_20, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.INT8), sigma=0.5)

    def test_cifar10_resnet110_int8_sigma1_eb(self):
        self.certify_for_each_sample_size_eb(config.CIFAR_RESNET_110, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.INT8), sigma=1.0)
    
    def test_imagenet_resnet50_int8_sigma2_eb(self):
        self.certify_for_each_sample_size_eb(config.RESNET50, Dataset.IMAGENET, ap.Quantize(ap.QuantizationType.INT8), sigma=2.0)

    #N ablation tests
    def n_ablation_cifar10_resnet110(self):
        for N in [1000, 100000, 1000000]:
            for sigma in [0.25, 0.5, 1.0]:
                self.certify_for_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, N = N, sigma= sigma)
    
    def n_ablation_cifar10_resnet20(self):
        for N in [1000, 100000, 1000000]:
            for sigma in [0.25, 0.5, 1.0]:
                self.certify_for_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, N = N, sigma= sigma)
    
    def n_ablation_imagenet_resnet50(self):
        for N in [1000, 100000, 1000000]:
            for sigma in [0.5, 1.0, 2.0]:
                self.certify_for_each_sample_size(config.RESNET50, Dataset.IMAGENET, N= N, sigma= sigma)


class hyperparam(TestCase):
    # grid search tests for gamma
    def gridsearch_gamma(self, net, dataset, approximation, param_list, count=100, sigma=1.0, N=10000):
        samples = [i*100 for i in range(1, 11)]
        irs_mean_radius = []
        res_map = {}
        original_args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=100, sigma=sigma, N=N, N2= None, N_chi= None, approx= approximation)
        rs_cache, original_radii, original_pA, original_time = IncrementalRS(original_args).certify_original()
        perturbed_base_classifier = IncrementalRS(original_args).perturb_network(approximation)

        irs_mean_radius = {gamma:0 for gamma in param_list}
        for gamma in param_list:
            for sample in samples:
                print("Certifying Sample Size ", sample)
                args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=0.001, alpha_chi=0.001, batch=100, sigma=sigma, N=N, N2=sample, N_chi=sample, approx=approximation)
                res = IncrementalRS(args).certify_perturbed_irs_and_base(rs_cache, perturbed_base_classifier, original_radii, original_pA, original_time, gamma)
                original, irs, base = res.get_mean_radius_all()
                irs_mean_radius[gamma] += irs
                res_map[sample] = res
            irs_mean_radius[gamma] /= len(samples)
        
        # Dataframe for mean_radius and gamma
        df = pd.DataFrame(data={'gamma':irs_mean_radius.keys(), 'mean_radius':irs_mean_radius.values()})
        df.to_csv(os.path.join(get_result_dir(args), 'grid_search.txt'), index=None, sep='\t', mode='a')

    def grid_cifar10_resnet20_int8_sigma05(self):
        self.gridsearch_gamma(config.CIFAR_RESNET_20, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.INT8), [0.9, 0.95, 0.975, 0.99, 0.995, 0.999], count = 100, sigma = 0.5, N = 10000)

    def grid_cifar10_resnet110_int8_sigma05(self):
        self.gridsearch_gamma(config.CIFAR_RESNET_110, Dataset.CIFAR10, ap.Quantize(ap.QuantizationType.INT8), [0.9, 0.95, 0.975, 0.99, 0.995, 0.999], count = 100, sigma = 0.5, N = 10000)
    
    def grid_imagenet_resnet50_int8_sigma1(self):
        self.gridsearch_gamma(config.RESNET50, Dataset.IMAGENET, ap.Quantize(ap.QuantizationType.INT8), [0.9, 0.95, 0.975, 0.99, 0.995, 0.999], count = 50, sigma = 1.0, N = 10000)
    
class accuracy(TestCase):
    def smoothed_acc_each_sample_size(self, net, dataset, approximation=None, N=1000,sigmas= None):
        smoothed_acc_table = []
        if approximation == None:
            approximations = [None, ap.Quantize(ap.QuantizationType.FP16), ap.Quantize(ap.QuantizationType.BF16), ap.Quantize(ap.QuantizationType.INT8), ap.Prune(5, True), ap.Prune(10, True), ap.Prune(20, True)]
        else:
            approximations = [approximation]
        for approximation in approximations:
            for sigma in sigmas:
                smoothed_acc = compute_smoothed_accuracy(net, approximation, dataset, alpha1 = 0.001, batch = 100 ,sigma = sigma, N = N)
                smoothed_acc_table.append([dataset, net, approximation, sigma, smoothed_acc])
        print('Smoothed accuracy table: ')
        print(tabulate.tabulate(smoothed_acc_table, headers=['dataset', 'net', 'approximation', 'sigma', 'smoothed_accuracy']))
      
    def std_acc_each_sample_size(self, net, dataset, approximation = None, count = 500, batch = 100, sigmas= None):
        std_acc_table = []
        if approximation == None:
            approximations = [None, ap.Quantize(ap.QuantizationType.FP16), ap.Quantize(ap.QuantizationType.BF16), ap.Quantize(ap.QuantizationType.INT8), ap.Prune(5, True), ap.Prune(10, True), ap.Prune(20, True)]
        else:
            approximations = [approximation]
        for approximation in approximations:
            for sigma in sigmas:
                std_acc = compute_std_accuracy(net, approximation, dataset, count, batch ,sigma = sigma)
                std_acc_table.append([dataset, net, approximation, sigma, std_acc])
        print('Standard accuracy table: ')
        print(tabulate.tabulate(std_acc_table, headers=['dataset', 'net', 'approximation', 'sigma', 'standard_accuracy']))
       
    #standard accuracy test cases
    def standard_acc_cifar10_resnet20(self):
        self.standard_acc_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigmas=[0.25, 0.5, 1.0])
    def standard_acc_cifar10_resnet110(self):
        self.standard_acc_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, sigmas=[0.25, 0.5, 1.0])
    def standard_acc_imagenet_resnet50(self):
        self.standard_acc_each_sample_size(config.RESNET50, Dataset.IMAGENET, sigmas= [0.5, 1.0, 2.0])
    
    #smoothed accuracy test cases
    def smoothed_acc_resnet20_sigma25(self):
        self.smoothed_acc_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigmas=[0.25])
    def smoothed_acc_resnet20_sigma05(self):
        self.smoothed_acc_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigmas=[0.5])
    def smoothed_acc_resnet20_sigma1(self):
        self.smoothed_acc_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigmas=[1.0])
    def smoothed_acc_cifar10_resnet20(self):
        self.smoothed_acc_each_sample_size(config.CIFAR_RESNET_20, Dataset.CIFAR10, sigmas=[0.25, 0.5, 1.0])
    def smoothed_acc_cifar10_resnet110(self):
        self.smoothed_acc_each_sample_size(config.CIFAR_RESNET_110, Dataset.CIFAR10, sigmas=[0.25, 0.5, 1.0])
    def smoothed_acc_imagenet_resnet50(self):
        self.smoothed_acc_each_sample_size(config.RESNET50, Dataset.IMAGENET, sigmas= [0.5, 1.0, 2.0])
    
        

