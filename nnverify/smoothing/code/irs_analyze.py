
from math import inf
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from nnverify import util
import nnverify.config as config
import nnverify.proof_transfer.approximate as ap
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import get_dataset

from nnverify.common.dataset import Dataset
from incremental_rs import IncrementalRSArgs, get_result_dir, IncrementalRSResults, get_device
from scipy.interpolate import interp1d
import tabulate
import argparse
import os
import numpy as np
from datasets import get_dataset, get_num_classes
from core import Smooth

# Plot margin
eps = 0.05

def get_plot_dir(args):
    dir_name = get_result_dir(args)
    plot_dir = os.path.join(dir_name, 'plots')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return plot_dir

def plot_mean_radius(args, res_map):
    line1 = []
    line2 = []
    line3 = []
    max_sample = 0

    for sample in res_map.keys():
        res = res_map[sample]
        original_mean, irs_mean, base_mean = res.get_mean_radius()
        line1.append((base_mean, sample))
        line2.append((irs_mean, sample))
        line3.append((original_mean, sample))
        max_sample = max(sample, max_sample)

    # Plot matplotlib graph
    sns.set_style("darkgrid")
    plt.plot(*zip(*line1), label='baseline', color="blue")
    plt.plot(*zip(*line2), label='IRS', color="orange")
    plt.plot(*zip(*line3), label='original', linestyle = 'dashed', color = 'black')
    plt.xlabel('average certified radius')
    plt.ylabel(r'number of samples $n_p$')
    plt.ylim(0, max_sample)
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'mean_radius_vs_N2.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')

def plot_radius_time_with_eb(args, res_maps):
    max_time = 0
    x1, y1, x2, y2 = [], [], [], []
    mp1, mp2, mp_err1, mp_err2, mp_err3, mp_err4 = {}, {}, {}, {}, {}, {}

    for sample in res_maps[0].keys():
        original_means = []
        base_means = []
        irs_means = []
        base_times = []
        irs_times = []

        for res_map in res_maps:
            res = res_map[sample]
            original_mean, irs_mean, base_mean = res.get_mean_radius()
            original_time, irs_time, base_time = res.get_mean_time_elapsed()

            original_means.append(original_mean)
            base_means.append(base_mean)
            irs_means.append(irs_mean)
            base_times.append(base_time)
            irs_times.append(irs_time)

        base_mean = np.mean(base_means)
        irs_mean = np.mean(irs_means)

        x1.append(base_mean)
        mp1[base_mean] = base_time
        mp_err1[base_mean] = np.std(base_means)
        mp_err3[base_mean] = np.std(base_times)

        x2.append(irs_mean)
        mp2[irs_mean] = irs_time
        mp_err2[irs_mean] = np.std(irs_means)
        mp_err4[irs_mean] = np.std(irs_times)

        max_time = max(irs_time, base_time, max_time)

    # Sort the line    
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    y1 = [mp1[x] for x in x1]
    y2 = [mp2[x] for x in x2]
    eb1 = [mp_err1[x] for x in x1]
    eb2 = [mp_err2[x] for x in x2]
    eb3 = [mp_err3[x] for x in x1]
    eb4 = [mp_err4[x] for x in x2]

    # Plot matplotlib graph
    sns.set_style("darkgrid")
    plt.plot(x1, y1, label='baseline', color="blue")
    plt.plot(x2, y2, label='IRS', color="orange")

    # Plot markers
    plt.scatter(x1, y1, color="blue", s=10)
    plt.scatter(x2, y2, color="orange", s=10)

    plt.errorbar(x1, y1, xerr = eb1, yerr=eb3, color="blue", linestyle = 'None')
    plt.errorbar(x2, y2, xerr = eb2, yerr=eb4, color="orange", linestyle = 'None')

    plt.vlines(x = original_mean, ymin = 0, ymax = max_time*(1+eps), label='original', linestyle = 'dashed', color = 'black')
    plt.xlabel('average certified radius')
    plt.ylabel('average certification time (sec)')
    plt.ylim(0, max_time*(1+eps))
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'mean_radius_vs_time_with_eb.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')


def plot_cert_accuracy(args, res_map, radius = 0.5):
    line1 = []
    line2 = []
    line3 = []
    max_sample = 0

    for sample in res_map.keys():
        res = res_map[sample]
        original_cert, irs_cert, base_cert = res.get_cert_accuracy(radius)
        line1.append((base_cert, sample))
        line2.append((irs_cert, sample))
        line3.append((original_cert, sample))
        max_sample = max(sample, max_sample)

    # Plot matplotlib graph
    sns.set_style("darkgrid")
    plt.plot(*zip(*line1), label='baseline', color="blue")
    plt.plot(*zip(*line2), label='IRS', color="orange")
    plt.plot(*zip(*line3), label='original', linestyle = 'dashed', color = 'black')
    plt.ylim(0, max_sample)
    plt.xlabel('certification accuracy')

    plt.ylabel(r'number of samples $n_p$')
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'cert_accuracy_vs_N2.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')


def plot_accuracy_time(args, res_map, radius = 0.5):
    max_time = 0
    x1, y1, x2, y2 = [], [], [], []
    mp1, mp2 = {}, {}
    
    for sample in res_map.keys():
        res = res_map[sample]
        original_acc, irs_acc, base_acc = res.get_cert_accuracy(radius)
        original_time, irs_time, base_time = res.get_mean_time_elapsed()

        x1.append(base_acc)
        mp1[base_acc] = base_time
        x2.append(irs_acc)
        mp2[irs_acc] = irs_time

        max_time = max(irs_time, base_time, max_time)
   
    # Sort the line    
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    y1 = [mp1[x] for x in x1]
    y2 = [mp2[x] for x in x2]

    # Plot matplotlib graph
    sns.set_style("darkgrid")
    plt.plot(x1, y1, label='baseline', color="blue")
    plt.plot(x2, y2, label='IRS', color="orange")

    # Plot markers
    plt.scatter(x1, y1, color="blue", s=10)
    plt.scatter(x2, y2, color="orange", s=10)

    plt.vlines(x = original_acc, ymin = 0, ymax = max_time*(1+eps), label='original', linestyle = 'dashed', color = 'black')
    plt.xlabel('certification accuracy at r = ' + str(radius))
    plt.ylabel('average certification time (sec)')
    plt.ylim(0, max_time*(1+eps))
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'cert_acc_vs_time_r=' + str(radius) + '.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')

    # Compute speedup 
    x_range = np.linspace(max(np.min(x1), np.min(x2)), min(np.max(x1), np.max(x2)), 100)
    # y1_vals = np.array([interpolate(x1, y1, x) for x in x_range])
    # y2_vals = np.array([interpolate(x2, y2, x) for x in x_range])
    # speedup = np.mean(y1_vals/y2_vals)
    vals1 = interp1d(x1, y1)
    vals2 = interp1d(x2, y2)
    speedup = np.mean(vals1(x_range)/vals2(x_range))
    

    # write speedup to file
    with open(os.path.join(plot_dir,'cert_acc_speedup_r=' + str(radius) + '.txt'), 'w') as f:
        f.write('Speedup: '+str(speedup))

def plot_pA_distribution(args, res_map):
    res = res_map[list(res_map.keys())[0]]
    sns.set_style("darkgrid")
    
    if (res.base_pA is None):
        print('Note: No perturbed network pA found')
        return

    sns.histplot(res.base_pA[res.base_pA >= 0.5], label = 'perturbed network', kde = 'True', stat= 'percent')
    plt.xlabel('pA')
    plt.ylabel('percentage')
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'pA_distribution.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')

def interpolate(xs, ys, x_new):
    for i in range(1, len(xs)):
        if xs[i-1] <= x_new and x_new <= xs[i]:
            return ys[i-1] + (ys[i] - ys[i-1]) * (x_new - xs[i-1]) / (xs[i] - xs[i-1])
    raise Exception('x_new is out of range')

def compute_average_speedup(res_map):
    x1, y1, x2, y2 = [], [], [], []
    mp1, mp2 = {}, {}

    for sample in res_map.keys():
        res = res_map[sample]
        _, irs_mean_radius, base_mean_radius = res.get_mean_radius()
        _, irs_time, base_time = res.get_mean_time_elapsed()

        x1.append(base_mean_radius)
        x2.append(irs_mean_radius)
        mp1[base_mean_radius] = base_time
        mp2[irs_mean_radius] = irs_time

    # Sort the line    
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    y1 = [mp1[x] for x in x1]
    y2 = [mp2[x] for x in x2]

    # Compute speedup 
    x_range = np.linspace(max(np.min(x1), np.min(x2)), min(np.max(x1), np.max(x2)), 100)
    y1_vals = np.array([interpolate(x1, y1, x) for x in x_range])
    y2_vals = np.array([interpolate(x2, y2, x) for x in x_range])
    speedup = np.mean(y1_vals/y2_vals)

    min_speedup, max_speedup = np.min(y1_vals/y2_vals), np.max(y1_vals/y2_vals)
    print('Speedup range: ', min_speedup, max_speedup)
    
    print('Speedup: ', speedup)
    return speedup

def plot_radius_time(args, res_map):
    max_time = 0
    x1, y1, x2, y2 = [], [], [], []
    mp1, mp2 = {}, {}

    for sample in res_map.keys():
        res = res_map[sample]
        original_mean, irs_mean, base_mean = res.get_mean_radius()
        original_time, irs_time, base_time = res.get_mean_time_elapsed()

        x1.append(base_mean)
        mp1[base_mean] = base_time
        x2.append(irs_mean)
        mp2[irs_mean] = irs_time

        max_time = max(irs_time, base_time, max_time)

    # Sort the line    
    x1 = np.sort(x1)
    x2 = np.sort(x2)
    y1 = [mp1[x] for x in x1]
    y2 = [mp2[x] for x in x2]

    # Plot matplotlib graph
    sns.set_style("darkgrid")
    plt.plot(x1, y1, label='baseline', color="blue")
    plt.plot(x2, y2, label='IRS', color="orange")

    # Plot markers
    plt.scatter(x1, y1, color="blue", s=10)
    plt.scatter(x2, y2, color="orange", s=10)

    plt.vlines(x = original_mean, ymin = 0, ymax = max_time*(1+eps), label='original', linestyle = 'dashed', color = 'black')
    plt.xlabel('average certified radius')
    plt.ylabel('average certification time (sec)')
    plt.ylim(0, max_time*(1+eps))
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'mean_radius_vs_time.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')

    speedup = compute_average_speedup(res_map)

    # write speedup to file
    with open(os.path.join(plot_dir,'speedup.txt'), 'w') as f:
        f.write('Speedup: '+str(speedup))


def plot_radius_tot_time(args, res_map):
    line1 = []
    line2 = []
    max_time = 0
    for sample in res_map.keys():
        res = res_map[sample]
        original_mean, irs_mean, base_mean = res.get_mean_radius()
        original_time, irs_time, base_time = res.get_tot_cert_time()
        line1.append((base_mean, base_time))
        line2.append((irs_mean, irs_time))
        max_time = max(irs_time, base_time, max_time)

    # Plot matplotlib graph
    sns.set_style("darkgrid")
    plt.plot(*zip(*line1), label='baseline', color = 'blue')
    plt.plot(*zip(*line2), label='IRS', color = 'orange')
    plt.vlines(x = original_mean, ymin = 0, ymax = max_time, label='original', linestyle = 'dashed', color = 'green')
    plt.xlabel('average certified radius')
    plt.ylabel('total certification time (sec)')
    plt.ylim(0, max_time)
    plt.legend()

    plot_dir = get_plot_dir(args)
    plot_loc = os.path.join(plot_dir,'mean_radius_vs_total_time.png')
    print('Plotting to: ', plot_loc)
    plt.savefig(plot_loc, dpi=300)
    plt.close('all')

def compute_cert_stats(args, res_map):
    irs_better = 0
    base_better = 0
    equal = 0

    # Compute certification success rate
    for sample in res_map.keys():
        res = res_map[sample]
        irs_radii = res.irs_radii
        base_radii = res.base_radii

        irs_better = irs_better + np.sum(irs_radii > base_radii)
        base_better = base_better + np.sum(irs_radii < base_radii)
        equal = equal + np.sum(irs_radii == base_radii)

    print('IRS better: ', irs_better/(irs_better + base_better + equal))
    print('Base better: ', base_better/(irs_better + base_better + equal))
    print('Equal: ', equal/(irs_better + base_better + equal))


def compute_std_accuracy(net, approximation, dataset, count, batch, sigma):
    net_location = config.NET_HOME + str.replace(net, 'sigma', '%.2f' % sigma)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if approximation == None:
        perturbed_base_classifier = util.get_net(net_location, dataset).torch_net.to(device).eval()
    else:
        perturbed_base_classifier = approximation.approximate(net_location, dataset, should_check_accuracy=False, conv=True, dummy=False).torch_net.to(device).eval()
    
    # compute accuracy
    dataset = get_dataset(dataset, 'test')
    correct = 0                

    #TODO: Make this efficient with batching            
    for i in tqdm(range(len(dataset))):
        (x, label) = dataset[i]
        prediction = perturbed_base_classifier(x.to(device).unsqueeze(0)).flatten().argmax().item()
        correct += int(prediction == label)

    print('Standard accuracy: ', correct/len(dataset))
    return correct/len(dataset)

def compute_smoothed_accuracy(net, approximation, dataset, alpha1, batch, sigma, N):
    net_location = config.NET_HOME + str.replace(net, 'sigma', '%.2f' % sigma)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if approximation == None:
        base_classifier = util.get_net(net_location, dataset).torch_net
        network = Smooth(base_classifier, dataset, get_num_classes(dataset), sigma, dataset_name = dataset)
    else:
        perturbed_base_classifier = approximation.approximate(net_location, dataset, should_check_accuracy=False, conv=True, dummy=False).torch_net
        device = get_device(approximation)
        network = Smooth(perturbed_base_classifier, dataset, get_num_classes(dataset), sigma, device=device, dataset_name = dataset)

    dataset = get_dataset(dataset, 'test')
    correct = 0  
    for i in tqdm(range(len(dataset))):
        (x, label) = dataset[i]
        x = x.to(device)
        prediction = network.predict(x, N, alpha1, batch)
        correct += int(prediction == label)
    
    print('Smoothed accuracy: ', correct/len(dataset))
    return correct/len(dataset)


def plot_stored(net, approximation, dataset, count, alpha1, alpha_chi, batch, sigma, N):
    res_map = get_res_map(net, approximation, dataset, count, alpha1, alpha_chi, batch, sigma, N)
    args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=alpha1, alpha_chi= alpha_chi, batch=batch, sigma=sigma, N=N, N2=None, N_chi=None, approx=approximation)
    
    plot_mean_radius(args, res_map)
    plot_cert_accuracy(args, res_map)
    plot_radius_tot_time(args, res_map)
    plot_pA_distribution(args, res_map)
    plot_radius_time(args, res_map)
    plot_accuracy_time(args, res_map)


def get_res_map(net, approximation, dataset, count, alpha1, alpha_chi, batch, sigma, N):
    samples = [(int)(0.01 * i * N) for i in range(1, 11)]
    res_map = {}
    for sample in samples:
        args = IncrementalRSArgs(net=net, dataset=dataset, count=count, alpha1=alpha1, alpha_chi= alpha_chi, batch=batch, sigma=sigma, N=N, N2=sample, N_chi=sample, approx=approximation)
        res = IncrementalRSResults(args)
        res.load()
        res_map[sample] = res

    return res_map

def get_sigmas_from_query(query, dataset):
    if query == None:
        if dataset == Dataset.IMAGENET:
            return [0.5, 1.0, 2.0]
        elif dataset == Dataset.CIFAR10:
            return [0.25, 0.5, 1.0]
    else:
        return [query]

def get_approximations_from_query(query):
    if query == 'all':
        return APPROXIMATIONS.values()
    elif query == 'quant':
        return [APPROXIMATIONS['int8'], APPROXIMATIONS['fp16'], APPROXIMATIONS['bf16']]
    elif query == 'prune':
        return [APPROXIMATIONS['prune5'], APPROXIMATIONS['prune10'], APPROXIMATIONS['prune20']]
    elif query in APPROXIMATIONS.keys():
        return [APPROXIMATIONS[query]]
    else:
        raise Exception('Invalid approximation query')


if __name__=='__main__':
    DATASETS = {'ImageNet': Dataset.IMAGENET, 'CIFAR10':Dataset.CIFAR10}

    APPROXIMATIONS = {'int8':ap.Quantize(ap.QuantizationType.INT8), 'fp16': ap.Quantize(ap.QuantizationType.FP16), 'bf16': ap.Quantize(ap.QuantizationType.BF16), 'prune5': ap.Prune(5, True), 'prune10': ap.Prune(10, True), 'prune20': ap.Prune(20, True)}

    NETS = {'resnet50': config.RESNET50, 'resnet110': config.CIFAR_RESNET_110, 'resnet20': config.CIFAR_RESNET_20}

    color_blue = '\033[94m'
    color_def = '\033[0m'

    parser = argparse.ArgumentParser(description='Plot Stored IRS Experiment Data')
    parser.add_argument("--dataset", choices=DATASETS.keys(), help="which dataset")
    parser.add_argument("--net", choices=NETS.keys(), help="pytorch net")
    parser.add_argument("--approximation", default='all', help="type of approximation")
    parser.add_argument("--sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch", type=int, default=100, help="batch size")
    parser.add_argument("--N", type=int, default=10000, help="number of samples to use")
    parser.add_argument("--N2", type=int, default=1000)
    parser.add_argument("--Nchi", type=int, default=1000)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--alpha1", type=float, default=0.001, help="failure probability")
    parser.add_argument("--alpha_chi", type=float, default=0.001, help="chi estimation probability")
    parser.add_argument("--plot", action='store_true', help="plot the stored data")
    parser.add_argument("--std_acc", action='store_true', help="compute standard accuracy")
    parser.add_argument("--smoothed_acc", action='store_true', help="compute smooted accuracy")
    parser.add_argument("--chi", action='store_true', help="compute zeta")
    parse_args = parser.parse_args()

    approximations = get_approximations_from_query(parse_args.approximation)
    dataset = DATASETS[parse_args.dataset]
    sigmas = get_sigmas_from_query(parse_args.sigma, dataset)
    
    tables = {}
    tables['speedup'] = []
    tables['chi'] = []

    if parse_args.std_acc:
        acc_table = []
        for sigma in sigmas:
            acc = compute_std_accuracy(NETS[parse_args.net], None, dataset, parse_args.count, parse_args.batch, sigma)
            acc_table.append([parse_args.dataset, parse_args.net, 'original', sigma, acc])
    
    if parse_args.smoothed_acc:
        smoothed_acc_table = []
        for sigma in sigmas:
            smoothed_acc = compute_smoothed_accuracy(NETS[parse_args.net], None, dataset, parse_args.alpha1, parse_args.batch, sigma, parse_args.N)
            smoothed_acc_table.append([parse_args.dataset, parse_args.net, 'original', sigma, smoothed_acc])

    for approximation in approximations:
        for sigma in sigmas:
            print(color_blue, 'Analysis for', color_def, '\n approximation:', approximation, '\n sigma:', sigma, '\n dataset:', dataset, '\n net:', parse_args.net)
            if parse_args.plot:
                plot_stored(NETS[parse_args.net], approximation, dataset, parse_args.count, parse_args.alpha1, parse_args.alpha_chi, parse_args.batch, sigma, parse_args.N)
            if parse_args.std_acc:
                acc = compute_std_accuracy(NETS[parse_args.net], approximation, dataset, parse_args.count, parse_args.batch, sigma)
                acc_table.append([parse_args.dataset, parse_args.net, approximation, sigma, acc])
            if parse_args.smoothed_acc:
                smoothed_acc = compute_smoothed_accuracy(NETS[parse_args.net], approximation, dataset, parse_args.alpha1, parse_args.batch, sigma, parse_args.N)
                smoothed_acc_table.append([parse_args.dataset, parse_args.net, approximation, sigma, smoothed_acc])
            if parse_args.chi:
                res_map = get_res_map(NETS[parse_args.net], approximation, dataset, parse_args.count, parse_args.alpha1, parse_args.alpha_chi, parse_args.batch, sigma, parse_args.N)
                for sample in res_map.keys():
                    res = res_map[sample]
                mean_chi = np.mean(res.irs_chi[res.irs_chi != None])
                print('Mean chi: ', mean_chi)
                tables['chi'].append([parse_args.dataset, parse_args.net, approximation, sigma, mean_chi])

            res_map = get_res_map(NETS[parse_args.net], approximation, dataset, parse_args.count, parse_args.alpha1, parse_args.alpha_chi, parse_args.batch, sigma, parse_args.N)
            compute_cert_stats(parse_args, res_map)
            
            # Speedup
            speedup = compute_average_speedup(res_map)
            tables['speedup'].append([parse_args.dataset, parse_args.net, approximation, sigma, speedup])
    
    print('Speedup table: ')
    print(tabulate.tabulate(tables['speedup'], headers=['dataset', 'net', 'approximation', 'sigma', 'speedup']))

    if parse_args.std_acc:
        print('Standard accuracy table: ')
        print(tabulate.tabulate(acc_table, headers=['dataset', 'net', 'approximation', 'sigma', 'accuracy']))
    
    if parse_args.smoothed_acc:
        print('Smoothed accuracy table: ')
        print(tabulate.tabulate(smoothed_acc_table, headers=['dataset', 'net', 'approximation', 'sigma', 'smoothed_accuracy']))

    if parse_args.chi:
        print('Chi table: ')
        print(tabulate.tabulate(tables['chi'], headers=['dataset', 'net', 'approximation', 'sigma', 'mean_chi']))
