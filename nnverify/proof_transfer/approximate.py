'''
Copied from https://github.com/uiuc-arc/FANC/blob/main/proof_transfer/approximate.py
WIP to work with this repo
Generate the approximated networks provided the original network. The input is in saved pytorch format.
The generated outputs are in ONNX format.
'''
import torch
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
import numpy as np
import torch.nn as nn
from nnverify.common import Domain, strip_name

from torch.nn import functional as F
from enum import Enum
from nnverify import util, parse
from nnverify.common.network import LayerType
from nnverify.common.dataset import Dataset
from nnverify.training.base import train
from nnverify.training.lirpa import train as lirpa_train
from nnverify.training.schedules import get_lr_policy, get_optimizer
from nnverify.training.training_args import TrainArgs


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QuantizationType(Enum):
    INT8 = 1
    INT16 = 2
    INT32 = 3
    FP16 = 4
    BF16 = 5


class Quantize:
    def __init__(self, qt_type):
        self.qt_type = qt_type
        self.approx_type = 'quantize'

    def __repr__(self):
        return str(self.qt_type)

    def approximate(self, net_name, dataset, should_check_accuracy = True, conv = False, dummy=True):
        """
        Approximate the network with the given quantization type.
        net_name: the name of the network
        dataset: the dataset used to approximate the network
        should_check_accuracy: whether to check the accuracy of the approximated network
        conv: whether to approximate the convolutional layers
        dummy: whether to use dummy quantization
        """
        net = util.get_net(net_name, dataset)
        net.torch_net.eval()

        if self.qt_type == QuantizationType.INT8:
            if dummy:
                net = util.get_net(net_name, dataset)
                dummy_quant(net, 8, conv)
            else:
                self._quantize(net, torch.qint8)           
        elif self.qt_type == QuantizationType.INT16:
            if dummy:
                dummy_quant(net, 16, conv)
            else:
                raise ValueError("Unsupported approximation!")
        elif self.qt_type == QuantizationType.FP16:
            if dummy:
                dummy_quant_float(net, conv)
            else:
                net.torch_net = net.torch_net.to(dtype=torch.float16).to(dtype=torch.float32)
        elif self.qt_type == QuantizationType.BF16:
            net.torch_net = net.torch_net.to(dtype=torch.bfloat16).to(dtype=torch.float32)
        else:
            raise ValueError("Unsupported approximation!")
        
        if should_check_accuracy:
            check_accuracy(net, dataset)
        return net

    def _quantize(self, net, dtype):
        net.torch_net = torch.ao.quantization.quantize_dynamic(
                        net.torch_net,  # the original model
                            {torch.nn.Conv2d},  # a set of layers to dynamically quantize
                                dtype=dtype)  # the target dtype for quantized weights

class Prune:
    def __init__(self, percent, torch_prune = False):
        self.percent = percent
        self.torch_prune = torch_prune
        self.approx_type = 'prune'

    def __repr__(self):
        return 'prune' + str(self.percent)

    def approximate(self, net_name, dataset, should_check_accuracy = True, conv = False, dummy = False):
        net = util.get_net(net_name, dataset)
        net_format = net.net_format
        if should_check_accuracy:
            check_accuracy(net, dataset)
        prune_model(net.torch_net, dataset, prune_percent=self.percent, conv = conv, net_format= net_format, torch_prune= self.torch_prune)
        return net


class Finetune:
    def __init__(self, train_args=TrainArgs(epochs=20, lr=0.001)):
        self.train_args = train_args

    def __repr__(self):
        return 'finetune:e' + str(self.train_args.epochs) + ',lr:' + str(self.train_args.lr) \
               + '_' + strip_name(self.train_args.trainer)

    def approximate(self, net_name, dataset, should_check_accuracy = True, conv = False):
        model = util.get_net(net_name, dataset)
        if should_check_accuracy:
            check_accuracy(net, dataset)

        model = util.get_torch_net(net_name, dataset)
        optimizer = get_optimizer(model, self.train_args)
        lr_policy = get_lr_policy(self.train_args.lr_schedule)(optimizer, self.train_args)
        criterion = nn.CrossEntropyLoss()
        trainloader = util.prepare_data(dataset, train=True, batch_size=self.train_args.batch_size, normalize=True)

        ## fine-tune the model ##
        for epoch in range(self.train_args.epochs):
            if self.train_args.trainer == Domain.BASE:
                train(model, 'cpu', trainloader, criterion, optimizer, epoch, self.train_args)
            else:
                lirpa_train(model, 'cpu', trainloader, criterion, optimizer, epoch, self.train_args, dataset)
        net = parse.parse_torch_layers(model)
        return net


class Random:
    def __init__(self, ptb_perc, layers=None):
        self.ptb_perc = ptb_perc

    def __repr__(self):
        return 'random' + str(self.ptb_perc)

    def approximate(self, net_name, dataset, should_check_accuracy = True, conv = False):
        net = util.get_net(net_name, dataset)
        if should_check_accuracy:
            check_accuracy(net, dataset)
        for layer in net:
            if layer.type is not LayerType.Linear:
                continue

            rand_tensor = 2 * torch.rand(layer.weight.shape) - 1  # in the range [-1, 1]
            max_ptb = torch.abs(layer.weight) * self.ptb_perc
            layer.weight = layer.weight + rand_tensor * max_ptb
        return net


def get_approx_net_name(net_name, approx_type):
    tmp_str = net_name.split('.')
    tmp_str[-1:-1] = [str(approx_type).split('.')[-1]]
    return ".".join(tmp_str)


def prune_model(model, dataset, skip_layer=0, prune_percent=50, post_finetune=False, conv = False, net_format = 'torch', torch_prune = False):
    density(model)
    #prev_accuracy = check_accuracy(model, dataset)

    if (torch_prune):
        prune_weights_torch(model, prune_percent)
    else:
        prune_weights(model, prune_percent, skip_layer=skip_layer)

    #check_accuracy(model, dataset)
    density(model)

    #print("Fine tune the network to get accuracy:", prev_accuracy)

    if post_finetune:
        ## TODO: Fix this
        finetune(model, dataset, req_accuracy=prev_accuracy)

    #density(model)

def prune_weights_torch(net, per, conv = False):
    parameters_to_prune = []
    for layer in net.modules():
        if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue
        print('Pruning layer: ', layer.type, ' | Percentage: ', per)
        parameters_to_prune.append((layer, 'weight'))
        #prune.l1_unstructured(layer, name="weight", amount= per/100.00)
    prune.global_unstructured( tuple(parameters_to_prune), pruning_method=prune.L1Unstructured, amount= per/100.00)




def prune_weights(net, per, skip_layer=0, conv = False):

    with torch.no_grad():
    
        for layer in net.modules():
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue

            weight = layer.weight

            per_it = per
            if skip_layer > 0:
                skip_layer -= 1
                per_it = 0

            print('Pruning layer: ', layer.type, ' | Percentage: ', per_it)
            cutoff = np.percentile(np.abs(weight.detach().numpy()), per_it)

            if len(weight.shape) == 4:
                for w in range(weight.shape[0]):
                    for x in range(weight.shape[1]):
                        for y in range(weight.shape[2]):
                            for z in range(weight.shape[3]):
                                if abs(weight[w][x][y][z]) < cutoff:
                                        weight[w][x][y][z] = 0
                                    

            if len(weight.shape) == 2:
                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        if abs(weight[i][j]) < cutoff:
                                weight[i][j] = 0

            elif len(weight.shape) == 1:
                for i in range(weight.shape[0]):
                    if abs(weight[i]) < cutoff:
                            weight[i] = 0


def density(net, conv = False):
    count = 0
    count_nz = 0

    for layer in net.modules():

        if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
            continue

        
        

        weight = layer.weight

        # Transform the parameter as required.
        if len(weight.shape) == 2:
            for i in range(weight.shape[0]):
                for j in range(weight.shape[1]):
                    count += 1
                    if weight[i][j] != 0:
                        count_nz += 1

        elif len(weight.shape) == 1:
            for i in range(weight.shape[0]):
                count += 1
                if weight[i][j] != 0:
                    count_nz += 1
        
        elif len(weight.shape) == 4:
            for w in range(weight.shape[0]):
                for x in range(weight.shape[1]):
                    for y in range(weight.shape[2]):
                        for z in range(weight.shape[3]):
                            count += 1
                            if weight[w][x][y][z] != 0:
                                count_nz += 1
                            
    print('Density :', count_nz * 1.0 / count)


'''
WIP: Get bounds for a layer using interval propogation. 
TODO: 
1. Use multiple layers
2. Refactor the propagation code to a separate analyzer module
'''


def get_bounds(images, params):
    eps = 0.05
    is_conv = True
    if not is_conv:

        lb = (images - eps).reshape(images.shape[0], -1)
        ub = (images + eps).reshape(images.shape[0], -1)

        pos_wt = F.relu(params[0])
        neg_wt = -F.relu(-params[0])

        oub = F.relu(ub @ pos_wt.T + lb @ neg_wt.T)
        olb = F.relu(lb @ pos_wt.T + ub @ neg_wt.T)
    else:
        lb = (images - eps).reshape(images.shape[0], -1)
        ub = (images + eps).reshape(images.shape[0], -1)

        weight = params[0]
        bias = params[1]

        num_kernel = weight.shape[0]

        k_h, k_w = 4, 4
        s_h, s_w = 2, 2
        p_h, p_w = 1, 1

        input_h, input_w = 28, 28

        output_h = int((input_h + 2 * p_h - k_h) / s_h + 1)
        output_w = int((input_w + 2 * p_w - k_w) / s_w + 1)

        linear_cof = []

        size = 784
        shape = (1, 28, 28)

        cof = torch.eye(size).reshape(size, *shape)
        pad2d = (p_w, p_w, p_h, p_h)
        cof = F.pad(cof, pad2d)

        for i in range(output_h):
            w_cof = []
            for j in range(output_w):
                h_start = i * s_h
                h_end = h_start + k_h
                w_start = j * s_w
                w_end = w_start + k_w

                w_cof.append(cof[:, :, h_start: h_end, w_start: w_end])

            linear_cof.append(torch.stack(w_cof, dim=1))

        linear_cof = torch.stack(linear_cof, dim=1).reshape(size, output_h, output_w, -1)

        new_weight = weight.reshape(num_kernel, -1).T
        new_cof = linear_cof @ new_weight
        new_cof = new_cof.permute(0, 3, 1, 2).reshape(size, -1)

        pos_wt = F.relu(new_cof)
        neg_wt = -F.relu(-new_cof)

        bias = bias.view(-1, 1, 1).expand(num_kernel, output_h, output_w).reshape(1, -1)

        oub = F.relu(ub @ pos_wt.T + lb @ neg_wt.T)
        olb = F.relu(lb @ pos_wt.T + ub @ neg_wt.T)

    return olb, oub


"""
Currently only works with onnx
"""


def check_accuracy(net, dataset):
    if dataset == Dataset.ACAS:
        return None

    testloader = util.prepare_data(dataset, False, batch_size=net.input_shape[0])
    inputs, _ = next(iter(testloader))

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = util.compute_output_tensor(inputs, net)

            predicted = outputs[1].argmax()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('Acc: %.3f%% (%d/%d)' % (100. * correct / total, correct, total))
    return correct / total



def dummy_quant(net, quant_bit, conv = False):
    # Dummy quantized
    # Calculate max to do the quantization symmetric, per-tensor

    def quant(x, scale):
        return int(x * scale)

    def unquant(x, scale):
        return x / scale
    with torch.no_grad(): 
        
        for layer in net.modules():
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue



            weight = layer.weight
            if len(weight.shape) == 2:
                abs_max = weight.abs().max()

                scale = (2 ** (quant_bit - 1)) / abs_max

                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        weight[i][j] = unquant(quant(weight[i][j], scale), scale)

            elif len(layer.weight.shape) == 1:
                abs_max = weight.abs().max()
                scale = (2 ** (quant_bit - 1)) / abs_max

                for i in range(weight.shape[0]):
                    weight[i] = unquant(quant(weight[i], scale), scale)
            
            elif len(layer.weight.shape) == 4:
                abs_max = weight.abs().max()
                scale = (2 ** (quant_bit - 1)) / abs_max

                for w in range(weight.shape[0]):
                    for x in range(weight.shape[1]):
                        for y in range(weight.shape[2]):
                            for z in range(weight.shape[3]):
                                weight[w][x][y][z] = unquant(quant(weight[w][x][y][z], scale), scale)


            else:
                print('Param shape length is: ', len(weight.shape))

            # Update the weight
            layer.weight = weight


def dummy_quant_float(net, conv = False):
    # Dummy quantized
    # Calculate max to do the quantization symmetric, per-tensor

    def quant(x):
        return torch.Tensor([float(np.float16(x.item()))])

    with torch.no_grad(): 
        
        for layer in net.modules():
            if not (isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d)):
                continue


            weight = layer.weight
            if len(weight.shape) == 2:
                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        weight[i][j] = quant(weight[i][j])

            elif len(layer.weight.shape) == 1:
                for i in range(weight.shape[0]):
                    weight[i] = quant(weight[i])

            elif len(layer.weight.shape) == 4:
                for w in range(weight.shape[0]):
                    for x in range(weight.shape[1]):
                        for y in range(weight.shape[2]):
                            for z in range(weight.shape[3]):
                                weight[w][x][y][z] = quant(weight[w][x][y][z])

            else:
                print('Param shape length is: ', len(weight.shape))

            # Update the weight
            layer.weight = weight


