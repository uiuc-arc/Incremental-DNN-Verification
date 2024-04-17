import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from time import time
import datetime

from nnverify.common.dataset import Dataset


class RSCache():
    def __init__(self):
        self.pA_arr = []
        self.radii = []
        self.seeds = []
        self.predictions = []
        self.chi_arr = []
        self.cAHat_arr = []
        self.time_arr = []

    def add(self, pABar, radius, seed, predictions, chi, cAHat):
        self.pA_arr.append(pABar)
        self.radii.append(radius)
        self.seeds.append(seed)
        self.predictions.append(predictions)
        self.chi_arr.append(chi)
        self.cAHat_arr.append(cAHat)
    def add_time_elapsed(self, time_elapsed):
        self.time_arr.append(time_elapsed)


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, dataset, num_classes: int, sigma: float, device=None, dataset_name = Dataset.IMAGENET):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        if device == None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print("Initiating smooth network on device: ", self.device) 
        if dataset_name == Dataset.IMAGENET:
            # Removing the Dataparallel wrapper used for multi-gpu training
        
            base_classifier = torch.nn.Sequential(base_classifier[0], base_classifier[1].module)

        self.base_classifier = base_classifier.to(self.device)
        self.num_classes = num_classes
        self.sigma = sigma
        self.dataset = dataset
        self.rs_cache = RSCache()

        
    
    def certify(self, count: int, n: int, alpha: float, batch_size: int):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.

        """
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)

        n0 = n//100
        if (count < 0 or count > len(self.dataset)):
            count = len(self.dataset)

        self.base_classifier.eval()
        

        for i in range(len(self.dataset)):
            if i == count:
                break

            (x, label) = self.dataset[i]
            x = x.to(self.device)

            before_time = time()
            
            cAHat, radius, correct = self.compute_radius(x, n, n0, batch_size, alpha, label)

            time_elapsed = datetime.timedelta(seconds=(time() - before_time))
            self.rs_cache.add_time_elapsed(time_elapsed.total_seconds())
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, cAHat, radius, correct, str(time_elapsed)), flush=True)
    
  
    
    def compute_radius(self, x, n, n0, batch_size, alpha, label):
        # draw samples of f(x+ epsilon)
        counts_selection, _, _ = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        correct = int(cAHat == label)
        if (correct != 1):
            self.rs_cache.add(0.0, 0.0, None, None, 0.0, cAHat)
            return cAHat, 0.0, correct
        # draw more samples of f(x + epsilon)
        counts_estimation, seeds, predictions = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        print('pA: ', pABar)
        
    
        if pABar < 0.5:
            cAHat = Smooth.ABSTAIN
            radius = 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
        
        self.rs_cache.add(pABar, radius, seeds, predictions, None, cAHat)

        return cAHat, radius, correct

    def compute_radius_with_chi(self, x, n_chi, batch_size, alpha_chi, label, seeds, predictions, original_pA):

        # Estimate chi with n_chi samples
        chi, counts_selection = self._estimate_chi(x, n_chi, batch_size, alpha_chi, seeds, predictions)
        print('chi: ', chi)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        correct = int(cAHat == label)
        print('pA: ', original_pA - chi)
        if (original_pA - chi) < 0.5:
            cAHat = Smooth.ABSTAIN
            radius = 0.0
        else:
            radius = self.sigma/2 * (norm.ppf(original_pA - chi) - norm.ppf(1 - original_pA + chi)) 
        
        self.rs_cache.add(original_pA - chi, radius, None, None, chi, cAHat)
        return cAHat, radius, correct

    def certify_with_cache(self, count: int, n2: int, n_chi: int, alpha1, alpha_chi, batch_size: int, original_rs_cache: np.ndarray, gamma: int):
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)
        n0 = n2//10
        if (count < 0 or count > len(self.dataset)):
            count = len(self.dataset)

        self.base_classifier.eval()

       

        for i in range(len(self.dataset)):
            if i == count:
                break

            (x, label) = self.dataset[i]
            x = x.to(self.device)
            
            original_cAHat = original_rs_cache.cAHat_arr[i]
            original_correct = int(original_cAHat == label)
            original_pA = original_rs_cache.pA_arr[i]
            before_time = time()
            if (original_correct != 1):
                time_elapsed = datetime.timedelta(seconds=(time() - before_time))
                self.rs_cache.add(original_pA, 0.0, None, None, 0.0, original_cAHat)
                self.rs_cache.add_time_elapsed(time_elapsed.total_seconds())
                print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, original_cAHat,0.0, original_correct, str(time_elapsed)), flush=True)
                continue
            
            
            if original_pA < gamma:
                predictions = original_rs_cache.predictions[i]
                seeds = original_rs_cache.seeds[i]
                cAHat, radius, correct = self.compute_radius_with_chi(x, n_chi, batch_size, alpha_chi, label, seeds, predictions, original_pA)
            else:
                cAHat, radius, correct = self.compute_radius(x, n2, n0, batch_size, alpha1+alpha_chi, label)

            time_elapsed = datetime.timedelta(seconds=(time() - before_time))
            self.rs_cache.add_time_elapsed(time_elapsed.total_seconds())

            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(i, label, cAHat, radius, correct, str(time_elapsed)), flush=True)
            

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts, _, _ = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size: int) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            seeds = []
            preds = []

            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                seeds.append(torch.seed())
                noise = torch.randn(batch.shape).to(self.device) * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)

                preds.append(predictions)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)

            return counts, seeds, preds

    
    def _estimate_chi(self, x: torch.tensor, num: int, batch_size: int, alpha_chi: float, seeds, predictions):
        with torch.no_grad():
            different = 0
            total = num
            counts = np.zeros(self.num_classes, dtype=int)
            for i in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                batch = x.repeat((this_batch_size, 1, 1, 1))
                torch.manual_seed(seeds[i])
                noise = torch.randn(batch.shape).to(self.device) * self.sigma

                # Original predictions are previously computed
                original_predictions = predictions[i]
                perturbed_predictions = self.base_classifier(batch + noise).argmax(1)

                original_predictions = original_predictions.cpu().numpy()
                perturbed_predictions = perturbed_predictions.cpu().numpy()
                different += np.sum(original_predictions != perturbed_predictions)
                counts += self._count_arr(perturbed_predictions, self.num_classes)
            return self._upper_confidence_bound(different, total, alpha_chi), counts
        
    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

    def _upper_confidence_bound(self, NA: int, N: int, alpha: float):
       return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[1]

