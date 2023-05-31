""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
# import setGPU
import sys
from pathlib import Path

from nnverify import util
from nnverify.config import NET_HOME
sys.path.append(str(Path(__file__).resolve().parent))

from datasets import get_dataset, get_num_classes
from core import Smooth
from time import time
import torch
import datetime

# parser = argparse.ArgumentParser(description='Predict on many examples')
# parser.add_argument("dataset", choices=DATASETS, help="which dataset")
# parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
# parser.add_argument("sigma", type=float, help="noise hyperparameter")
# parser.add_argument("outfile", type=str, help="output file")
# parser.add_argument("--batch", type=int, default=1000, help="batch size")
# parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
# parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
# parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
# parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
# parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
# args = parser.parse_args()

class SmoothingArgs():
    def __init__(self, net, dataset, sigma, N0, N, alpha, batch, count=100):
        self.net_location = NET_HOME + str.replace(net, 'sigma', str(sigma)[:4])
        self.dataset = dataset
        self.sigma = sigma
        self.N = N
        self.alpha = alpha
        self.batch = batch
        self.N0 = N0
        self.count = count


class SmoothingAnalyzer():
    def __init__(self, args):
        # load the base classifier
        self.base_classifier = util.get_net(args.net_location, args.dataset).torch_net
        self.args = args
        self.smoothed_classifier = Smooth(self.base_classifier, get_num_classes(args.dataset), args.sigma)
        self.dataset = get_dataset(self.args.dataset, 'test')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def predict(self):
        """Predict on many examples from a dataset."""

        print("idx\tlabel\tpredict\tcorrect\ttime", flush=True)

        # iterate through the dataset
        for i in range(len(self.dataset)):

            # only certify every args.skip examples, and stop after args.max examples
            # if i % self.args.skip != 0:
            #     continue

            if i == self.args.count:
                break

            (x, label) = self.dataset[i]
            x = x.to(self.device)
            before_time = time()

            # make the prediction
            prediction = self.smoothed_classifier.predict(x, self.args.N, self.args.alpha, self.args.batch)

            after_time = time()
            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            # log the prediction and whether it was correct
            print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), flush=True)
    

    def certify(self):
        """Certify the prediction of the base classifier around x"""
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", flush=True)

        # iterate through the dataset
        for i in range(len(self.dataset)):

            # only certify every args.skip examples, and stop after args.max examples
            #if i % self.args.skip != 0:
            #    continue
            if i == self.args.count:
                break

            (x, label) = self.dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.to(self.device)
            prediction, radius = self.smoothed_classifier.certify(x, self.args.N0, self.args.N, self.args.alpha, self.args.batch)
            after_time = time()
            correct = int(prediction == label)

            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, label, prediction, radius, correct, time_elapsed), flush=True)


           
