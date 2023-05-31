# Incremental Neural Network Verifiers

Verifier runs through the unittest framework. A new unit test can be added to run the verifier with a specific 
configuration. 
Current unit tests are located in `nnverify/tests`. 

Networks used for the experiment are available [here](https://drive.google.com/drive/folders/1cPBdu2L1ctUszRw5KV69i5EsotRlINS3?usp=sharing)

## Publications

[Incremental Verification of Neural Networks](https://arxiv.org/abs/2304.01874)\
Shubham Ugare, Debangshu Banerjee, Sasa Misailovic, and Gagandeep Singh\
<strong> PLDI 2023 </strong>


## Installation

<details><summary> Instructions for IVAN </summary>
<p>

### Step 1: Installing Gurobi

GUROBI installation instructions can be found at `https://www.gurobi.com/documentation/9.5/quickstart_linux/software_installation_guid.html`

For Linux-based systems the installation steps are:
Install Gurobi:
```
wget https://packages.gurobi.com/9.1/gurobi9.1.2_linux64.tar.gz
tar -xvf gurobi9.1.2_linux64.tar.gz
cd gurobi912/linux64/src/build
sed -ie 's/^C++FLAGS =.*$/& -fPIC/' Makefile
make
cp libgurobi_c++.a ../../lib/
cd ../../
cp lib/libgurobi91.so /usr/local/lib -> (You may need to use sudo command for this)   
python3 setup.py install
cd ../../
```

Update environment variables:
i) Run following export commands in command prompt/terminal (these environment values are only valid for the current session) 
ii) Or copy the lines in the .bashrc file (or .zshrc if using zshell), and save the file 

```
export GUROBI_HOME="$HOME/opt/gurobi950/linux64"
export GRB_LICENSE_FILE="$HOME/gurobi.lic"
export PATH="${PATH}:${GUROBI_HOME}/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/$HOME/usr/local/lib:/usr/local/lib
```

Getting the free academic license
To run GUROBI one also needs to get a free academic license. https://www.gurobi.com/documentation/9.5/quickstart_linux/retrieving_a_free_academic.html#subsection:academiclicense

a) Register using any academic email ID on the GUROBI website. b) Generate the license on https://portal.gurobi.com/iam/licenses/request/

Choose Named-user Academic


c)Use the command in the command prompt to generate the licesne. 


(If not automatically done, place the license in one of the following locations “/opt/gurobi/gurobi.lic” or “$HOME/gurobi.lic”)

### Step 2: Installing Python dependencies in a virtual environment

First, make sure you have venv (https://docs.python.org/3/library/venv.html).
If venv is not already installed, install it with the following command (Use appropriate python version)

`sudo apt-get install python3.8-venv`

(One can also use other environments such as conda, however we have not tested the experiments on other Python environments) 

To create the virtual environment,

`python3 -m venv env`

Then to enter the virtual environment, run

`source env/bin/activate`

Install all packages including the compiler with

`pip install -r requirements.txt` 

Even if installation of any of the libraries does not work, ignore and continue with the next steps

### Step 3: Running experiments 

Caveats:
1. The speedup results obtained in the experiment can vary depending on the machine

2. Our experiments run our tool IVAN and the baseline on a fixed number of randomly chosen inputs from the dataset. We report the average speedup on each verification instance. The speedup results in the paper are for count=100. One can change the count=20 to a smaller value for faster run of all experiments. However, the average speedup result may vary depending on this value.
 
3. Speedups are also dependent on the timeout used for verification. To accurate reproduce the results from the paper, we advise not changing those timeout values otherwise the observed speedups can be less or more than the ones reported in the paper.

Instructions for running experiments:
A single experiment runs IVAN and the baseline for verifying a set of properties on a fixed network and fixed type of modification.  

One can run a single experiment from the test using the following command. This will take about 1 to 2 hours. 

`python3 -m unittest -v nnverify.tests.test_pldi.TestIVAN.test1`

Running the experiment will result in following console output

The following part of the console output present the verifier name, network name, number of inputs in the experiment and the timeout for the verification.
```
test1 (nnverify.tests.test_pldi.TestIVAN) ... Running IVAN on the original network
Running on the network:  nnverify/nets/mnist-net_256x2.onnx
Number of verification instances:  100
Timeout of verification:  100
```

There are 4 possible outcomes of verification that are also printed on the console

    `VERIFIED` - The robustness property is verified
    `ADV_EXAMPLE` - The verifier found a adversarial example for the property
    `MISS_CLASSIFIED` - DNN missclassified the input. This is a trivial counter-example for the property
    `UNKNOWN` - The verifier timed out
    

All experiments in the paper consider multiple combinations of networks and network perturbations for evaluating IVAN’s effectiveness in verification compared to the baseline. The goal of experiments is to compute IVAN’s speedup over the baselines. 
 
All experiment unit tests are located in nnverify/test_pldi.py. All tests can be run using the following command. 

`python3 -m unittest -v nnverify.tests.test_pldi.TestIVAN`

The total running time for these tests is about 20 hours. Table 2, Figures 5, and 6 are the results of these experiments. 


How to see the speedup results
The results are printed at stdout. But it is easier to check them in the file results/proof_transfer.csv at the end of the execution. In The csv file each experiment result is summarized in 3 lines, including speedup and extra properties verified. 



This includes information about 
a) time of the experiment b) network name c) network update type d) Number of verification instances e) time taken by IVAN and baseline f) proof tree size by IVAN and the baseline g) extra properties verified by IVAN compared to the baseline  


Results with following details are also pickled in results/pickle/ directory. This includes for i) time taken ii) verification output iii) proof tree size

The ablation study experiments (Table 2) from the paper are also included in the same file. One can run those experiments in the following cases:


Reuse →  `python3 -m unittest -v nnverify.tests.test_pldi.TestReuse`

Reorder → `python3 -m unittest -v nnverify.tests.test_pldi.TestReorder`
  

Similar to the previous case, the runtime is roughly 20 hours and can be made smaller by decreasing the count of verification instances.


Hyperparameter sensitivity experiments (Figure 8) can be performed using  	

`python3 -m unittest -v nnverify.tests.test_pldi.TestSensitivity`


## Adding New Experiments


Similar to existing experiments one can easily add new experiments using a unit test. One can add this test in existing test file nnverify/test_pldi.py or can create a new test file. 

More information about the adding unittests in python is available here https://docs.python.org/3/library/unittest.html. 

A test function looks like following

```
    def test_new(self):
        args = pt.TransferArgs(
            net=’mnist_0.1.onnx’, 
            domain=Domain.LP, 
            split=Split.RELU_ESIP_SCORE,
            approx=ap.Quantize(ap.QuantizationType.INT8)
            dataset=Dataset.MNIST,    
            eps=0.02,
            count=100, 
            pt_method=IVAN(0.003, 0.003),
            timeout=100)
        pt.proof_transfer(args)
```

Here, 

`net`: location of the onnx or torch network. The networks should be placed in nnverify/nets directory. 

`domain`: The abstract domain used by the verifier. The paper experiments are with LP and DeepZ domains. We have recently added more domains. 

`split`: The branching heuristic used

`approx`: The modification performed on the network. 

`dataset`: Includes MNIST, CIFAR10, and ACAS-XU

`eps`: Robustness radius

`count`: Number of verification properties

`timeout`: timeout for each verification instance

`pt_method`: This is to choose the exact incremental verification technique used in the paper. `IVAN(alpha, beta)` combines all the main techniques. `REORDERING(alpha, beta)` uses just the reordering technique, whereas `REUSE` uses just the reuse technique. The latter 2 were used for the ablation study. alpha and beta are hyperparameters that can be tuned for better results. 



</p>
</details>
<details><summary> Instructions for IRS </summary>
 <p>

### Incremental Randomized Smoothing Certification
  This repository contains the code for the paper [Incremental Randomized Smoothing Certification](). The models used for the experiments can be found [here](). 
  
  Incremental Randomized Smoothing (IRS) is the first approach for efficient incremental robustness certification for randomized smoothing(RS).  Given an original network $f$ and its smoothed version $g$, and a modified network $f^p$ with its smoothed version $g^p$, IRS incrementally certifies the robustness of $g^p$ by reusing the information from the execution of RS certification on $g$. The figure below presents the high-level workflow of IRS.
  
  <img width="1014" alt="workflow" src="https://github.com/shubhamugare/nn_verify/assets/68882529/10ac3af7-93db-4701-9dc5-ddb252bc4d4c">

  
  IRS takes the original classifier $f$ and input $x$. It is built on top of the standard RS framework. IRS reuses the $\underline{pA}$ and $\underline{pB}$ estimates computed for $f$ on $x$ by RS. IRS estimate $ζx$, the upper bound on the probability that outputs of $f$ and $f^p$ are distinct, from $f$ and $f^p$. For the smoothed classifier $g^p$ obtained from the updated classifier $f^p$ it computes the certified radius by combining $\underline{pA}$ and $\underline{pB}$ with $ζx$.
  
  We extensively evaluate the performance of IRS on state-of-the-art models on CIFAR10 (ResNet-20,ResNet-110) and ImageNet (ResNet-50) datasets, considering several common approximations such as pruning and quantization. Our results demonstrate speedups of up to 3x over the standard non-incremental RS baseline, highlighting the effectiveness of IRS in improving certification efficiency.
  
### Installation and Setup in a Virtual Environment
  #### Step 1: Virtual Environment Setup
  ##### Conda 
  First, make sure you have conda. Refer to the following link for an installation guide. https://conda.io/projects/conda/en/latest/user-guide/install/index.html
  To create the virtual environment,
  
  `conda create --name env`
  
  Then to enter the virtual environment, run
  
  `conda activate env`
  
  ##### venv
  
  In addition to Conda, venv (https://docs.python.org/3/library/venv.html) can be used to run IRS in a virtual environment. 
  If venv is not already installed, install it with the following command (Use appropriate python version)

  `sudo apt-get install python3.8-venv` 

  To create the virtual environment,

  `python3 -m venv env`

  Then to enter the virtual environment, run

  `source env/bin/activate`

  
  #### Step 2: Dependencies Installation
  Following the creation of the conda environment or venv, install all packages with
  
  `pip install -r requirements.txt`
  
  Finally run, 
  
  `pip install --upgrade torch torchvision`
  
  #### Step 3: Downloading Trained Models
  Download the networks used for the experiment [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4).
  
  Then, move the networks to `nnverify/nets`.
  
  Finally, create an environment variable for ImageNet path with the following command.
  
  `export IMAGENET_DIR= $PATH_TO_IMAGENET`
  
### Running experiments
  #### Caveats
   1. The speedup results obtained in the experiment can vary depending on the machine.
  
   2. Our experiments run the Incremental Randomized Smoothing (IRS) algorithm and baseline randomized smoothing approach on a range of sample sizes. We compute and report the average speedup in certification time for a particular average certified radius (ACR). The speedup results in the paper are for sample values from {1%, … 10%} of the sample size used to certify the original network, enabling a fast recertification while maintaining a sufficiently high ACR. Users can change the range of the sample values provided they have a different sample budget. However, IRS is more advantageous when the chosen sample range is small, enabling a more efficient recertification. 
  
  3. The speedup results reported in the paper are for $count=500$ images. One can change to a smaller value for faster run of all experiments. However, the average speedup result may vary depending on this value.
  
 4. For the main experiments in the paper, we used $N = 10^4$ samples for the certification of the original network. However, this parameter can be modified depending on the user’s sample budget. As shown in the paper, increasing the value of $N$ (ie. to $10^5$ or $10^6$) results in greater speedup for IRS over baseline recertification from scratch. 
  
  5. Int8 quantization is run on CPU as current dynamic quantization in PyTorch only supports CPU. Consequently, experiments with int8 quantization take longer time to run than other approximations, which run in GPU.
  
  #### Instructions for Running Main Experiments
  All experiments are found in `nnverify/tests/test_incremental_smoothing.py`
  
  A single experiment runs the certification of the original network using randomized smoothing, the certification of the modified network  using incremental randomized smoothing, and the certification of the modified network using randomized smoothing from scratch for a particular approximation and $\sigma$.

 One can run a single experiment from the test using the following command. 
 
 `python -m unittest -v nnverify.tests.test_incremental_smoothing.irs_plot.test_cifar10_resnet110_int8_sigma1`
  
  All experiments in the paper consider multiple combinations of networks and network perturbations for evaluating IRS’ effectiveness in verification compared to the baseline. The goal of the experiments is to compute IRS’ speedup over the baselines.

 Tests across all approximations for a particular network can be run using the following command. 
  
  `python -m unittest -v nnverify.tests.test_incremental_smoothing.irs_plot.test_cifar10_resnet110_sigma1`

 For more consistent results, certification can be run for multiple trials with error bars using the following command.
  
  `python -m unittest -v nnverify.tests.test_incremental_smoothing.irs_plot.test_cifar10_resnet110_int8_sigma1_eb`
 
 In the paper, Figures 3 through 12 and Tables 1, 3, 5, 8, and 9 are the results of these experiments. 
  
 #### How to See the Speedup Results
  The average speedup range is printed at stdout. Detailed results can be viewed in the `results/` directory created when running an experiment. The `results/` folder contains the following for each combination of network, approximation, and $\sigma$:
  1. A txt file of the average speedup in certification time of certifying a particular ACR between IRS and baseline. 
  
  2. Plots of average certified radius vs average certification time, certification accuracy vs average certification time, average certified radius vs sample size, and certification accuracy vs sample size for IRS and baseline. 
  
  3. Plots of the distribution of $\underline{pA}$ values for the perturbed network certified with baseline randomized smoothing.
  
  4. A summary.txt file with the ACR, average $ζx$ values, average certification times, and proportion of $\underline{pA}$ of the original network greater than the threshold parameter $\gamma$.
 
  5. Subfolders of numpy arrays of certification time, average certified radius, and $ζx$ values for each sample size

  6. Pickle files of the result for each sample size
 
  7. A log file with all the console output
 
 #### Other Experiments
  Sensitivity experiments (Table 4) for changing the sample size $N$ can be run using
  
   `python -m unittest -v nnverify.tests.test_incremental_smoothing.irs_plot.n_ablation_cifar10_resnet110`

  Sensitivity experiments for the threshold parameter $\gamma$ (Table 5) can be performed using
  
   `python -m unittest -v nnverify.tests.test_incremental_smoothing.hyperparam.grid_cifar10_resnet110_int8_sigma05`
  
  Tables for standard accuracy of original and modified networks (Table 6) can be generated with 
  
  `python -m unittest -v nnverify.tests.test_incremental_smoothing.accuracy.standard_acc_cifar10_resnet110`
  
  Tables for smoothed accuracy of original and modified networks (Table 7) can be generated with
  
  `python -m unittest -v nnverify.tests.test_incremental_smoothing.accuracy.smoothed_acc_cifar10_resnet110`
  
### Analyzing Stored Results
  `nnverify/smoothing/code/irs_analyze.py` enables the user to run analysis on stored results from the IRS experiments. For instance, the following command can be useed to plot the visualizatiions and generate tables for a particular network, approximation, and $\sigma$.
  
  `python3 nnverify/smoothing/code/irs_analyze.py --dataset CIFAR10 --net resnet110 --approximation int8 --sigma 1.00`
  
  | Parameter Name        | Type           | Description  |
  | ------------- |:-------------:| -----:|
  | `dataset`      | str, choices = 'ImageNet', 'CIFAR10' | The dataset the network was trained on |
  | `net`     | str, choices = 'resnet20', 'resnet50', 'resnet110'      | The torch network. The networks should be placed in nnverify/nets directory. |
  | `approximation`     | str, choices = 'all, 'int8', 'fp16', 'bf16', 'prune5', 'prune10', 'prune20'      | The modification performed on the network |
  | `sigma`     | float      | The noise hyperparmeter used to train and certify the network |
  | `batch`     | int, default = 100      |   Batch Size |
  | `N`     | int, default = 10000      | Number of Samples Used to Certify the Original Network |
  | `N2`     | int, default = 1000      | Number of samples used to certify the modified network using baseline randomized smoothing |
  | `Nchi`     | int, default = 1000      | Number of samples used to certify the modified network with incremental randomized smoothing |
  | `count`     | int, default = 500      | Number of Images to Run Certification On |
  | `alpha1`     | float, default = 0.001      | Randomized Smoothing Failure Probability |
  | `alpha_chi`     | float, default = 0.001      | Chi Estimation Failure Probability |
  | `plot`     | bool, default = True      | Whether or not to plot the stored data |
  | `std_acc`     | bool, default = True      | Whether or not to compute the standard accuracy of the network |
  | `smoothed_acc`     | bool, default = True      | Whether or not to compute the smoothed accuracy of the network |
  | `chi`     | bool, default = True      | Whether or not to compute the mean chi values |
  
### Adding New Experiments
  Similar to existing experiments one can easily add new experiments using a unit test. One can add this test in existing test file `nnverify/tests/test_incremental_smoothing.py` or can create a new test file in `nnverify/tests/`.
  
 More information about the adding unittests in python is available here 
  
  https://docs.python.org/3/library/unittest.html.
 
 A test function looks like following
  
 ```python
  def test_new_testcase(self):
   self.certify_for_each_sample_size_eb(net = config.CIFAR_RESNET_110, dataset = Dataset.CIFAR10, 
   approximation = ap.Quantize(ap.QuantizationType.INT8), N = 10000, sigma = 1.00, reapeats = 3)
  ```
  
  ##### .certify_for_each_sample_size_eb()
  | Parameter Name        | Type           | Description  |
  | ------------- |:-------------:| -----:|
  | `dataset`      | str, choices = 'ImageNet', 'CIFAR10' | The dataset the network was trained on |
  | `net`     | str, choices = 'resnet20', 'resnet50', 'resnet110'      | The torch network. The networks should be placed in `nnverify/nets` directory. |
  | `approximation`     | str, Default = None. choices = None, 'int8', 'fp16', 'bf16', 'prune5', 'prune10', 'prune20'      | The modification performed on the network. Default runs all approximations. |
  | `sigma`     | float, default = 1.00      | The noise hyperparmeter used to train and certify the network |
  | `N`     | int, default = 10000      | Number of Samples Used to Certify the Original Network |
  | `reapeat`     | int, default = 3      | Number of Trials to Run Certification |
    
</p>
</details>

