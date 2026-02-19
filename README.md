# SEE-OOD: Supervised Exploration for Enhanced Out-of-Distribution Detection

This repository contains the official implementation for the paper "SEE-OoD: Supervised Exploration for Enhanced Out-of-Distribution Detection". In particular, this repository provides source code and commands for reproducing results for comparison experiments (Section 5) and numerical examples (Section 4). For the case study in the paper, we provide the source code for model architecture and training but not the actual dataset we used. However, the case study dataset can be requested via email to the authors.

### Keywords: Machine Learning, Out-of-Distribution Learning, Generative Adversarial Networks (GANs), etc.

## Repository Structure Overview

Here we provide an overview of the structure of this repository and the functionality of each important folder or file.

- The `config/` folder consists of `.yaml` configuration files used for all experiments, including SEE-OOD training configurations and OoD data sampling configurations.

- The `checkpoint/` folder is designed to collect training checkpoints and log files, as well as pre-sampled OoD samples.

- The `Document/` folder contains figures or results plots for all experiments.

- The `main/` and `models/` folders consist of relevant driver code and implemented model architectures, respectively.

## Environment Configuration

The development OS and key cloud server environments for this codebase are outlined below. It is **highly recommended** to reproduce the results on HPCs or Linux-based system because the experiments are extensive and the experiment automation is more compatible with those systems.

```
Red Hat Enterprise Linux (RHEL) 8.10
python>=3.9.7
pytorch>=1.12.1
CUDA>=12.8.0
CUDA Driver Version=570.124.06
```

Although the code is developed on a High-Performance Cluster (HPC) with RHEL system, it should work smoothly without any difficulties on a Linux-based OS. For Windows system, we recommend running on a WSL system like Ubuntu in order to do full automation of the experiments. Here we provide detailed instructions on setting up the environments.

### Prerequisite

Before setting up the environment, please make sure you have the following prerequisite software or package management libraries downloaded.

- **Anaconda (or Miniconda)**: environment and package management system.
- **pip**: python package management library.
- **Ubuntu 22.04** (For Windows users only): open-source Linux operating system, required to use WSL (Windows Subsystem for Linux) in Windows computer.

Note that when downloading those packages, please download them such that they are in compatible with your systems and computer configurations.

### Environment Setup

For your ease, we provide the following setup commands (it is also wrapped into the `setup.sh` script):`

```
conda create -n SEEOOD python=3.9.7
conda activate SEEOOD

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install pandas
pip install matplotlib
pip install scikit-learn
pip install geomloss
pip install icecream
pip install tqdm
pip install tensorboard
pip install torchinfo
pip install PyYAML
```

### Remarks, Tips, and Troubleshooting

For all users, please make sure that you activate the conda environment before running the code below. For Windows users, it is highly recommended to use WSL with Ubuntu 22.04 to set up and run all the code. If you are using native Windows Powershell and Git Bash, it may not be compatible with all the automation `.sh` code but should still work with all the python commands. The fixing strategy will be copying and pasting all commands from the `.sh` automation script and run them individually in the terminal.

In addition, Linux and Windows systems have different default file formatting. For Windows users, if you encountered errors like `$'\r': command not found` or `No such file or directory: 'checkpoint/log/FashionMNIST\r/log-8.txt` because of the existing of hidden character `\r`, you can resolve this by modifying the `.sh` automation script in your WSL:

```
sudo apt install dos2unix # installation (only run once)
dos2unix ****.sh # Just replace **** with the `.sh` file you run
```

Users with our recommended Linux OS or HPCs should not have this issue as the files and code are developed on the same system.

## Running the Code

In general, the order and commands for training and evaluating the proposed SEE-OOD model is provided below.

0. **Note:** we have provided all the experimental results and checkpoints in this repository, so if you only want to reproduce the results from the checkpoints, you may directly go to step 4 and run the commands without any problem. However, if you have run steps 1 - 3 to reproduce from scratch, you must wait until it finishes to run step 4.

1. **Setup:** Determine experimental setup and configuration, including which regime and the number of OoD samples exposed during training. Below, we use Regime-I _FashionMNIST_ experiment with 32 observed OoD samples as an example.

2. **Sampling:** Modify the relevant `sampling-fashionmnist.yaml` file in the `config/sampling/` folder and run:

```
cd main/
python3 sample_ood.py --config=../config/sampling/sample-fashionmnist.yaml --n_ood=32
```

Note that we can also choose to sample every possible setting at first. This is provided in `main/sample_ood.sh` file to automate this process; you may find commands inside. A random seed is set so that the reproducibility is guaranteed.

3. **Training & Evaluation:** Setup training configuration by modifying relevant `.yaml` file in `config/GAN/` folder. In this example, the relevant file is `OOD-GAN-FashionMNIST.yaml`. Inside this file, we can adjust the training parameters. Then, to run experiments with 32 OoD samples observed for each class (i.e., regime I), please use the following command:

```
cd ..
bash jobs/FashionMNIST/32.sh
```

We have also provided an overall running script named `jobs/run.sh` where all relevant commands for all experimental settings can be found. After this step, the results can be found in the file `checkpoint/log/FashionMNIST/log-32.txt`.

For all other experiments, the running procedure follows exactly the same as above. The only difference is the configuration files to be modified.

4. **Summarizing Results & Visualization (Optional):** After ALL experiments for ALL settings are finished, the results can be summarized by running:

```
bash jobs/summarize.sh > checkpoint/log/summary.txt
```

And to visualize the results using figures, please run:

```
bash jobs/get_results.sh
```

Note that this step is considered as the very last step of the experiments. ALL experiments for ALL datasets mentioned in Section 5 of the paper need to be finished before this step; in other words, this is just a complementary step to transform `.txt` log files to plots.

### Remarks, Tips, and Troubleshooting

Note that our experiments are GPU-hungry and are time-consuming for the training process. Depending on different hardwares, the runtime spans from around 30 minutes to around 3 hours for **each setting**.

For each automation script, you may always modify it to choose which experiment to run. We leave the users flexibility for whether or not to automating the entire process.

## Numerical Illustration & Case Study

For numerical illustration, it is highly advised to run the code on CPU because running on GPU for small batch size will usually result in unnecessary resource waste. To simplify, we provide those pre-trained checkpoints and the results can be visualized by running:

```
python get_results.py --sim --setting=I
python get_results.py --sim --setting=II
```

The results will be saved in the `Document/` folder. To generate data and train from scratch, the commands are provided as follow:

```
bash jobs/simulation/simulation.sh
```

As for the case study, the dataset is not disclosed at this moment so it has to be requested by emailing the authors. After the dataset is obtained, the experiments can be done following the exact same procedure mentioned in the preceding section.

## Baseline Methods

This paper compares the proposed method with 11 different baseline methods. For implementation of these baseline methods, we choose not to re-implement ourselves but utilize the original implementations for details. Here we provide links to the official implementations to those baseline methods:

- **MSP, ODIN, & MAHA:** https://github.com/pokaxpoka/deep_Mahalanobis_detector
- **VOS:** https://github.com/deeplearning-wisc/vos
- **WOOD:** https://github.com/wyn430/WOOD
- **GAN-Synthesis (Confident Classifier):** https://github.com/alinlab/Confident_classifier
- **Energy & Energy Finetuning:** https://github.com/wetliu/energy_ood
- **Outlier Exposure:** https://github.com/hendrycks/outlier-exposure
- **ATD:** https://github.com/rohban-lab/ATD
- **DeepSAD:** https://github.com/lukasruff/Deep-SAD-PyTorch

## 📝 Notes & Suggestions

It is highly recommended to run these experiments on a high-performance cluster (HPC) in parallel because the experiments in this paper are enormous and are often of large-scale. To run on HPC, it is advised to modify each `*.sh` command file in the `jobs/` folder to submittable bash scripts according to the requirements of the HPC that you have the access to. In this repo, we do not provide according modification as different HPC platform has different format; however, the results consistency won't be affected as long as the python command remains unchanged.
