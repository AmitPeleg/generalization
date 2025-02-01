# Bias of Stochastic Gradient Descent or the Architecture: Disentangling the Effects of Overparameterization of Neural Networks

#### [project page](https://bias-sgd-or-architecture.github.io) | [paper](https://arxiv.org/abs/2407.03848) 
PyTorch implementation of the paper "Bias of Stochastic Gradient Descent or the Architecture: Disentangling the Effects of Overparameterization of Neural Networks", ICML 2024.

> Bias of Stochastic Gradient Descent or the Architecture |
> [Amit Peleg](mailto:amit.peleg@uni-tuebingen.de), [Matthias Hein](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/maschinelles-lernen/team/prof-dr-matthias-hein/) |
> ICML, 2024

Our implementation is based on the paper "Loss Landscapes are All You Need: Neural Network Generalization Can Be Explained Without the Implicit Bias of Gradient Descent" (ICLR 2023) and their [github repository](https://github.com/Ping-C/optimizer).

## Installation
Clone this repository
```bash
git clone https://github.com/AmitPeleg/generalization
cd generalization/
```

To create a conda environment:
```bash
conda env create -f environment.yml
conda activate generalization
```

## Configs

The different configurations to reproduce the figures in the paper are in the [`configs`](configs) folder. The configuration files are in `yaml` format. The configuration files are named according to the figure they reproduce. For example, the configs in [`configs/fig1/`](configs/fig1/) reproduce Figure 1 in the paper.

## Experiments
### Single experiment
To reproduce a single experiment in the paper, run the following,

```bash
python train.py --config configs/path/to/config.yaml
```

The results for this config will be saved in the [`output`](output) folder.

The [`train.py`](train.py) script always runs on a single GPU (cuda:0). 
You can choose the GPU by setting the environment variable `CUDA_VISIBLE_DEVICES`.
It is also possible to run the same config on multiple GPUs by launching the script multiple times with different `CUDA_VISIBLE_DEVICES`.
Running on multiple GPUs will produce the same results as on a single GPU.

### Entire figure
To reproduce an entire figure, run the following, 
```bash
python run_exps.py <fig_name>
```
A list of figure names can be found in the [`configs`](configs) folder.

The outputs/errors of each experiment are saved in the [`logs`](logs) folder, and the results are stored in the [`output`](output) folder.

Each config runs on a single GPU. 
The [`run_exps.py`](run_exps.py) script launches the scripts of a specific figure using the available GPUs.
If you don't want to use all GPUs, do not set the `CUDA_VISIBLE_DEVICES` environment variable, but rather set `GPU_LIST` in the [`run_exps.py`](run_exps.py) script (line 73).
To better utilize each GPU, you can adjust the batch size in the config files.

### Provided results
We provide the results for [Figure 1-5 for the mnist dataset and lenet architecture](https://nc.mlcloud.uni-tuebingen.de/index.php/s/fzc7xj8mp6YDc4Q).
The results should be downloaded and extracted to the [`output`](output) folder.
They can also be downloaded directly,
```bash
mkdir output
cd output
wget https://nc.mlcloud.uni-tuebingen.de/index.php/s/fzc7xj8mp6YDc4Q/download/fig_1_to_5_lenet_mnist.zip
unzip fig_1_to_5_lenet_mnist.zip
```

## Creating figures

After running the experiment, to plot the figure, run the following,

```bash
python -m create_figs.run_figs <fig_name>
```
A list of figure names can be found using `python -m create_figs.run_figs --help`.
There may be slight differences between the provided results and the paper due to code reorganization and changes in the randomization seed.

## Citation
```
@inproceedings{Peleg2024ICML_Bias_of_Stochastic,
  author={Amit Peleg and Matthias Hein},
  title={Bias of Stochastic Gradient Descent or the Architecture: Disentangling the Effects of Overparameterization of Neural Networks},
  booktitle={ICML},
  year={2024}
}
```
