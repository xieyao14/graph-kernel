# -*- coding: utf-8 -*-
"""
PP model on graph
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
import arrow
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj, to_networkx, degree
from torch_geometric.transforms import LaplacianLambdaMax

from model import ExpBasis, BaseExponentialCosineBasis, TemporalParametricKernelChebnetLocalFilterOnGraph, TemporalParametricKernelL3netLocalFilterOnGraph, TemporalDeepBasisL3netLocalFilterOnGraphKernel, TemporalDeepBasisGATLocalFilterOnGraphKernel, TemporalPointProcessOnGraph
from visualization import plot_fitted_temporal_graph_model
from utils import eval_points_generate, train

import yaml
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-config_yaml_file", type=str)
args = parser.parse_args()
with open(args.config_yaml_file,'r') as f:
    config = yaml.safe_load(f)

    
cwd = os.getcwd()
os.chdir(cwd)

seed = config["seed"]
torch.manual_seed(seed)
np.random.seed(seed)

## model configuration
T = [config["T0"], config["T1"]]
tau_max  = config["tau_max"]
mu       = config["mu"]
n_basis_time = config["n_basis_time"]
n_basis_loc  = config["n_basis_loc"]
data_dim = 2
data_name    = config["data"]
loss_type    = config["loss"]
use_seq      = config["use_seq"]
model_name   = "Model_GAT_%s_for_%s_nbasistime%d_nbasisloc%d_seqs%d" % (loss_type, data_name, n_basis_time, n_basis_loc, use_seq)
save_path    = "results/saved_models/%s" % model_name
G = torch.load("data/%s/graph.pt" % data_name)

device = config["device"]

## true model
trg_model = None


## initialize model
# order_list = list(range(n_basis_loc))
# order_list = [1]
kernel = TemporalDeepBasisGATLocalFilterOnGraphKernel(T=T, G=G, tau_max=tau_max, device=device,
                                        n_basis_time=n_basis_time, n_basis_loc=n_basis_loc,
                                        data_dim=data_dim, basis_dim=1, nn_width_basis_time=32,
                                        init_gain=5e-1, init_bias=1e-2, init_std=1e-2)
init_model = TemporalPointProcessOnGraph(device=device, T=T, G=G, mu=mu*np.ones(G.x.shape[0]), tau_max=tau_max, loss=loss_type,
                                         kernel=kernel, data_dim=data_dim, numerical_int=True,
                                         eval_res=50, int_res=50, int_res_loc=200,
                                         pen_res_time=100, l2_res_time=100)
if config["warm_start"]:
    print("Warm start. Loading model...")
    init_model.load_state_dict(torch.load("results/saved_models/warm_starts/%s-old.pth" % model_name))

## initialize training configuration
ts = np.linspace(T[0], T[1], 50)
ns = np.arange(G.x.shape[0])
eval_points = eval_points_generate(ts, ns)

## data loading
data          = np.load("data/%s/%s.npy" % (data_name, data_name))
data          = torch.FloatTensor(data)
train_data    = data[:int(use_seq*0.8)]
test_data     = data[int(use_seq*0.8):use_seq]
plot_points   = test_data[0]

# training
save_model = True
train_llks, test_llks, test_maes, test_mres, losses, wall_time, lam_mins, ts, bs = train(init_model, trg_model, config, train_data, test_data, eval_points=eval_points, plot_points=plot_points, plot_ngrid=100, modelname=model_name, save_model=save_model, save_path=save_path, load_iter=100)

if save_model:
    np.save("results/saved_models/%s/losses.npy" % model_name, losses)
    np.save("results/saved_models/%s/metrics.npy" % model_name, [train_llks, test_llks, test_maes, test_mres])
    np.save("results/saved_models/%s/wall_times.npy" % model_name, wall_time)
    if config["penalty"]:
        np.save("results/saved_models/%s/lam_mins.npy" % model_name, [lam_mins, ts, bs])