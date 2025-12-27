# -*- coding: utf-8 -*-
"""
PP model on graph
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
import arrow
import os
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj, to_networkx, degree
from torch_geometric.transforms import LaplacianLambdaMax

from visualization import plot_fitted_temporal_graph_model

"""# Utils"""

def lap2mat(edge_index, edge_weight):
    n_node = edge_index.max() + 1
    n_edge = edge_index.shape[1]

    lap_mat = torch.zeros((n_node, n_node))
    for i in range(n_edge):
        lap_mat[edge_index[0, i], edge_index[1, i]] = edge_weight[i]
    
    return lap_mat

def eval_points_generate(ts, ns):
    """
    Generate points for intensity evaluation

    Args:
    - ts: numpy array, eval times
    - ns: numpy array, eval nodes
    """
    return np.array(list(itertools.product(ts, ns)))

def graph_generator(n_node, max_D=4, weight_range=[0., 1.], seed=100):
    
    np.random.seed(seed)
    adjacency_mat = np.zeros((n_node, n_node))
    n_neib   = np.random.randint(low=1, high=max_D+1, size=n_node)
    for i in range(n_node):
        neighbor = np.random.choice(a=np.delete(np.arange(n_node), i, 0), size=n_neib[i], replace=False)
        weight   = np.random.uniform(low=weight_range[0], high=weight_range[1], size=n_neib[i])
        adjacency_mat[i, neighbor] = weight

    edge_index = torch.tensor(np.vstack(np.where(adjacency_mat != 0)), dtype=torch.long)
    edge_weight = torch.tensor(adjacency_mat[np.where(adjacency_mat != 0)], dtype=torch.float)
    node = torch.tensor(np.arange(n_node).reshape(-1, 1), dtype=torch.float)

    G = Data(x=node, edge_index=edge_index, edge_weight=edge_weight)

    return G, adjacency_mat

def chain_graph_generator(n_node, weight_range=[0., 1.], seed=100):
    
    np.random.seed(seed)
    adjacency_mat = np.zeros((n_node, n_node))
    for i in range(n_node-1):
        weight   = np.random.uniform(low=weight_range[0], high=weight_range[1], size=1)
        adjacency_mat[i, i+1] = weight[0]
        adjacency_mat[i+1, i] = weight[0]

    edge_index = torch.tensor(np.vstack(np.where(adjacency_mat != 0)), dtype=torch.long)
    edge_weight = torch.tensor(adjacency_mat[np.where(adjacency_mat != 0)], dtype=torch.float)
    node = torch.tensor(np.arange(n_node).reshape(-1, 1), dtype=torch.float)

    G = Data(x=node, edge_index=edge_index, edge_weight=edge_weight)

    return G, adjacency_mat

def ring_graph_generator_4_nb(n_node, weight_range=[0., 1.], seed=100):
    
    np.random.seed(seed)
    adjacency_mat = np.zeros((n_node, n_node))
    for i in range(n_node):
        weight   = np.random.uniform(low=weight_range[0], high=weight_range[1], size=1)
        adjacency_mat[i, (i+1) % n_node] = weight[0]
        adjacency_mat[(i+1) % n_node, i] = weight[0]

    for i in range(n_node):
        weight   = np.random.uniform(low=weight_range[0], high=weight_range[1], size=1)
        adjacency_mat[i, (i+2) % n_node] = weight[0]
        adjacency_mat[(i+2) % n_node, i] = weight[0]

    edge_index = torch.tensor(np.vstack(np.where(adjacency_mat != 0)), dtype=torch.long)
    edge_weight = torch.tensor(adjacency_mat[np.where(adjacency_mat != 0)], dtype=torch.float)
    node = torch.tensor(np.arange(n_node).reshape(-1, 1), dtype=torch.float)

    G = Data(x=node, edge_index=edge_index, edge_weight=edge_weight)

    return G, adjacency_mat

def generate_ring_pos(n_node):

    delta_theta = np.pi * 2 / n_node
    pos = np.vstack([np.cos(np.arange(n_node) * delta_theta), np.sin(np.arange(n_node) * delta_theta)]).T

    return {i: pos[i] for i in range(n_node)}




"""# Training Functions"""

def config_generate(lr=1e-2,
                    epoch=500,
                    batch_size=64,
                    lam_reg=100,
                    penalty=False,
                    reg=False,
                    mae_eval=False,
                    t_init=1e1,
                    t_upp=1e6,
                    t_mul=1.3,
                    b_bar=10.,
                    b_upp=-5.,
                    device="cpu"):

    if not penalty:
        print("no penalty!")
    if not reg:
        print("no reg!")
    if not mae_eval:
        print("no mae_eval!")

    config = {
        'lr': lr,
        'epoch': epoch,
        'batch_size': batch_size,
        'lam_reg': lam_reg,
        'penalty': penalty,
        'reg': reg,
        'mae_eval': mae_eval,
        't_init': t_init,
        't_upp': t_upp,
        't_mul': t_mul,
        'b_bar': b_bar,
        'b_upp': b_upp,
        'device': device
    }

    return config

def train(model,
            trg_model,
            config,
            train_data,
            test_data,
            eval_points,
            plot_points,
            plot_ngrid,
            lam_ylim=None,
            modelname="pp",
            save_model=True,
            save_path=None,
            load_iter=10):
    """training procedure"""

    if save_model:
        if os.path.exists(save_path):
            print("Duplicated folder!")
            return None
        else:
            print("Create folder!")
            os.makedirs(save_path)

    if trg_model is not None:
        plot_fitted_temporal_graph_model(trg_model,
                                     plot_points,
                                     T_plot=model.T,
                                     ngrid=plot_ngrid,
                                     annotation=False,
                                     lam_ylim=lam_ylim,
                                     filename="True",
                                     savefig=save_model,
                                     savepath=save_path)

    n_events_train = (train_data[:, :, 0] > 0).sum()
    n_events_test = (test_data[:, :, 0] > 0).sum()
    print("[%s] #Training sequences: %d" % (arrow.now(), train_data.shape[0]))
    print("[%s] #Testing sequences: %d" % (arrow.now(), test_data.shape[0]))
    print("[%s] #Training events: %d" % (arrow.now(), n_events_train))
    print("[%s] #Testing events: %d" % (arrow.now(), n_events_test))
    train_loader = DataLoader(torch.utils.data.TensorDataset(train_data),
                              shuffle=True, batch_size=config['batch_size'], drop_last=False)
    test_loader  = DataLoader(torch.utils.data.TensorDataset(test_data),
                              shuffle=False, batch_size=config['batch_size'], drop_last=False)

    # Evaluate synthetic data with True model
    if trg_model is not None and config["mae_eval"]:
        print("[%s] Compute true intensity..." % arrow.now())
        true_lams = []
        for batch in test_loader:
            true_lams.append(trg_model.sample_intensity_parametric(eval_points, batch[0], device="cpu"))
        true_lams = np.concatenate(true_lams, axis=0)                               # [ n_test_seq, len(eval_points) ]


    train_loss = []
    test_loss  = []
    test_maes  = []
    test_mres  = []
    wall_time  = []
    lam_mins   = []
    losses     = []
    ts         = []
    bs         = []

    if config["penalty"]:
        b = -config["b_bar"]

    i = 0
    fea = 0
    bar_flag = 0
    t = config["t_upp"]

    model.to(config["device"])
    optimizer = optim.Adadelta(model.parameters(), lr=config["lr"])

    # Data fitting
    print("[%s] Start Model Learning..." % arrow.now())
    t0 = arrow.now()
    while i < config["epoch"]:
        try:
            epoch_total_loss    = 0
            epoch_model_loss    = 0
            epoch_bar_loss      = 0
            epoch_reg_loss      = 0
            num_overshot        = 0
            lam_min             = np.inf

            for batch in train_loader:
                X         = batch[0].to(config["device"])
                optimizer.zero_grad()
                model_rets = model(X)
                model_loss = model_rets[0]
                reg        = model_rets[1]
                loss      = model_loss

                if config["reg"]:
                    loss = loss + reg * config["lam_reg"]

                if config["penalty"]:
                    if model.loss_type == "cont_l2_loss":
                        lams      = model_rets[2]
                    else:
                        lams      = model.penalty_grid_lams(X)
                    b_temp    = min(lams.min().item() - config["b_bar"], config["b_upp"])
                    if b_temp < b:
                        num_overshot += 1
                    else: b_temp = b
                    bar       = - torch.log(lams - b_temp).mean()
#                     bar       = (- torch.log(torch.clamp(lams - b, min=1e-5))).mean()
                    lam_min   = min(lam_min, lams.min().item())

                    loss      = loss + bar / t

                loss.backward()
                optimizer.step()

                epoch_total_loss += loss
                epoch_model_loss += model_loss
                if config["reg"]:
                    epoch_reg_loss += config["lam_reg"] * reg
                if config["penalty"]:
                    epoch_bar_loss += bar / t

            with torch.no_grad():
                test_model_loss = 0
                for batch in test_loader:
                    X_test      = batch[0].to(config["device"])
                    te_model_rets       = model(X_test)
                    test_model_loss     += te_model_rets[0]
                
                train_e_loss = (epoch_model_loss / n_events_train).item()
                test_e_loss  = (test_model_loss / n_events_test).item()
                if model.loss_type == "likelihood":
                    print("[%s] Epoch : %d,\tTraining llk per event: %.8f" % (arrow.now(), i, -train_e_loss))
                    print("[%s] Epoch : %d,\tTesting llk per event: %.8f" % (arrow.now(), i, -test_e_loss))
                else:
                    print("[%s] Epoch : %d,\tTraining loss per event: %.8f" % (arrow.now(), i, train_e_loss))
                    print("[%s] Epoch : %d,\tTesting loss per event: %.8f" % (arrow.now(), i, test_e_loss))
                train_loss.append(train_e_loss)
                test_loss.append(test_e_loss)

                if config["mae_eval"]:
                    lams_test = []
                    for batch in test_loader:
                        lams_test.append(model.sample_intensity(eval_points, batch[0], device=config["device"]))
                    lams_test = np.concatenate(lams_test, axis=0)
                    test_mae   = np.mean(np.abs(lams_test - true_lams))
                    test_mre   = np.mean(np.abs(lams_test - true_lams) / true_lams)
                    print("[%s] Epoch : %d,\tTesting lams MAE : %.8f" % (arrow.now(), i, test_mae))
                    print("[%s] Epoch : %d,\tTesting lams MRE : %.8f" % (arrow.now(), i, test_mre))
                    test_maes.append(test_mae)
                    test_mres.append(test_mre)

            t_e = arrow.now()
            wall_time.append((t_e - t0).total_seconds())
            
            if model.loss_type == "likelihood":
                logout = "[%s] Epoch : %d, \ttotal_loss : %.8f, \tllk_loss : %.8f" % (arrow.now(), i, epoch_total_loss, epoch_model_loss)
            else:
                logout = "[%s] Epoch : %d, \ttotal_loss : %.8f, \tl2_loss : %.8f" % (arrow.now(), i, epoch_total_loss, epoch_model_loss)
            ret    = [epoch_total_loss.item(), epoch_model_loss.item()]
            if config["reg"]:
                logout = logout + "\treg_loss : %.8f" % (epoch_reg_loss)
                ret.append(epoch_reg_loss.item())
            if config["penalty"]:
                logout = logout + "\tbarrier_loss : %.8f, min_lam : %.8f, num_oveshot : %d, t : %.5f, b : %.5f" % (epoch_bar_loss, lam_min, num_overshot, t, b)
                ret.append(epoch_bar_loss.item())

            print(logout)
            losses.append(ret)

            if (i+1) % load_iter == 0:
                model.cpu()
                plot_fitted_temporal_graph_model(model,
                                                plot_points,
                                                T_plot=model.T,
                                                ngrid=plot_ngrid,
                                                annotation=False,
                                                lam_ylim=lam_ylim,
                                                filename="Epoch %d" % i,
                                                savefig=save_model,
                                                savepath=save_path)

                if save_model:
                    torch.save(model.state_dict(), "%s/%s-%d.pth" % (save_path, modelname, i))

                model.to(config["device"])

            if config["penalty"]:
                lam_mins.append(lam_min)
                ts.append(t)
                bs.append(b)
                if lam_min < config["b_bar"] + config["b_upp"]:
                    t = config["t_init"]
                    b = lam_min - config["b_bar"]
                    bar_flag = 1
                else:
                    t = min(t*config["t_mul"], config["t_upp"])
                    b = config["b_upp"]

            num_overshot = 0
            i += 1

        except KeyboardInterrupt:
            break

    return train_loss, test_loss, test_maes, test_mres, losses, wall_time, lam_mins, ts, bs