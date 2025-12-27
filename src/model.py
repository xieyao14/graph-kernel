# -*- coding: utf-8 -*-
"""
PP model on graph
"""


import numpy as np
import itertools
from abc import ABC, abstractmethod
import arrow

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj, to_networkx, degree
from torch_geometric.transforms import LaplacianLambdaMax

from utils import lap2mat

"""# Proposed Model

## Kernel Basis
"""

class DeepNetworkBasis(torch.nn.Module):
    """
    Deep Neural Network Basis Kernel

    This class directly models the kernel-induced feature mapping by a deep 
    neural network.
    """
    def __init__(self, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-3, nn_width=5):
        """
        Args:
        - data_dim:  dimension of input data point
        - basis_dim: dimension of basis function
        - nn_width:  the width of each layer in NN
        """
        super(DeepNetworkBasis, self).__init__()
        # configurations
        self.data_dim  = data_dim
        self.basis_dim = basis_dim
        # init parameters for net
        self.init_gain   = init_gain
        self.init_bias   = init_bias
        # network for basis function
        self.net = torch.nn.Sequential(
            # torch.nn.Linear(data_dim, nn_width),  # [ data_dim, n_hidden_nodes ]
            # torch.nn.ReLU(), 
            torch.nn.Linear(data_dim, nn_width),  # [ data_dim, n_hidden_nodes ]
            torch.nn.Softplus(beta=100), 
            # torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            # torch.nn.ReLU(), 
            torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            torch.nn.Softplus(beta=100),                  
            # torch.nn.Linear(nn_width, nn_width),  # [ n_hidden_nodes, n_hidden_nodes ]
            # torch.nn.Softplus(), 
            torch.nn.Linear(nn_width, basis_dim), # [ n_hidden_nodes, basis_dim ]
            # torch.nn.Softplus(beta=1)
            # torch.nn.Sigmoid()
        )
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        """
        initialize weight matrices in network
        """
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight, gain=self.init_gain)
            m.bias.data.fill_(self.init_bias)

    def forward(self, x):
        """
        customized forward function returning basis function evaluated at x
        with size [ batch_size, data_dim ]
        """
        return self.net(x) * 1 # * 2 - 1                                        # [ batch_size, basis_dim ]


class GraphLocalFilterBasis_Chebnet(torch.nn.Module):
    """
    Basis on graph implemented by graph local filter
    """
    def __init__(self, B, learnable=False):
        """
        Args:
        - B: Graph local filter, can be learnable or not.
        """
        super(GraphLocalFilterBasis_Chebnet, self).__init__()
        # configurations
        self.B               = torch.nn.Parameter(B, requires_grad=learnable)
        self.mask            = torch.nn.Parameter((B != 0.).float(), requires_grad=False)

    def get_filter(self):
        return self.B * self.mask

    def forward(self, x, y):
        """
        Args:
        - x:  [ batch_size, 1 ]
        - y:  [ batch_size, 1 ]
        """
        return self.get_filter()[x, y]                                       # [ batch_size, 1 ]


class GraphLocalFilterBasis_L3net(torch.nn.Module):
    """
    Basis on graph implemented by graph local filter
    """
    def __init__(self, B, learnable=True):
        """
        Args:
        - B: Graph local filter, can be learnable or not.
        """
        super(GraphLocalFilterBasis_L3net, self).__init__()
        # configurations
        self.B               = torch.nn.Parameter(B, requires_grad=learnable)
        self.mask            = torch.nn.Parameter((B != 0.).float(), requires_grad=False)

    def get_filter(self):
        return self.B * self.mask

    def forward(self, x, y):
        """
        Args:
        - x:  [ batch_size, 1 ]
        - y:  [ batch_size, 1 ]
        """
        return self.get_filter()[x, y]                                       # [ batch_size, 1 ]


class GraphLocalFilterBasis_GAT(torch.nn.Module):
    """
    Basis on graph implemented by graph local filter
    """
    def __init__(self, B, learnable=True):
        """
        Args:
        - B: Graph local filter, can be learnable or not.
        """
        super(GraphLocalFilterBasis_GAT, self).__init__()
        # configurations
        self.B               = torch.nn.Parameter(B, requires_grad=learnable)
        self.mask            = torch.nn.Parameter((B != 0.).float(), requires_grad=False)

    def get_filter(self):
        norm_f   = torch.nn.functional.softmax(self.B, dim=0)
        masked_f = norm_f * self.mask
        masked_f = masked_f / masked_f.sum(0)
        return masked_f

    def forward(self, x, y):
        """
        Args:
        - x:  [ batch_size, 1 ]
        - y:  [ batch_size, 1 ]
        """
        return self.get_filter()[x, y]                                       # [ batch_size, 1 ]


"""## Kernel"""

class TemporalDeepBasisChebnetLocalFilterOnGraphKernel(torch.nn.Module):
    """
    Temporal Basis Kernel with Chebnet local filter on graph.
    """
    def __init__(self, device, tau_max, T, G,
                 n_basis_time, n_basis_loc, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-1, init_std=1e1,
                 nn_width_basis_time=5, graph_normalize="sym"):
        """
        Arg:
        - G: torch_geometric object
        """
        super(TemporalDeepBasisChebnetLocalFilterOnGraphKernel, self).__init__()
        # configurations
        self.device          = device
        self.n_basis_time    = n_basis_time
        self.n_basis_loc     = n_basis_loc
        self.data_dim        = data_dim
        self.basis_dim       = basis_dim
        self.tau_max         = tau_max
        self.T               = T
        self.n_node          = G.x.shape[0]
        self.init_std        = init_std

        Trans                = LaplacianLambdaMax(normalization=graph_normalize, is_undirected=G.is_undirected)
        laplacian            = lap2mat(*get_laplacian(G.edge_index, G.edge_weight, normalization=graph_normalize))
        lambda_max           = Trans(G).lambda_max
        self.laplacian       = 2 * laplacian / lambda_max - torch.eye(laplacian.shape[0])            
                                                                                # Scaled and normalized Laplacian matrix of G

        self.xbasiss_time    = torch.nn.ModuleList([])
        self.ybasiss_time    = torch.nn.ModuleList([])
        self.Bbasiss_loc     = torch.nn.ModuleList([])
        self.weights         = torch.nn.ParameterList([])
        # self.weights_mark   = torch.nn.ParameterList([])

        for i in range(n_basis_time):
            self.xbasiss_time.append(DeepNetworkBasis(1, basis_dim=1, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width_basis_time))
            self.ybasiss_time.append(DeepNetworkBasis(1, basis_dim=1, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width_basis_time))
        
        filter_list = []
        filter_list.append(torch.eye(self.laplacian.shape[0]))
        filter_list.append(self.laplacian)
        for i in range(n_basis_loc-2):
            filter_list.append(2 * torch.matmul(self.laplacian, filter_list[-1]) - filter_list[-2])
        for i in range(n_basis_loc):
            self.Bbasiss_loc.append(GraphLocalFilterBasis_Chebnet(filter_list[i]))

        self.weights.append(torch.nn.Parameter(torch.empty(n_basis_time, n_basis_loc).uniform_(-init_std,to=init_std), requires_grad=True))


    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, data_dim ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ], history
        """

        K_t = []
        K_l = []
        mask_t = (x[:, 0]-y[:, 0]).unsqueeze(-1) <= self.tau_max                 # [ batch_size ]

        for xbasis_t, ybasis_t in zip(self.xbasiss_time, self.ybasiss_time):
            xbasis_func_time = xbasis_t((x[:, 0]-y[:, 0]).unsqueeze(-1) / (self.T[1] - self.T[0])) * mask_t    # [ batch_size, basis_dim ]
            ybasis_func_time = ybasis_t((y[:, 0].unsqueeze(-1) - self.T[0]) / (self.T[1] - self.T[0]))                       # [ batch_size, basis_dim ]
            ki_t          = (xbasis_func_time * ybasis_func_time).sum(1)    # [ batch_size ]
            K_t.append(ki_t)
        K_t = torch.stack(K_t, 0)                                               # [ n_basis_time, batch_size ]

        mask = mask_t.bool()
        for Bbasis in self.Bbasiss_loc:
            Bbasis_func = torch.zeros_like(xbasis_func_time)
            Bbasis_func[mask] = Bbasis(y[:, [1]][mask].long(), x[:, [1]][mask].long())
            K_l.append(Bbasis_func.squeeze(-1))                                 # [ batch_size ]
        K_l = torch.stack(K_l, 0)                                               # [ n_basis_loc, batch_size ]

        # weight_soft      = torch.nn.functional.softplus(self.weights[0], beta=100)  # [ n_basis_time, n_basis_loc ]
        weight_soft      = self.weights[0]                                      # [ n_basis_time, n_basis_loc ]
        K = (torch.permute(torch.einsum('il,jl->ijl', K_t, K_l), (-1, -2, -3)) * weight_soft.T).sum((-1, -2))

        return K      # [ batch_size ]


class TemporalDeepBasisL3netLocalFilterOnGraphKernel(torch.nn.Module):
    """
    Temporal Basis Kernel with L3net local filter on graph.
    """
    def __init__(self, device, tau_max, T, G,
                 n_basis_time, loc_order_list, data_dim, basis_dim, 
                 init_gain=5e-1, init_bias=1e-1, init_std=1e1,
                 nn_width_basis_time=5):
        """
        Arg:
        - G: torch_geometric object
        """
        super(TemporalDeepBasisL3netLocalFilterOnGraphKernel, self).__init__()
        # configurations
        self.device          = device
        self.n_basis_time    = n_basis_time
        self.n_basis_loc     = len(loc_order_list)
        self.data_dim        = data_dim
        self.basis_dim       = basis_dim
        self.tau_max         = tau_max
        self.T               = T
        self.n_node          = G.x.shape[0]
        self.init_std        = init_std

        self.xbasiss_time    = torch.nn.ModuleList([])
        self.ybasiss_time    = torch.nn.ModuleList([])
        self.Bbasiss_loc     = torch.nn.ModuleList([])
        self.weights         = torch.nn.ParameterList([])
        # self.weights_mark   = torch.nn.ParameterList([])

        for i in range(n_basis_time):
            self.xbasiss_time.append(DeepNetworkBasis(1, basis_dim=1, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width_basis_time))
            self.ybasiss_time.append(DeepNetworkBasis(1, basis_dim=1, 
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width_basis_time))
        
        self.A_ = to_dense_adj(G.edge_index)[0]
        filter_list = []
        for order in loc_order_list:
            if order == 0:
                filter_list.append(self.init_local_filter(torch.eye(self.A_.shape[1]).float()))
            else:
                A_total = torch.zeros_like(self.A_)
                for i in range(1, order + 1):
                    A_total += self.A_.matrix_power(i)
                filter_list.append(self.init_local_filter((A_total != 0).float()))
        for i in range(self.n_basis_loc):
            self.Bbasiss_loc.append(GraphLocalFilterBasis_L3net(filter_list[i], learnable=True))

        self.weights.append(torch.nn.Parameter(torch.empty(self.n_basis_time, self.n_basis_loc).uniform_(-init_std,to=init_std), requires_grad=True))

    
    def init_local_filter(self, mask):
        """
        Initialize local filter with normal distribution on k-hop neighborhood.
        """
        in_size = self.n_basis_loc ** 2 * self.A_.sum(1).mean()
        std_ = np.sqrt(1. / in_size)
        return torch.randn((mask.shape[0], mask.shape[0])) * std_ * mask


    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, data_dim ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ], history
        """

        K_t = []
        K_l = []
        mask_t = (x[:, 0]-y[:, 0]).unsqueeze(-1) <= self.tau_max                 # [ batch_size ]

        for xbasis_t, ybasis_t in zip(self.xbasiss_time, self.ybasiss_time):
            xbasis_func_time = xbasis_t((x[:, 0]-y[:, 0]).unsqueeze(-1) / (self.T[1] - self.T[0])) * mask_t    # [ batch_size, basis_dim ]
            ybasis_func_time = ybasis_t((y[:, 0].unsqueeze(-1) - self.T[0]) / (self.T[1] - self.T[0]))                       # [ batch_size, basis_dim ]
            ki_t          = (xbasis_func_time * ybasis_func_time).sum(1)    # [ batch_size ]
            K_t.append(ki_t)
        K_t = torch.stack(K_t, 0)                                               # [ n_basis_time, batch_size ]

        mask = mask_t.bool()
        for Bbasis in self.Bbasiss_loc:
            Bbasis_func = torch.zeros_like(xbasis_func_time)
            Bbasis_func[mask] = Bbasis(y[:, [1]][mask].long(), x[:, [1]][mask].long())
            K_l.append(Bbasis_func.squeeze(-1))                                 # [ batch_size ]
        K_l = torch.stack(K_l, 0)                                               # [ n_basis_loc, batch_size ]

        # weight_soft      = torch.nn.functional.softplus(self.weights[0], beta=100)  # [ n_basis_time, n_basis_loc ]
        weight_soft      = self.weights[0]                                      # [ n_basis_time, n_basis_loc ]
        K = (torch.permute(torch.einsum('il,jl->ijl', K_t, K_l), (-1, -2, -3)) * weight_soft.T).sum((-1, -2))

        return K      # [ batch_size ]


class TemporalDeepBasisGATLocalFilterOnGraphKernel(torch.nn.Module):
    """
    Temporal Basis Kernel with GAT local filter on graph.
    """
    def __init__(self, device, tau_max, T, G,
                 n_basis_time, n_basis_loc, data_dim, basis_dim,
                 init_gain=5e-1, init_bias=1e-1, init_std=1e1,
                 nn_width_basis_time=5):
        """
        Arg:
        - G: torch_geometric object
        """
        super(TemporalDeepBasisGATLocalFilterOnGraphKernel, self).__init__()
        # configurations
        self.device          = device
        self.n_basis_time    = n_basis_time
        self.n_basis_loc     = n_basis_loc
        self.data_dim        = data_dim
        self.basis_dim       = basis_dim
        self.tau_max         = tau_max
        self.T               = T
        self.n_node          = G.x.shape[0]
        self.init_std        = init_std

        self.xbasiss_time    = torch.nn.ModuleList([])
        self.ybasiss_time    = torch.nn.ModuleList([])
        self.Bbasiss_loc     = torch.nn.ModuleList([])
        self.weights         = torch.nn.ParameterList([])
        # self.weights_mark   = torch.nn.ParameterList([])

        for i in range(n_basis_time):
            self.xbasiss_time.append(DeepNetworkBasis(1, basis_dim=1,
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width_basis_time))
            self.ybasiss_time.append(DeepNetworkBasis(1, basis_dim=1,
                                                 init_gain=init_gain, init_bias=init_bias,
                                                 nn_width=nn_width_basis_time))

        self.A_ = to_dense_adj(G.edge_index)[0]
        for i in range(self.n_basis_loc):
            self.Bbasiss_loc.append(GraphLocalFilterBasis_GAT(self.init_local_filter(torch.ones((self.n_node, self.n_node))), learnable=True))

        self.weights.append(torch.nn.Parameter(torch.empty(self.n_basis_time, self.n_basis_loc).uniform_(-init_std,to=init_std), requires_grad=True))

    
    def init_local_filter(self, mask):
        """
        Initialize local filter with normal distribution on k-hop neighborhood.
        """
        in_size = self.n_basis_loc ** 2 * self.A_.sum(1).mean()
        std_ = np.sqrt(1. / in_size)
        return torch.randn((mask.shape[0], mask.shape[0])) * std_ * mask


    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with
        size [ batch_size, data_dim ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ], history
        """

        K_t = []
        K_l = []
        mask_t = (x[:, 0]-y[:, 0]).unsqueeze(-1) <= self.tau_max                 # [ batch_size ]

        for xbasis_t, ybasis_t in zip(self.xbasiss_time, self.ybasiss_time):
            xbasis_func_time = xbasis_t((x[:, 0]-y[:, 0]).unsqueeze(-1) / (self.T[1] - self.T[0])) * mask_t    # [ batch_size, basis_dim ]
            ybasis_func_time = ybasis_t((y[:, 0].unsqueeze(-1) - self.T[0]) / (self.T[1] - self.T[0]))                       # [ batch_size, basis_dim ]
            ki_t          = (xbasis_func_time * ybasis_func_time).sum(1)    # [ batch_size ]
            K_t.append(ki_t)
        K_t = torch.stack(K_t, 0)                                               # [ n_basis_time, batch_size ]

        mask = mask_t.bool()
        for Bbasis in self.Bbasiss_loc:
            Bbasis_func = torch.zeros_like(xbasis_func_time)
            Bbasis_func[mask] = Bbasis(y[:, [1]][mask].long(), x[:, [1]][mask].long())
            K_l.append(Bbasis_func.squeeze(-1))                                 # [ batch_size ]
        K_l = torch.stack(K_l, 0)                                               # [ n_basis_loc, batch_size ]

        # weight_soft      = torch.nn.functional.softplus(self.weights[0], beta=100)  # [ n_basis_time, n_basis_loc ]
        weight_soft      = self.weights[0]                                      # [ n_basis_time, n_basis_loc ]
        K = (torch.permute(torch.einsum('il,jl->ijl', K_t, K_l), (-1, -2, -3)) * weight_soft.T).sum((-1, -2))

        return K      # [ batch_size ]
    

"""### Kernel with parametric temporal basis"""

class TemporalParametricKernelChebnetLocalFilterOnGraph(torch.nn.Module):
    """
    Temporal Basis Kernel with Chebnet local filter on graph.
    """
    def __init__(self, device, tau_max, T, G,
                 time_basis_list, n_basis_loc,
                 data_dim, basis_dim,
                 basis_weight=None, init_std=1.,
                 graph_normalize="sym"):
        """
        Arg:
        - G: torch_geometric object
        """
        super(TemporalParametricKernelChebnetLocalFilterOnGraph, self).__init__()
        # configurations
        self.device          = device
        self.n_basis_time    = len(time_basis_list)
        self.n_basis_loc     = n_basis_loc
        self.data_dim        = data_dim
        self.basis_dim       = basis_dim
        self.tau_max         = tau_max
        self.T               = T
        self.n_node          = G.x.shape[0]

        Trans                = LaplacianLambdaMax(normalization=graph_normalize, is_undirected=G.is_undirected)
        laplacian            = lap2mat(*get_laplacian(G.edge_index, G.edge_weight, normalization=graph_normalize))
        lambda_max           = Trans(G).lambda_max
        self.laplacian       = 2 * laplacian / lambda_max - torch.eye(laplacian.shape[0])            
                                                                                # Scaled and normalized Laplacian matrix of G

        self.time_basiss     = torch.nn.ModuleList([])
        self.Bbasiss_loc     = torch.nn.ModuleList([])
        self.weights         = torch.nn.ParameterList([])
        # self.weights_mark   = torch.nn.ParameterList([])

        for i in range(len(time_basis_list)):
            self.time_basiss.append(time_basis_list[i])
        
        filter_list = []
        filter_list.append(torch.eye(self.laplacian.shape[0]))
        filter_list.append(self.laplacian)
        for i in range(self.n_basis_loc-2):
            filter_list.append(2 * torch.matmul(self.laplacian, filter_list[-1]) - filter_list[-2])
        for i in range(self.n_basis_loc):
            self.Bbasiss_loc.append(GraphLocalFilterBasis_Chebnet(filter_list[i]))
        
        if basis_weight is not None:
            self.weights.append(torch.nn.Parameter(torch.Tensor(basis_weight), requires_grad=False))
        else: 
            self.weights.append(torch.nn.Parameter(torch.empty(self.n_basis_time, self.n_basis_loc).uniform_(-init_std,to=init_std), requires_grad=True))


    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, data_dim ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ], history
        """

        K_t = []
        K_l = []
        mask_t = (x[:, 0]-y[:, 0]) <= self.tau_max                              # [ batch_size ]

        for basis in self.time_basiss:
            K_t.append(basis(x[:, 0], y[:, 0]))
        K_t = torch.stack(K_t, 0)                                               # [ n_basis_time, batch_size ]

        mask = mask_t.bool()
        for Bbasis in self.Bbasiss_loc:
            Bbasis_func = torch.zeros_like(K_t[0])
            Bbasis_func[mask] = Bbasis(y[:, 1][mask].long(), x[:, 1][mask].long())
            K_l.append(Bbasis_func)                                             # [ batch_size ]
        K_l = torch.stack(K_l, 0)                                               # [ n_basis_loc, batch_size ]

        # weight_soft      = torch.nn.functional.softplus(self.weights[0], beta=100)  # [ n_basis_time, n_basis_loc ]
        weight_soft      = self.weights[0]                                      # [ n_basis_time, n_basis_loc ]
        K = (torch.permute(torch.einsum('il,jl->ijl', K_t, K_l), (-1, -2, -3)) * weight_soft.T).sum((-1, -2))

        return K      # [ batch_size ]


class TemporalParametricKernelL3netLocalFilterOnGraph(torch.nn.Module):
    """
    Temporal Basis Kernel with Chebnet local filter on graph.
    """
    def __init__(self, device, tau_max, T, G,
                 time_basis_list, loc_filter_list,
                 data_dim, basis_dim,
                 basis_weight=None, init_std=1.):
        """
        Arg:
        - G: torch_geometric object
        """
        super(TemporalParametricKernelL3netLocalFilterOnGraph, self).__init__()
        # configurations
        self.device          = device
        self.n_basis_time    = len(time_basis_list)
        self.n_basis_loc     = len(loc_filter_list)
        self.data_dim        = data_dim
        self.basis_dim       = basis_dim
        self.tau_max         = tau_max
        self.T               = T
        self.n_node          = G.x.shape[0]

        self.time_basiss     = torch.nn.ModuleList([])
        self.Bbasiss_loc     = torch.nn.ModuleList([])
        self.weights         = torch.nn.ParameterList([])

        for time_basis in time_basis_list:
            self.time_basiss.append(time_basis)
        
        for filter in loc_filter_list:
            self.Bbasiss_loc.append(GraphLocalFilterBasis_L3net(filter))
        
        if basis_weight is not None:
            self.weights.append(torch.nn.Parameter(torch.Tensor(basis_weight), requires_grad=False))
        else: 
            self.weights.append(torch.nn.Parameter(torch.empty(self.n_basis_time, self.n_basis_loc).uniform_(-init_std,to=init_std), requires_grad=True))


    def forward(self, x, y):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, data_dim ], where
        - x: the first input with size  [ batch_size, data_dim ]
        - y: the second input with size [ batch_size, data_dim ], history
        """

        K_t = []
        K_l = []
        mask_t = (x[:, 0]-y[:, 0]) <= self.tau_max                              # [ batch_size ]

        for basis in self.time_basiss:
            K_t.append(basis(x[:, 0], y[:, 0]))
        K_t = torch.stack(K_t, 0)                                               # [ n_basis_time, batch_size ]

        mask = mask_t.bool()
        for Bbasis in self.Bbasiss_loc:
            Bbasis_func = torch.zeros_like(K_t[0])
            Bbasis_func[mask] = Bbasis(y[:, 1][mask].long(), x[:, 1][mask].long())
            K_l.append(Bbasis_func)                                             # [ batch_size ]
        K_l = torch.stack(K_l, 0)                                               # [ n_basis_loc, batch_size ]

        # weight_soft      = torch.nn.functional.softplus(self.weights[0], beta=100)  # [ n_basis_time, n_basis_loc ]
        weight_soft      = self.weights[0]                                      # [ n_basis_time, n_basis_loc ]
        K = (torch.permute(torch.einsum('il,jl->ijl', K_t, K_l), (-1, -2, -3)) * weight_soft.T).sum((-1, -2))

        return K      # [ batch_size ]

class ExpBasis(torch.nn.Module):
    """
    Kernel function including the diffusion-type model proposed by Musmeci and
    Vere-Jones (1992).
    """
    def __init__(self, beta=1.):
        super(ExpBasis, self).__init__()
        self.beta    = beta

    def forward(self, t, his_t):
        delta_t = t - his_t
        return self.beta * torch.exp(- self.beta * delta_t)


class BaseExponentialCosineBasis(torch.nn.Module):
    """
    Exponential Cosine Kernel without touching 0 in cosine.
    """
    def __init__(self, alpha, beta, freq):
        """
        Arg:
        - alpha: kernel magnitude
        - beta: decaying rate
        """
        super(BaseExponentialCosineBasis, self).__init__()
        self._alpha = alpha
        self._beta  = beta
        self._freq  = freq
    
    def forward(self, t, his_t):
        return self._alpha * torch.exp(- self._beta * torch.abs(t - his_t)) * \
                (torch.cos(self._freq * torch.clamp(his_t, min=0., max=None)) * 0.5 + 0.5)


class InfiniteRankBasis(torch.nn.Module):
    """
    Stationary Decaying Cosine Kernel
    """
    def __init__(self, eta, r_freq, r_decay):
        """
        Arg:
        - beta: decaying rate
        """
        super(InfiniteRankBasis, self).__init__()
        self._eta   = eta
        self._r_freq   = r_freq
        self._r_decay  = r_decay
    
    def forward(self, t, his_t):
        """
        customized forward function returning kernel evaluation at x and y with 
        size [ batch_size, batch_size ], where
        - x: the first input with size  [ batch_size, data_dim = 1 ]
        - y: the second input with size [ batch_size, data_dim = 1 ] 
        """

        rn = 15
        tp = his_t                   # [ batch_size ]
        tau = t - his_t              # [ batch_size ]
        ttp = tp.unsqueeze(-1).repeat(1, rn)             # [ batch_size, rn ]
        ttau = tau.unsqueeze(-1).repeat(1, rn)           # [ batch_size, rn ]
        rank = torch.arange(1, 1+rn, 1, device=t.device)  # [ rn ]
        sigj = 1 / rank * self._r_decay    # [ rn ]
        coefj = 1 / torch.pow(2, rank)        # [ rn ]

        F = self._eta * (0.3 + torch.cos(2 + ttp ** 0.7 * self._r_freq * (rank+1) * np.pi)) * \
            torch.exp(- ttau ** 2 / 2 / sigj ** 2) * coefj      # [ batch_size, rn ]

        return F.sum(1)

    
"""## Point process"""

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor_linspace(start, stop, num):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    
    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    
    return out

class BasePointProcess(torch.nn.Module):
    """
    Point Process Base Class
    """
    @abstractmethod
    def __init__(self, T, S, data_dim, device, numerical_int=True, int_res=100, eval_res=50, eval_points=None):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag, use simpson integration in 1D case.
        - int_res:       numerical integral resolution
        """
        super(BasePointProcess, self).__init__()
        # configuration
        self.data_dim      = data_dim
        self.T             = T # time horizon. e.g. (0, 1)
        self.S             = S # bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        self.numerical_int = numerical_int
        self.int_res       = int_res
        self.eval_res      = eval_res 
        self.device        = device
        # assert len(S) + 1 == self.data_dim, "Invalid space dimension"

        # numerical likelihood integral preparation
        self.tt       = np.linspace(self.T[0], self.T[1], eval_res)  # [ eval_res ]
        self.ss       = [ np.linspace(S_k[0], S_k[1], eval_res) for S_k in self.S ]     # [ data_dim - 1, eval_res ]
        # spatio-temporal coordinates that need to be evaluated
        self.t_coords = torch.ones((eval_res ** (len(S)), 1), device=self.device)                     # [ eval_res^(data_dim - 1), 1 ]
        self.s_coords = torch.FloatTensor(np.array(list(itertools.product(*self.ss)))).to(self.device) # [ eval_res^(data_dim - 1), data_dim - 1 ]
        # unit volumn
        self.unit_vol = np.prod([ S_k[1] - S_k[0] for S_k in self.S ] + [ self.T[1] - self.T[0] ]) / (self.eval_res) ** (len(S)+1)

        if eval_points is not None:
            self.t_eval = torch.ones((eval_points.shape[0], 1), device=self.device)
            self.s_eval = eval_points.to(self.device)

    def eval(self, X):
        """
        return conditional intensity evaluation at grid points, the numerical 
        integral can be further calculated by summing up these evaluations and 
        scaling by the unit volumn.

        -- X: all the data points.
        """
        batch_size, seq_len, _ = X.shape
        n_eval_points = self.s_eval.shape[0]
        integral = []
        for t in self.tt:
            # all possible points at time t (x_t) 
            t_coord = self.t_eval * t
            xt      = torch.cat([t_coord, self.s_eval], 1) # [ n_eval_points, data_dim ] 
            xt      = xt\
                .unsqueeze_(0)\
                .repeat(batch_size, 1, 1)\
                .reshape(-1, self.data_dim)                  # [ batch_size * n_eval_points, data_dim ]
            # history points before time t (H_t)
            mask = ((X[:, :, 0].clone() < t) * (X[:, :, 0].clone() > 0))\
                .unsqueeze_(-1)\
                .repeat(1, 1, self.data_dim)                 # [ batch_size, seq_len, data_dim ]
            ht   = X * mask                                  # [ batch_size, seq_len, data_dim ]
            ht   = ht\
                .unsqueeze_(1)\
                .repeat(1, n_eval_points, 1, 1)\
                .reshape(-1, seq_len, self.data_dim)         # [ batch_size * n_eval_points, seq_len, data_dim ]
            # lambda and integral 
            # lams = torch.nn.functional.softplus(self.cond_lambda(xt, ht), beta=10)\
            #     .reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            lams = self.cond_lambda(xt, ht).reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            integral.append(lams)                            
        # NOTE: second dimension is time, third dimension is mark space
        integral = torch.stack(integral, 1)                  # [ batch_size, int_res, int_res^(data_dim - 1) ]
        return integral
    
    def numerical_sample(self, X):
        """
        return conditional intensity evaluation at grid points, the numerical 
        integral can be further calculated by summing up these evaluations and 
        scaling by the unit volumn.

        -- X: all the data points.
        """
        batch_size, seq_len, _ = X.shape
        integral = []
        for t in self.tt:
            # all possible points at time t (x_t) 
            t_coord = self.t_coords * t
            xt      = torch.cat([t_coord, self.s_coords], 1) # [ int_res^(data_dim - 1), data_dim ] 
            xt      = xt\
                .unsqueeze_(0)\
                .repeat(batch_size, 1, 1)\
                .reshape(-1, self.data_dim)                  # [ batch_size * int_res^(data_dim - 1), data_dim ]
            # history points before time t (H_t)
            mask = ((X[:, :, 0].clone() < t) * (X[:, :, 0].clone() > 0))\
                .unsqueeze_(-1)\
                .repeat(1, 1, self.data_dim)                 # [ batch_size, seq_len, data_dim ]
            ht   = X * mask                                  # [ batch_size, seq_len, data_dim ]
            ht   = ht\
                .unsqueeze_(1)\
                .repeat(1, self.eval_res ** (self.data_dim - 1), 1, 1)\
                .reshape(-1, seq_len, self.data_dim)         # [ batch_size * int_res^(data_dim - 1), seq_len, data_dim ]
            # lambda and integral 
            # lams = torch.nn.functional.softplus(self.cond_lambda(xt, ht), beta=10)\
            #     .reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            lams = self.cond_lambda(xt, ht).reshape(batch_size, -1)                     # [ batch_size, int_res^(data_dim - 1) ]
            integral.append(lams)                            
        # NOTE: second dimension is time, third dimension is mark space
        integral = torch.stack(integral, 1)                  # [ batch_size, int_res, int_res^(data_dim - 1) ]
        return integral

    def numerical_integral(self, X):
        """
        return conditional intensity integral over time, using Simpson Quadrature
        numerical integration in torchquad.
        
        Now only work in 1D case (data_dim == 1). Higher dimension case remain to implement.

        -- X: all the data points.
        """

        # if self.data_dim != 1:
        #     raise NotImplementedError("Can only do quadrature integration on 1D data!")

        # integral_0 = self.numerical_sample(X) # [ batch_size, int_res, int_res^(data_dim - 1) ]
        # integral   = integral_0.squeeze()

        # h  = (self.T[1] - self.T[0]) / (self.eval_res - 1)

        # even_idx = np.arange(2, self.eval_res-1, 2)
        # odd_idx  = np.arange(1, self.eval_res-1, 2)

        # return (h / 3 * (integral[:, 0] + integral[:, -1] + 2 * integral[:, even_idx].sum(1) \
        #                 + 4 * integral[:, odd_idx].sum(1))).sum() # scalar

        return self.numerical_sample(X).sum() * self.unit_vol
    
    def cond_lambda(self, xi, hti):
        """
        return conditional intensity given x
        Args:
        - xi:   current i-th point       [ batch_size, data_dim ]
        - hti:  history points before ti [ batch_size, seq_len, data_dim ]
        Return:
        - lami: i-th lambda              [ batch_size ]
        """
        # if length of the history is zero
        if hti.size()[0] == 0:
            return self.mu(xi[:, 1]) * torch.ones(xi.shape[0], device = xi.device)
        # otherwise treat zero in the time (the first) dimension as invalid points
        batch_size, seq_len, _ = hti.shape
        mask = hti[:, :, 0].clone() > 0                                          # [ batch_size, seq_len ]
        xii  = xi.unsqueeze(1).repeat(1, seq_len, 1).reshape(-1, self.data_dim) # [ batch_size * seq_len, data_dim ]
        hti  = hti.reshape(-1, self.data_dim)                                    # [ batch_size * seq_len, data_dim ]
        K    = self.kernel(xii, hti).reshape(batch_size, seq_len)                # [ batch_size, seq_len ]
        K    = K * mask                                                          # [ batch_size, seq_len ]
        lami = K.sum(1) + self.mu(xi[:, 1])                                      # [ batch_size ]
        return lami

    def log_likelihood(self, X, n_sampled_fouriers=200):
        """
        return log-likelihood given sequence X
        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   sequence of lambda    [ batch_size, seq_len ]
        - loglik: log-likelihood        [ batch_size ]
        """
        batch_size, seq_len, _ = X.shape
        # lams     = [
        #     torch.nn.functional.softplus(self.cond_lambda(
        #         X[:, i, :].clone(), 
        #         X[:, :i, :].clone()), beta=10) + 1e-5
        #     for i in range(seq_len) ]
        lams     = [
            self.cond_lambda(X[:, i, :].clone(), X[:, :i, :].clone())
            for i in range(seq_len) ]
        lams     = torch.stack(lams, dim=1)                                   # [ batch_size, seq_len ]
        # log-likelihood
        mask     = X[:, :, 0] > 0                                             # [ batch_size, seq_len ]
        # print((lams*mask).min())
        sumlog   = torch.log(torch.clamp(lams, min=1e-5)) * mask                       # [ batch_size, seq_len ]
        # sumlog   = torch.log(lams) * mask                       # [ batch_size, seq_len ]
        if self.numerical_int:
            integral = self.numerical_integral(X)                             # scalar
            loglik = sumlog.sum() - integral                                  # scalar
        else: 
            # TODO: integral in analytical form
            pass
        return loglik

    @abstractmethod
    def mu(self):
        """
        return base intensity
        """
        raise NotImplementedError()

    # def numerical_integral(self, X):
    #     """
    #     return efficient computation of conditional intensity function integral
    #     in log-likelihood of multiple sequences.

    #     -- X: all the data points.
    #     """
    #     raise NotImplementedError()

    @abstractmethod
    def forward(self, X):
        """
        custom forward function returning conditional intensities and corresponding log-likelihood
        """
        # return conditional intensities and corresponding log-likelihood
        return self.log_likelihood(X)

    
class TemporalPointProcessOnGraph(BasePointProcess):
    """
    Point Process on graph with deep temporal basis and graph local filter.
    """
    def __init__(self, device,
                 T, G, mu, tau_max, kernel, loss="grid_l2_loss",
                 data_dim=2, numerical_int=True,
                 eval_res=100, int_res=100, int_res_loc=200,
                 pen_res_time=50, l2_res_time=100):
        """
        Args:
        - T:             time horizon. e.g. (0, 1)
        - S:             bounded space for marks. e.g. a two dimensional box region [(0, 1), (0, 1)]
        - n_basis:       number of basis functions
        - n_mark_cate:   number of unique mark categories
        - basis_dim:     dimension of basis function
        - data_dim:      dimension of input data
        - numerical_int: numerical integral flag
        - int_res:       numerical integral resolution
        - nn_width:      the width of each layer in kernel basis NN
        """
        super(TemporalPointProcessOnGraph, self).__init__(T, (), data_dim, device, numerical_int, int_res, eval_res)
        # configuration
        self.loss_type      = loss
        self.n_node         = G.x.shape[0]
        self._mu            = torch.nn.Parameter(torch.tensor(mu), requires_grad=False)
        self.int_res_loc    = int_res_loc
        self.pen_res_time   = pen_res_time
        self.l2_res_time    = l2_res_time
        # self.tgrids         = torch.linspace(self.T[0], self.T[1], self.pen_res_time, device=self.device)    # [ pen_res ]
        # mm                  = [ torch.linspace(s[0], s[1], self.pen_res) for s in self.S ]
        # self.mgrids         = torch.tensor(list(itertools.product(*mm)), device=self.device)              # [ int_res^(data_dim - 1), data_dim - 1 ]
        self.tgrids         = torch.linspace(self.T[0], self.T[1], self.pen_res_time, device=self.device)    # [ pen_res ]
        self.lgrids         = torch.arange(G.x.shape[0], device=self.device).unsqueeze(-1)              # [ n_node, 1 ]
        # deep nn basis kernel on graph
        self.kernel         = kernel
        self.n_basis_time   = kernel.n_basis_time
        self.n_basis_loc    = kernel.n_basis_loc
        self.data_dim       = kernel.data_dim
        self.basis_dim      = kernel.basis_dim

        # self.eval_res decide the spatio-temporal grids that feature functions evaluated on, which are used for all the approximation later on.
        self.tt             = torch.linspace(0, self.kernel.tau_max, self.eval_res).to(self.device).reshape(-1, 1)
        self.tt_l2          = torch.linspace(self.T[0], self.T[1], self.l2_res_time+1).to(self.device)         # [ l2_res_time+1 ]


    def mu(self, X):
        """
        return base intensity
        """
        return self._mu[X.long()]


    def nn_evaluation(self, X):
        """
        implement nn evaluation required for SGD, including ybasis on events and
        xbasis on uniform grid with "self.eval_res".

        -- X: data points: [ bath_size, seq_len, data_dim ]
        """

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()                                               # [ batch_size, seq_len ]
        ls   = X[:, :, 1].clone()                                               # [ batch_size, seq_len ]
        mask_all = ts > 0

        # mask out entries with zero kernel values
        taus  = ts[:, None, :] - ts[:, :, None]                                 # [ batch_size, seq_len, seq_len ]
        self.mask = ((taus > 0) * (taus <= self.kernel.tau_max) * \
                     mask_all[:, None, :]).bool()                               # [ batch_size, seq_len, seq_len ]

        # penalized grids on location
        if self.lgrids.device != X.device:
            lgrids = self.lgrids.to(X.device)
        else:
            lgrids = self.lgrids.clone()                                        # [ pen_res_loc, 1 ]

        xbasis_time_grid  = []
        ybasis_time_event = []
        Bbasis_loc_filter = []
        weights           = []

        for xbasis_t, ybasis_t in zip(self.kernel.xbasiss_time, self.kernel.ybasiss_time):
            ybasis_time_event.append(ybasis_t((ts.unsqueeze(-1) - self.T[0]) / (self.T[1] - self.T[0])).squeeze(-1) * mask_all)  # [ batch_size, seq_len ]
            xbasis_time_grid.append(xbasis_t(self.tt / (self.T[1] - self.T[0])).squeeze(-1))  # [ eval_res ]

        self.xbasis_time_grids  = torch.stack(xbasis_time_grid, 0)              # [ n_basis_time, eval_res ]
        self.ybasis_time_events = torch.stack(ybasis_time_event, 0)             # [ n_basis_time, batch_size, seq_len ]

        for Bbasis in self.kernel.Bbasiss_loc:
            Bbasis_loc_filter.append(Bbasis.get_filter())
        self.Bbasis_loc_filters   = torch.stack(Bbasis_loc_filter, 0)           # [ n_basis_loc, n_node, n_node ]

        self.weights_soft       = self.kernel.weights[0].clone()+0              # [ n_basis_time, n_basis_loc ]
        # if len(self.weights.shape) == 1:
        #     self.weights = self.weights.unsqueeze(0)
        self.h = self.kernel.tau_max / (self.eval_res - 1)


    def event_lams(self, X):
        """
        return intensity at each event in sequence X

        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - lams:   intensities           [ batch_size, seq_len ]
        """

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()
        ls   = X[:, :, 1].clone()
        mask_all = ts > 0

        taus  = ts[:, None, :] - ts[:, :, None]                                 # [ batch_size, seq_len, seq_len ]
        ybasis_time_val = self.ybasis_time_events.clone()                       # [ n_basis_time, batch_size, seq_len ]
        xbasis_time_val = torch.zeros_like(ybasis_time_val).unsqueeze(-2).repeat(1, 1, seq_len, 1)
                                                                                # [ n_basis_time, batch_size, seq_len, seq_len ]

        ttaus = torch.clamp(taus[self.mask], max=self.kernel.tau_max-0.01)      # [ mask.sum() ]
        int_start_idx = (torch.div(ttaus, self.h, rounding_mode="floor")).long()  # [ mask.sum() ]
        int_end_idx   = int_start_idx + 1
        int_start     = self.xbasis_time_grids[:, int_start_idx]                # [ n_basis_time, mask.sum() ]
        int_end       = self.xbasis_time_grids[:, int_end_idx]                  # [ n_basis_time, mask.sum() ]
        int_prop      = (torch.remainder(ttaus, self.h) / self.h).unsqueeze(0).repeat(self.n_basis_time, 1)  # [ n_basis_time, mask.sum() ]

        xbasis_time_val[:, self.mask] = torch.lerp(int_start, int_end, int_prop) # [ n_basis_time, mask.sum() ]
        kernel_time = xbasis_time_val * ybasis_time_val[:, :, :, None]          # [ n_basis_time, batch_size, seq_len, seq_len ]

        ls_pairs = torch.stack((ls[:, None, :].repeat(1, seq_len, 1),
                               ls[:, :, None].repeat(1, 1, seq_len)), axis=-1)  # [ batch_size, seq_len, seq_len, 2 ]
        kernel_loc  = []
        for filter in self.Bbasis_loc_filters:
            kernel_loc.append(filter[ls_pairs[..., 1].long(), ls_pairs[..., 0].long()])
        kernel_loc  = torch.stack(kernel_loc, axis=0)                           # [ n_basis_loc, batch_size, seq_len, seq_len ]


        kernel_val = torch.einsum('ibmn,jbmn->ijbmn', kernel_time, kernel_loc)

        lams = (kernel_val.T * self.weights_soft.T).sum((-1, -2)).T.sum(1) + self.mu(X[:, :, 1]) # [ batch_size, seq_len ]

        # lams_sum = (lams * mask_all).sum()

        return lams


    def penalty_grid_lams(self, X):
        """
        return the penalty of lambda function, guaranting lambda function to be nonnegative
        using log barrier

        -- X: all the data points.
        """

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()
        ls   = X[:, :, 1].clone()
        mask_all = ts > 0

        if self.tgrids.device != X.device:
            tgrids = self.tgrids.to(X.device)
            lgrids = self.lgrids.to(X.device)
        else:
            tgrids = self.tgrids.clone()
            lgrids = self.lgrids.clone()

        lams = []

        # compute time kernel
        tts  = ts.unsqueeze(0).repeat(self.pen_res_time, 1, 1)                  # [ pen_res_time, batch_size, seq_len ]
        taus = (tgrids - tts.T).T                                               # [ pen_res_time, batch_size, seq_len ]
        mask = ((taus > 0) * (taus <= self.kernel.tau_max)).bool()              # [ pen_res_time, batch_size, seq_len ]

        ybasis_time_val = self.ybasis_time_events.clone()                       # [ n_basis_time, batch_size, seq_len ]
        xbasis_time_val = torch.zeros_like(ybasis_time_val).unsqueeze(1).repeat(1, self.pen_res_time, 1, 1)  # [ n_basis_time, pen_res_time, batch_size, seq_len ]

        ttaus = torch.clamp(taus[mask], max=self.kernel.tau_max-0.01)           # [ mask.sum() ]
        int_start_idx = (torch.div(ttaus, self.h, rounding_mode="floor")).long()  # [ mask.sum() ]
        int_end_idx   = int_start_idx + 1
        int_start     = self.xbasis_time_grids[:, int_start_idx]                # [ n_basis_time, mask.sum() ]
        int_end       = self.xbasis_time_grids[:, int_end_idx]                  # [ n_basis_time, mask.sum() ]
        int_prop      = (torch.remainder(ttaus, self.h) / self.h).unsqueeze(0).repeat(self.n_basis_time, 1)  # [ n_basis_time, mask.sum() ]

        xbasis_time_val[:, mask] = torch.lerp(int_start, int_end, int_prop)     # [ n_basis_time, pen_res_time, batch_size, seq_len ]
        pen_time = xbasis_time_val * ybasis_time_val[:, None, :, :]             # [ n_basis_time, pen_res_time, batch_size, seq_len ]

        # compute location kernel
        ls_pairs = torch.stack((lgrids[:, None].repeat(1, batch_size, seq_len),
                               ls[None, :, :].repeat(self.n_node, 1, 1)), axis=-1)
                                                                                # [ n_node, batch_size, seq_len, 2 ]
        pen_loc = []
        for filter in self.Bbasis_loc_filters:
            pen_loc.append(filter[ls_pairs[..., 1].long(), ls_pairs[..., 0].long()])
        pen_loc  = torch.stack(pen_loc, axis=0)                                 # [ n_basis_loc, n_node, batch_size, seq_len ]

        F_pen = torch.einsum('ihmn,jlmn->ijhlmn', pen_time, pen_loc)            # [ n_basis_time, n_basis_loc,
                                                                                #   pen_res_time, pen_res_loc, batch_size, seq_len ]

        F_pen = (F_pen.T * self.weights_soft.T).sum((-1, -2)).T.sum(-1) + self.mu(lgrids[None, :, :])
                                                                                # [ pen_res_time, pen_res_loc, batch_size ]


        return F_pen
    

    def time_basis_regularization(self, device):

        grids = torch.linspace(0.0,
                            self.kernel.tau_max-1e-4,
                            self.int_res, device=device)                      # [ int_res ]
        h   = self.kernel.tau_max / (self.int_res - 1)
        hs  = torch.ones(self.int_res, device=device) * h / 3                 # [ int_res ]
        hs[1] = h / 2

        int_start_idx = (torch.div(grids, self.h, rounding_mode="floor")).long()  # [ int_res ]
        int_end_idx   = int_start_idx + 1
        int_start     = self.xbasis_time_grids[:, int_start_idx]               # [ n_basis_time, int_res ]
        int_end       = self.xbasis_time_grids[:, int_end_idx]                 # [ n_basis_time, int_res ]
        int_prop      = (torch.remainder(grids, self.h) / self.h).unsqueeze(0).repeat(self.n_basis_time, 1)  # [ n_basis_time, int_res ]
        xbasis_time_val = torch.lerp(int_start, int_end, int_prop)              # [ n_basis_time, int_res ]

        w = ((grids / self.kernel.tau_max) * ((grids / self.kernel.tau_max) > 0.5)) ** 10
        reg = (xbasis_time_val ** 2 * w).sum() * h

        return reg


    def l2_grid_integ_loss(self, X):
        """
        return the integral of lambda function and the event count over each time interval determined by l2 grid

        -- X: all the data points.
        """

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()
        ls   = X[:, :, 1].clone()
        mask_all = ts > 0

        h_grid = (self.T[1] - self.T[0]) / self.l2_res_time
        # if self.tt_l2.device != X.device:
        #     ts_grid_all = self.tt_l2.to(X.device)
        # else:
        ts_grid_all = self.tt_l2.clone()
        ts_grid = ts_grid_all[:-1]


        ybasis_time_val = self.ybasis_time_events.clone()                       # [ n_basis_time, batch_size, seq_len ]

        grids = torch.linspace(0.0,
                            self.kernel.tau_max-1e-4,
                            self.int_res, device=X.device)                      # [ int_res ]
        h   = self.kernel.tau_max / (self.int_res - 1)
        hs  = torch.ones(self.int_res, device=X.device) * h / 3                 # [ int_res ]
        hs[1] = h / 2

        int_start_idx = (torch.div(grids, self.h, rounding_mode="floor")).long()  # [ int_res ]
        int_end_idx   = int_start_idx + 1
        int_start     = self.xbasis_time_grids[:, int_start_idx]               # [ n_basis_time, int_res ]
        int_end       = self.xbasis_time_grids[:, int_end_idx]                 # [ n_basis_time, int_res ]
        int_prop      = (torch.remainder(grids, self.h) / self.h).unsqueeze(0).repeat(self.n_basis_time, 1)  # [ n_basis_time, int_res ]

        xbasis_time_val = torch.lerp(int_start, int_end, int_prop)              # [ n_basis_time, int_res ]

        ## compute integration from start to each grid
        odd_idx  = np.arange(1, self.int_res-1, 2)
        simp_g        = torch.zeros_like(xbasis_time_val)                       # [ n_basis_time, int_res ]
        xbasis_time_val2   = xbasis_time_val.clone()
        xbasis_time_val2[:, odd_idx] = xbasis_time_val2[:, odd_idx] * 3
        simp_g[:, 1:] = xbasis_time_val[:, 1:] + xbasis_time_val2[:, :-1]       # [ n_basis_time, int_res ]
        simp_v        = (torch.cumsum(simp_g, dim=1) * hs)                      # [ n_basis_time, int_res ]

        ## compute integral over grid interval
        int_t_left = torch.clamp((ts_grid[:, None, None] - ts[None]), min=0, max=self.kernel.tau_max-1e-2)
                                                                                # [ l2_res_time, batch_size, seq_len ]
        int_t_right = torch.clamp((ts_grid[:, None, None] + h_grid - ts[None]), min=0, max=self.kernel.tau_max-1e-2)
                                                                                # [ l2_res_time, batch_size, seq_len ]
        int_start_idx_left = (torch.div(int_t_left, h, rounding_mode="floor") * mask_all).long()
        int_end_idx_left   = int_start_idx_left + 1
        int_start          = simp_v[:, int_start_idx_left]                      # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        int_end            = simp_v[:, int_end_idx_left]                        # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        int_prop           = (torch.remainder(int_t_left, h) / h).unsqueeze(0).repeat(self.n_basis_time, 1, 1, 1)
                                                                                # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        int_left  = torch.lerp(int_start, int_end, int_prop)                    # [ n_basis_time, l2_res_time, batch_size, seq_len ]

        int_start_idx_right = (torch.div(int_t_right, h, rounding_mode="floor") * mask_all).long()
        int_end_idx_right   = int_start_idx_right + 1
        int_start           = simp_v[:, int_start_idx_right]                    # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        int_end             = simp_v[:, int_end_idx_right]                      # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        int_prop            = (torch.remainder(int_t_right, h) / h).unsqueeze(0).repeat(self.n_basis_time, 1, 1, 1)
                                                                                # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        int_right  = torch.lerp(int_start, int_end, int_prop)                   # [ n_basis_time, l2_res_time, batch_size, seq_len ]

        int_time_bin_rightbasis = int_right - int_left                          # [ n_basis_time, l2_res_time, batch_size, seq_len ]
        integral_time_bin = int_time_bin_rightbasis * ybasis_time_val[:, None]  # [ n_basis_time, l2_res_time, batch_size, seq_len ]

        ker_val_loc  = []
        for filter in self.Bbasis_loc_filters:
            ker_val_loc.append(filter[ls.long(), :])
        ker_val_loc  = torch.stack(ker_val_loc, axis=0)                         # [ n_basis_loc, batch_size, seq_len, n_node ]

        integral_per_bin_over_time = (torch.einsum('inbl,jblm->ijnmbl', integral_time_bin, ker_val_loc).T * self.weights_soft.T).T.sum((0, 1, -1))
                                                                                # [ l2_res_time, n_node, batch_size ]
        # integral_per_bin_over_time = ((integral_time_bin[:, None, :, None] * ker_val_loc.permute(0, 3, 1, 2)[None, :, None, :]).T * self.weights_soft.T).T.sum((0, 1, -1))
        integral_per_bin_over_time = integral_per_bin_over_time + self._mu[None, :, None] * h_grid

        ## event count
        points = X[mask_all].clone()
        bin_idx = torch.bucketize(points[:, 0], ts_grid_all)
        digitized_points = torch.vstack([bin_idx, points[:, 1]]).T
        bin_v, bin_c = torch.unique(digitized_points, dim=0, return_counts=True)
        event_num = torch.zeros([self.l2_res_time, self.n_node], device=X.device)
        event_num[bin_v[:, 0].long()-1, bin_v[:, 1].long()] = bin_c.float()

        w = ((grids / self.kernel.tau_max) * ((grids / self.kernel.tau_max) > 0.5)) ** 10
        reg = (xbasis_time_val ** 2 * w).sum() * h

        return integral_per_bin_over_time, event_num, ((integral_per_bin_over_time.mean(-1) - event_num / batch_size)**2).sum(), reg

            
    def log_likelihood(self, X):
        """
        return log-likelihood given sequence X

        Args:
        - X:      input points sequence [ batch_size, seq_len, data_dim ]
        Return:
        - loglik: log-likelihood        [ batch_size ]
        """

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()
        ls   = X[:, :, 1].clone()
        mask_all = ts > 0

        lams = self.event_lams(X)
        integral, reg = self.numerical_integral(X)

        ## sometime needs normalization
        loglik = (torch.log(torch.clamp(lams, min=1e-5)) * mask_all).sum() - integral

        return loglik, reg


    def numerical_integral(self, X):
        """
        return efficient computation of conditional intensity function integral
        in log-likelihood of multiple sequences.

        -- X: all data points.
        """

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()
        ls   = X[:, :, 1].clone()
        mask_all = ts > 0
        n_event = mask_all.sum()
        baserate = self._mu.sum() * (self.T[1] - self.T[0]) * batch_size

        ## integration of time kernel
        # the resolution of temporal integration is int_res

        grids = torch.linspace(0.0,
                               self.kernel.tau_max-0.01,
                               self.int_res, device=X.device)                   # [ int_res ]
        h   = self.kernel.tau_max / (self.int_res - 1)
        hs  = torch.ones(self.int_res, device=X.device) * h / 3                 # [ int_res ]
        hs[1] = h / 2

        ybasis_time_val = self.ybasis_time_events.clone()                       # [ n_basis_time, batch_size, seq_len ]

        int_start_idx = (torch.div(grids, self.h, rounding_mode="floor")).long()  # [ int_res ]
        int_end_idx   = int_start_idx + 1
        int_start     = self.xbasis_time_grids[:, int_start_idx]                # [ n_basis_time, int_res ]
        int_end       = self.xbasis_time_grids[:, int_end_idx]                  # [ n_basis_time, int_res ]
        int_prop      = (torch.remainder(grids, self.h) / self.h).unsqueeze(0).repeat(self.n_basis_time, 1)  # [ n_basis_time, int_res ]

        xbasis_time_val = torch.lerp(int_start, int_end, int_prop)              # [ n_basis_time, int_res ]

        # compute integration from start to each grid
        odd_idx  = np.arange(1, self.int_res-1, 2)
        simp_g        = torch.zeros_like(xbasis_time_val)                       # [ n_basis_time, int_res ]
        xbasis_time_val2   = xbasis_time_val.clone()
        xbasis_time_val2[:, odd_idx] = xbasis_time_val2[:, odd_idx] * 3
        simp_g[:, 1:] = xbasis_time_val[:, 1:] + xbasis_time_val2[:, :-1]       # [ n_basis_time, int_res ]
        simp_v        = (torch.cumsum(simp_g, dim=1) * hs)                      # [ n_basis_time, int_res ]

        int_t = torch.clamp((self.T[1] - ts), max=self.kernel.tau_max-1e-1)     # [ batch_size, seq_len ]
        int_start_idx = (torch.div(int_t, h, rounding_mode="floor") * mask_all).long()
        int_end_idx   = int_start_idx + 1
        int_start     = simp_v[:, int_start_idx]                                # [ n_basis_time, batch_size, seq_len ]
        int_end       = simp_v[:, int_end_idx]                                  # [ n_basis_time, batch_size, seq_len ]
        int_prop      = (torch.remainder(int_t, h) / h).unsqueeze(0).repeat(self.n_basis_time, 1, 1)
                                                                                # [ n_basis_time, batch_size, seq_len ]

        int_xbasis_time  = torch.lerp(int_start, int_end, int_prop)             # [ n_basis_time, batch_size, seq_len ]
        integral_time = int_xbasis_time * ybasis_time_val * mask_all            # [ n_basis_time, batch_size, seq_len ]

        ## integration of location kernel
        integral_loc  = []
        for filter in self.Bbasis_loc_filters:
            integral_loc.append(filter[ls.long(), :].sum(-1))
        integral_loc  = torch.stack(integral_loc, axis=0)                       # [ n_basis_loc, batch_size, seq_len ]


        integral = torch.einsum('ibl,jbl->ijbl', integral_time, integral_loc).T * self.weights_soft.T

        # reg = None
        w = ((grids / self.kernel.tau_max) * ((grids / self.kernel.tau_max) > 0.5)) ** 10
        reg = (xbasis_time_val ** 2 * w).sum() * h

        return integral.sum() + baserate, reg
    
    
    def sample_intensity(self, points, seq, device):
        """
        return conditional intensity evaluation at grid points,

        - points: shape [ len(points), data_dim ]
        - seq: history, can be numpy array. [ batch_size=1, seq_len, data_dim ]
        """

        if not torch.is_tensor(seq):
            X = torch.tensor(seq, dtype=torch.float32).to(device)
        else:
            X = seq.to(device)
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32).to(device)
        else:
            points = points.to(device)

        batch_size, seq_len, _ = X.shape
        ts   = X[:, :, 0].clone()
        ls   = X[:, :, 1].clone()
        mask_all = ts > 0

        with torch.no_grad():

            self.nn_evaluation(X)

            ## compute time kernel
            tts   = ts.unsqueeze(0).repeat(len(points), 1, 1)                   # [ len(points), batch_size, seq_len ]
            taus = (points[:, 0] - tts.T).T                                     # [ len(points), batch_size, seq_len ]
            mask = ((taus > 0) * (taus <= self.kernel.tau_max)).bool()          # [ len(points), batch_size, seq_len ]


            ybasis_time_val = self.ybasis_time_events.clone()                   # [ n_basis_time, batch_size, seq_len ]
            xbasis_time_val = torch.zeros_like(ybasis_time_val).unsqueeze(1).repeat(1, len(points), 1, 1)
                                                                                # [ n_basis_time, len(points), batch_size, seq_len ]

            ttaus = torch.clamp(taus[mask], max=self.kernel.tau_max-0.01)       # [ mask.sum() ]
            int_start_idx = (torch.div(ttaus, self.h, rounding_mode="floor")).long()  # [ mask.sum() ]
            int_end_idx   = int_start_idx + 1
            int_start     = self.xbasis_time_grids[:, int_start_idx]            # [ n_basis_time, mask.sum() ]
            int_end       = self.xbasis_time_grids[:, int_end_idx]              # [ n_basis_time, mask.sum() ]
            int_prop      = (torch.remainder(ttaus, self.h) / self.h).unsqueeze(0).repeat(self.n_basis_time, 1)
                                                                                # [ n_basis_time, mask.sum() ]
            xbasis_time_val[:, mask] = torch.lerp(int_start, int_end, int_prop) # [ n_basis_time, mask.sum() ]

            kernel_time = xbasis_time_val * ybasis_time_val[:, None, :, :]      # [ n_basis_time, len(points), batch_size, seq_len ]

            ## compute location kernel
            lgrids = points[:, 1]                                               # [ len(points) ]
            ls_pairs = torch.stack((lgrids[:, None, None].repeat(1, batch_size, seq_len),
                               ls[None, :, :].repeat(len(points), 1, 1)), axis=-1)
                                                                                # [ len(points), batch_size, seq_len, 2 ]
            kernel_loc = []
            for filter in self.Bbasis_loc_filters:
                kernel_loc.append(filter[ls_pairs[..., 1].long(), ls_pairs[..., 0].long()])
            kernel_loc  = torch.stack(kernel_loc, axis=0)                       # [ n_basis_loc, len(points), batch_size, seq_len ]

            kernel_val = torch.einsum('ilbs,jlbs->ijlbs', kernel_time, kernel_loc)
            lams = (((kernel_val.T * self.weights_soft.T).sum((-1, -2)).T) * mask_all).sum(-1) + self.mu(lgrids[:, None])
                                                                                # [ len(points), batch_size ]

        return lams.cpu().numpy().T


    def sample_intensity_parametric(self, points, seq, device):
        """
        return conditional intensity evaluation at grid points, for parametric model

        - points: shape [ len(points), data_dim ]
        - seq: history, can be numpy array. [ batch_size=1, seq_len, data_dim ]
        """
        if not torch.is_tensor(seq):
            X = torch.tensor(seq, dtype=torch.float32).to(device)
        else:
            X = seq.to(device)
        if not torch.is_tensor(points):
            points = torch.tensor(points, dtype=torch.float32).to(device)
        else:
            points = points.to(device)

        batch_size, seq_len, _ = X.shape
        lams = []
        with torch.no_grad():
            for p in points:
                t = p[0]
                # expand point dim
                pt   = p.unsqueeze(0).repeat(batch_size, 1)                         # [ batch_size, data_dim ]
                # history points before time t (H_t)
                mask = ((X[:, :, 0] < t) * (X[:, :, 0] > self.T[0]))[:, :, None]    # [ batch_size, seq_len, data_dim ]
                ht   = X * mask                                                     # [ batch_size, seq_len, data_dim ]
                lam  = self.cond_lambda(pt, ht)                                     # [ batch_size ]
                lams.append(lam)
        lams = torch.stack(lams, 0)                                             # [ len(points), batch_size ]
        return lams.cpu().numpy().T
    

    def forward(self, X):
        """
        custom loss function
        """

        mask_all = X[:, :, 0] > 0
        self.nn_evaluation(X)

        if self.loss_type == "cont_l2_loss":
            event_lams = self.event_lams(X)
            pen_grid_lams = self.penalty_grid_lams(X)
            reg = self.time_basis_regularization(X.device)
            return ((pen_grid_lams**2).sum() * (self.T[1] - self.T[0]) / (self.pen_res_time - 1) - 2 * (event_lams * mask_all).sum()) / (self.T[1] - self.T[0]), reg, pen_grid_lams
        elif self.loss_type == "grid_l2_loss":
            _, _, l2_event_count_loss, reg = self.l2_grid_integ_loss(X)
            return l2_event_count_loss, reg
        elif self.loss_type == "likelihood":
            llk, reg = self.log_likelihood(X)
            return -llk, reg
        else:
            raise ValueError("Invalid loss function")
