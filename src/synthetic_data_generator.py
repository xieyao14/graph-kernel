# -*- coding: utf-8 -*-
"""
PP model on graph
"""


import numpy as np
import itertools
import arrow
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


"""# Synthetic Data Generation"""

def lebesgue_measure(S):
    """
    A helper function for calculating the Lebesgue measure for a space.
    It actually is the length of an one-dimensional space, and the area of
    a two-dimensional space.
    """
    sub_lebesgue_ms = [ sub_space[1] - sub_space[0] for sub_space in S ]
    return np.prod(sub_lebesgue_ms)


class TemporalPointProcessGenerator(object):
    """
    A stochastic temporal points generator.
    """

    def __init__(self, model, upper_bound):
        """
        Params:
        - lam:         point process model
        - upper_bound: upper bound for reject sampling
        """
        # model parameters
        self.lam         = model
        self.upper_bound = upper_bound

    def _Ogata_thinning(self, T):
        """
        To generate a realization of an Hawkes process in T, this
        function uses Ogata thining algorithm 1981.
        """
        retained_points = np.empty((0))
        # thining samples by acceptance rate.
        t = T[0]
        while t < T[1]:
            his_t = torch.FloatTensor(retained_points).unsqueeze(0).unsqueeze(-1)
            tt    = torch.FloatTensor([t]).unsqueeze(0)
            # lam_bar = self.lam.cond_lambda(tt, his_t).squeeze()
            # lam_bar = lam_bar.detach().numpy()
            lam_bar   = self.upper_bound
            u         = np.random.uniform()
            t = t - np.log(u) / lam_bar
            D         = np.random.uniform()
            # current time, location and generated historical times and locations.
            tt    = torch.FloatTensor([t]).unsqueeze(0)
            # thinning
            lam_value = self.lam.cond_lambda(tt, his_t).squeeze()
            # lam_value = (torch.nn.functional.softplus(lam_value) + 1e-5).detach().numpy()
            lam_value = lam_value.detach().numpy()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("Intensity %f is greater than upper bound %f." % (lam_value, lam_bar))
                # print("There are %d raw samples have been checked. %d samples have been retained." % \
                #     (i, retained_points.shape[0]))
                return None
            if lam_value < 0:
                print("Intensity %f is smaller than 0." % lam_value)
                return None
            # accept
            if lam_value >= D * lam_bar:
                # retained_points.append(homo_points[i])
                retained_points = np.concatenate([retained_points, [t]], axis=0)
            # monitor the process of the generation
            # if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
            #     print("[%s] %d raw samples have been checked. %d samples have been retained." % \
            #         (arrow.now(), i, retained_points.shape[0]))
        # # log the final results of the thinning algorithm
        # if verbose:
        #     print("[%s] thining samples %s based on %s." % \
        #         (arrow.now(), retained_points.shape, self.lam))
        if len(retained_points) == 0: return retained_points
        if retained_points[-1] > T[1]: return retained_points[:-1]
        else: return retained_points

    def generate(self, T=[0, 1], batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        pbar = tqdm(total = batch_size, desc="Generating sequence...")
        while b < batch_size:
            points      = self._Ogata_thinning(T)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            # print("[%s] %d-th sequence with %d points is generated." % (arrow.now(), b+1, len(points)))
            b += 1
            pbar.update(1)
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len))
        for b in range(batch_size):
            data[b, :len(points_list[b])] = points_list[b]
        return data, sizes


class SpatioTemporalPointProcessGenerator(object):
    """
    A stochastic marked spatial temporal point process generator
    """

    def __init__(self, model, upper_bound, T=[0, 1], S=[[0, 1], [0, 1]]):
        """
        Params:
        - lam:         point process model
        - upper_bound: upper bound for reject sampling
        """
        # model parameters
        self.lam         = model
        self.upper_bound = upper_bound
        self.T           = T
        self.S           = S

    def _homogeneous_poisson_sampling(self):
        """
        To generate a homogeneous Poisson point pattern in space S X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(S) occurring in S according to a
        Poisson distribution with mean lam * |S X T|.
        2. Sample each of the n location according to a uniform distribution on S
        respectively.
        Args:
            lam: intensity (or maximum intensity when used by thining algorithm)
            S:   [(min_t, max_t), (min_x, max_x), (min_y, max_y), ...] indicates the
                range of coordinates regarding a square (or cubic ...) region.
        Returns:
            samples: point process samples:
            [(t1, x1, y1), (t2, x2, y2), ..., (tn, xn, yn)]
        """
        _S     = [self.T] + self.S
        # sample the number of events from S
        n      = lebesgue_measure(_S)
        N      = np.random.poisson(size=1, lam=self.upper_bound * n)
        # simulate spatial sequence and temporal sequence separately.
        points = [ np.random.uniform(_S[i][0], _S[i][1], N) for i in range(len(_S)) ]
        points = np.array(points).transpose()
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points[:, 0].argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape))
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            x     = torch.FloatTensor(homo_points[i, :]).unsqueeze(0)
            his_x = torch.FloatTensor(retained_points).unsqueeze(0)
            # thinning
            lam_value = self.lam.cond_lambda(x, his_x).squeeze()
            lam_value = lam_value.detach().numpy()
            lam_bar   = self.upper_bound
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("Intensity %f is greater than upper bound %f." % (lam_value, lam_bar))
                print("There are %d raw samples have been checked. %d samples have been retained." % \
                    (i, retained_points.shape[0]))
                return None
            if lam_value < 0:
                print("Intensity %f is smaller than 0." % lam_value)
                return None
            # accept
            if lam_value >= D * lam_bar:
                retained_points = np.concatenate([retained_points, homo_points[[i]]], axis=0)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]))
        return retained_points

    def generate(self, batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        pbar = tqdm(total = batch_size, desc="Generating sequence...")
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling()
            points      = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            # print("[%s] %d-th sequence is generated with length %d." % (arrow.now(), b+1, len(points)))
            b += 1
            pbar.update(1)
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len, len(self.S)+1))
        for b in range(batch_size):
            data[b, :points_list[b].shape[0]] = points_list[b]
        return data, sizes


class GraphTemporalPointProcessGenerator(object):
    """
    A stochastic marked spatial temporal point process generator
    """

    def __init__(self, model, upper_bound, T=[0, 1], G=None):
        """
        Params:
        - lam:         point process model
        - upper_bound: upper bound for reject sampling
        """
        # model parameters
        self.lam         = model
        self.upper_bound = upper_bound
        self.T           = T
        self.G           = G
        self.n_node      = G.x.shape[0]

    def _homogeneous_poisson_sampling(self):
        """
        To generate a homogeneous Poisson point pattern in space G X T, it basically
        takes two steps:
        1. Simulate the number of events n = N(G X T) occurring in G X T according to a
        Poisson distribution with mean lam * |G X T|.
        2. Sample each of the n location according to a uniform distribution on G
        respectively.
        """
        # sample the number of events from S
        n      = (self.T[1] - self.T[0]) * self.n_node
        N      = np.random.poisson(size=1, lam=self.upper_bound * n)
        # simulate spatial sequence and temporal sequence separately.
        points_time = np.random.uniform(self.T[0], self.T[1], N)
        points_node = np.random.randint(low=0, high=self.n_node, size=N)
        points = np.stack((points_time, points_node), axis=0).transpose()
        # sort the sequence regarding the ascending order of the temporal sample.
        points = points[points[:, 0].argsort()]
        return points

    def _inhomogeneous_poisson_thinning(self, homo_points, verbose):
        """
        To generate a realization of an inhomogeneous Poisson process in S × T, this
        function uses a thining algorithm as follows. For a given intensity function
        lam(s, t):
        1. Define an upper bound max_lam for the intensity function lam(s, t)
        2. Simulate a homogeneous Poisson process with intensity max_lam.
        3. "Thin" the simulated process as follows,
            a. Compute p = lam(s, t)/max_lam for each point (s, t) of the homogeneous
            Poisson process
            b. Generate a sample u from the uniform distribution on (0, 1)
            c. Retain the locations for which u <= p.
        """
        retained_points = np.empty((0, homo_points.shape[1]))
        if verbose:
            print("[%s] generate %s samples from homogeneous poisson point process" % \
                (arrow.now(), homo_points.shape))
        # thining samples by acceptance rate.
        for i in range(homo_points.shape[0]):
            # current time, location and generated historical times and locations.
            x     = torch.FloatTensor(homo_points[i, :]).unsqueeze(0)
            his_x = torch.FloatTensor(retained_points).unsqueeze(0)
            # thinning
            lam_value = self.lam.cond_lambda(x, his_x).squeeze()
            lam_value = lam_value.detach().numpy()
            lam_bar   = self.upper_bound
            D         = np.random.uniform()
            # - if lam_value is greater than lam_bar, then skip the generation process
            #   and return None.
            if lam_value > lam_bar:
                print("Intensity %f is greater than upper bound %f." % (lam_value, lam_bar))
                print("There are %d raw samples have been checked. %d samples have been retained." % \
                    (i, retained_points.shape[0]))
                return None
            if lam_value < 0:
                print("Intensity %f is smaller than 0." % lam_value)
                return None
            # accept
            if lam_value >= D * lam_bar:
                retained_points = np.concatenate([retained_points, homo_points[[i]]], axis=0)
            # monitor the process of the generation
            if verbose and i != 0 and i % int(homo_points.shape[0] / 10) == 0:
                print("[%s] %d raw samples have been checked. %d samples have been retained." % \
                    (arrow.now(), i, retained_points.shape[0]))
        return retained_points

    def generate(self, batch_size=10, min_n_points=5, verbose=True):
        """
        generate spatio-temporal points given lambda and kernel function
        """
        points_list = []
        sizes       = []
        max_len     = 0
        b           = 0
        # generate inhomogeneous poisson points iterately
        pbar = tqdm(total = batch_size, desc="Generating sequence...")
        while b < batch_size:
            homo_points = self._homogeneous_poisson_sampling()
            points      = self._inhomogeneous_poisson_thinning(homo_points, verbose)
            if points is None or len(points) < min_n_points:
                continue
            max_len = points.shape[0] if max_len < points.shape[0] else max_len
            points_list.append(points)
            sizes.append(len(points))
            # print("[%s] %d-th sequence is generated with length %d." % (arrow.now(), b+1, len(points)))
            b += 1
            pbar.update(1)
        # fit the data into a tensor
        data = np.zeros((batch_size, max_len, 2))
        for b in range(batch_size):
            data[b, :points_list[b].shape[0]] = points_list[b]
        return data, sizes