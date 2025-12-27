# -*- coding: utf-8 -*-
"""
PP model on graph
"""


import numpy as np
import matplotlib.pyplot as plt
import itertools
from functools import partial
from tqdm import tqdm
tqdm = partial(tqdm, position=0, leave=True)

import networkx as nx
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.nn import GCNConv, SAGEConv, ChebConv
from torch_geometric.data import Data
from torch_geometric.utils import get_laplacian, to_dense_adj, to_networkx, degree
from torch_geometric.transforms import LaplacianLambdaMax


"""# Visualization"""

def calc_temporal_kernel(kernel, ngrid):
    tt = torch.linspace(kernel.T[0], kernel.T[1], ngrid)                        # [ ngrid ]
    ttau = torch.linspace(0, kernel.tau_max, ngrid)                             # [ ngrid ]
    t_tau_pair = torch.FloatTensor(list(itertools.product(ttau.numpy(), tt.numpy()))) # [ ngrid^2, 2 ]
    fixed_s = 0
    s  = torch.ones_like(t_tau_pair[:, 0]) * fixed_s
    xx = torch.stack([t_tau_pair[:, 0] + t_tau_pair[:, 1], s], dim=0).T         # [ ngrid^2, 2 ]
    yy = torch.stack([t_tau_pair[:, 1], s], dim=0).T                            # [ ngrid^2, 2 ]
    with torch.no_grad():
        vals = kernel(xx, yy)                                                   # [ ngrid^2 ]
        vals = vals.reshape(ngrid, ngrid).numpy().T                             # [ ngrid, ngrid ]
    return vals


def calc_graph_kernel(kernel):
    ss = torch.arange(kernel.n_node)                                            # [ n_node ]
    ssp = torch.arange(kernel.n_node)                                           # [ n_node ]
    s_sp_pair = torch.FloatTensor(list(itertools.product(ss.numpy(), ssp.numpy())))  # [ n_node^2, 2 ]
    fixed_t = 1.
    t  = torch.ones_like(s_sp_pair[:, 0]) * fixed_t
    xx = torch.stack([t, s_sp_pair[:, 0]], dim=0).T                             # [ n_node^2, 2 ]
    yy = torch.stack([t, s_sp_pair[:, 1]], dim=0).T                             # [ n_node^2, 2 ]
    with torch.no_grad():
        vals = kernel(xx, yy)                                                   # [ n_node^2 ]
        vals = vals.reshape(kernel.n_node, kernel.n_node).numpy().T             # [ n_node, n_node ]
    return vals


def plot_temporal_kernel(kernel, ngrid):

    ker_vals = calc_temporal_kernel(kernel, ngrid)
    
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(111)

    im = ax.imshow(ker_vals)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_xlabel(r"$\tau$", labelpad=-2, fontsize=12)
    ax.set_ylabel(r"$t^\prime$", labelpad=-3, fontsize=12)
    ax.set_xticks([0, ngrid-1])
    ax.set_xticklabels([0, kernel.tau_max])
    ax.set_yticks([0, ngrid-1])
    ax.set_yticklabels([kernel.T[0], kernel.T[1]])
    ax.set_title(r"Temporal $k(\cdot, \cdot, 0, 0)$", fontsize=15)
    plt.show()


def plot_graph_kernel(kernel, annotation=False):

    ker_vals = calc_graph_kernel(kernel)
    
    fig = plt.figure(figsize=(4, 3.5))
    ax = fig.add_subplot(111)

    im = ax.imshow(ker_vals)
    fig.colorbar(im, ax=ax, shrink=0.7)
    ax.set_xlabel("node", labelpad=-1, fontsize=12)
    ax.set_ylabel("node", labelpad=-1, fontsize=12)
    ax.set_xticks([0, kernel.n_node-1])
    ax.set_xticklabels([0, kernel.n_node-1])
    ax.set_yticks([0, kernel.n_node-1])
    ax.set_yticklabels([0, kernel.n_node-1])
    ax.set_title(r"Graph $k(1, 1, \cdot, \cdot)$", fontsize=15)

    if annotation:
        for i in range(kernel.n_node):
            for j in range(kernel.n_node):
                text = ax.text(j, i, np.round(ker_vals[i, j], 2),
                       ha="center", va="center", color="r")
    plt.show()


def calc_graph_pointprocess(model, points, T_plot, ngrid):
    ts      = torch.linspace(T_plot[0], T_plot[1], ngrid)
    lamvals = []
    with torch.no_grad():
        for t in ts:
            _t     = t.unsqueeze(0).repeat(model.n_node, 1)                 # [ n_node, 1 ]
            ss     = torch.arange(model.n_node).unsqueeze(-1)               # [ n_node, 1 ]
            x      = torch.cat([_t, ss], 1)                                 # [ n_node, 2 ]
            ind    = np.where((points[:, 0] < t) * (points[:, 0] > model.T[0]))[0]
            his_x  = points[ind, :]                                         # [ seq_len, 2 ]
            his_x  = his_x.unsqueeze(0).repeat(model.n_node, 1, 1)          # [ n_node, seq_len, 2 ]
            lamval = model.cond_lambda(x, his_x)                            # [ n_node ]
            lamval = lamval.numpy()
            lamvals.append(lamval)
    lamvals = np.stack(lamvals, 0).T                                        # [ n_node, ngrid ]
    return lamvals  


def plot_fitted_temporal_graph_model(model,
                                     points,
                                     T_plot,
                                     ngrid=1000,
                                     annotation=False,
                                     plot_events=False,
                                     time_kernel_ylim=None,
                                     graph_kernel_ylim=None,
                                     lam_ylim=None,
                                     filename="Epoch 0",
                                     savefig=False,
                                     savepath="aa"):
    """
    visualize the fitted model, including both the kernels and intensity.

    - points: event sequence, [ seq_len, 2 ]
    """          

    fig = plt.figure(figsize=(12, 3.5))

    ker_vals = calc_temporal_kernel(model.kernel, ngrid)
    ax1 = fig.add_subplot(131)
    if time_kernel_ylim:
        im = ax1.imshow(ker_vals, vmin=time_kernel_ylim[0], vmax=time_kernel_ylim[1])
    else:
        im = ax1.imshow(ker_vals)
    fig.colorbar(im, ax=ax1, shrink=0.7)
    ax1.set_xlabel(r"$\tau$", labelpad=-3, fontsize=12)
    ax1.set_ylabel(r"$t^\prime$", labelpad=-3, fontsize=12)
    ax1.set_xticks([0, ngrid-1])
    ax1.set_xticklabels([0, model.kernel.tau_max])
    ax1.set_yticks([0, ngrid-1])
    ax1.set_yticklabels([model.T[0], model.T[1]])
    ax1.set_title(r"Temporal $k(\cdot, \cdot, 0, 0)$", fontsize=15)

    ker_vals = calc_graph_kernel(model.kernel)
    ax2 = fig.add_subplot(132)
    if graph_kernel_ylim:
        im = ax2.imshow(ker_vals, vmin=graph_kernel_ylim[0], vmax=graph_kernel_ylim[1])
    else:
        im = ax2.imshow(ker_vals)
    fig.colorbar(im, ax=ax2, shrink=0.7)
    ax2.set_xlabel("node", labelpad=-1, fontsize=12)
    ax2.set_ylabel("node", labelpad=-1, fontsize=12)
    ax2.set_xticks([0, model.kernel.n_node-1])
    ax2.set_xticklabels([0, model.kernel.n_node-1])
    ax2.set_yticks([0, model.kernel.n_node-1])
    ax2.set_yticklabels([0, model.kernel.n_node-1])
    ax2.set_title(r"Graph $k(1, 1, \cdot, \cdot)$", fontsize=15)

    if annotation:
        for i in range(model.kernel.n_node):
            for j in range(model.kernel.n_node):
                text = ax2.text(j, i, np.round(ker_vals[i, j], 2),
                       ha="center", va="center", color="r")

    lamvals = calc_graph_pointprocess(model, points, T_plot, ngrid)
    ax3 = fig.add_subplot(133)
    axw, axh = ax3.figure.get_size_inches()
    ax3.figure.set_size_inches(axw, axh / 1.2)
    lam_scale = (np.min(lamvals), np.max(lamvals))
    if lam_ylim:
        im = ax3.imshow(lamvals, vmin=lam_ylim[0], vmax=lam_ylim[1], aspect="auto")
    else:
        im = ax3.imshow(lamvals, vmin=lam_scale[0], vmax=lam_scale[1], aspect="auto")
    ax3.set_xlabel("$t$", labelpad=-1, fontsize=12)
    ax3.set_ylabel("node", labelpad=-1, fontsize=12)
    ax3.set_xticks([0, ngrid - 1])
    ax3.set_xticklabels(np.round([T_plot[0], T_plot[1]], 1))
    ax3.set_yticks([0, model.kernel.n_node-1])
    ax3.set_yticklabels([0, model.kernel.n_node-1])
    fig.colorbar(im, ax=ax3, shrink=.8)
    # ax.title.set_text('lambda at t=%.1f' % lam_ts[j])
    ax3.set_title(filename + " lambda evolution", fontsize=15)

    if plot_events:
        ps = points[(points[:, 0] > T_plot[0]) * (points[:, 0] <= T_plot[1])]
        ax3.scatter((ps[:, 0] - T_plot[0]) / (T_plot[1] - T_plot[0]) * (ngrid-1), ps[:, 1], marker="x", c="red", s=20, label="event")
        ax3.legend()

    plt.show()
    if savefig: fig.savefig("%s/%s.png" % (savepath, filename+" Intensity evolution"))

def my_draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=None,
    label_pos=0.5,
    font_size=10,
    font_color="k",
    font_family="sans-serif",
    font_weight="normal",
    alpha=None,
    bbox=None,
    horizontalalignment="center",
    verticalalignment="center",
    ax=None,
    rotate=True,
    clip_on=True,
    rad=0
):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default={})
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : string (default='k' black)
        Font color string

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (deafult=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.gca()
    if edge_labels is None:
        labels = {(u, v): d for u, v, d in G.edges(data=True)}
    else:
        labels = edge_labels
    text_items = {}
    for (n1, n2), label in labels.items():
        (x1, y1) = pos[n1]
        (x2, y2) = pos[n2]
        (x, y) = (
            x1 * label_pos + x2 * (1.0 - label_pos),
            y1 * label_pos + y2 * (1.0 - label_pos),
        )
        pos_1 = ax.transData.transform(np.array(pos[n1]))
        pos_2 = ax.transData.transform(np.array(pos[n2]))
        linear_mid = 0.5*pos_1 + 0.5*pos_2
        d_pos = pos_2 - pos_1
        rotation_matrix = np.array([(0,1), (-1,0)])
        ctrl_1 = linear_mid + rad*rotation_matrix@d_pos
        ctrl_mid_1 = 0.5*pos_1 + 0.5*ctrl_1
        ctrl_mid_2 = 0.5*pos_2 + 0.5*ctrl_1
        bezier_mid = 0.5*ctrl_mid_1 + 0.5*ctrl_mid_2
        (x, y) = ax.transData.inverted().transform(bezier_mid)
        if n1 == n2: (x, y) = (x, y+.18)

        if rotate:
            # in degrees
            angle = np.arctan2(y2 - y1, x2 - x1) / (2.0 * np.pi) * 360
            # make label orientation "right-side-up"
            if angle > 90:
                angle -= 180
            if angle < -90:
                angle += 180
            # transform data coordinate angle to screen coordinate angle
            xy = np.array((x, y))
            trans_angle = ax.transData.transform_angles(
                np.array((angle,)), xy.reshape((1, 2))
            )[0]
        else:
            trans_angle = 0.0
        # use default box of white with white border
        if bbox is None:
            bbox = dict(boxstyle="round", ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))
        if not isinstance(label, str):
            label = str(label)  # this makes "1" and 1 labeled the same

        t = ax.text(
            x,
            y,
            label,
            size=font_size,
            color=font_color,
            family=font_family,
            weight=font_weight,
            alpha=alpha,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            rotation=trans_angle,
            transform=ax.transData,
            bbox=bbox,
            zorder=1,
            clip_on=clip_on,
        )
        text_items[(n1, n2)] = t

    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    return text_items


def draw_graph_support(g, pos, node_size=50, node_labels=None, draw_node_label=False):
    """
    - g: networkx object
    - pos: node position
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # nodes
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size)
    if draw_node_label:
        nx.draw_networkx_labels(g, pos, node_labels, ax=ax, font_color="white")

    # edges
    edgelist = [(u, v) for (u, v, d) in g.edges(data=True)]
    weightlist = [2 for (u, v, d) in g.edges(data=True)]
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=edgelist, width=weightlist)

    plt.box(False)
    plt.show()


def draw_graph(g, pos, node_size=50, node_labels=None, arc_rad=0.1, draw_loop=True):
    """
    - g: networkx object
    - pos: node position
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    # nodes
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=node_size)
    nx.draw_networkx_labels(g, pos, node_labels, ax=ax, font_color="white")

    # edges
    loop_edges   = [edge for edge in g.edges() if edge[0] == edge[1]]
    curved_edges = [edge for edge in g.edges() if reversed(edge) in g.edges() and edge[0] != edge[1]]
    straight_edges = list(set(g.edges()) - set(curved_edges) - set(loop_edges))
    edge_weights = nx.get_edge_attributes(g,'edge_weight')
    loop_edge_labels = {edge: np.round(edge_weights[edge], 3) for edge in loop_edges}
    curved_edge_labels = {edge: np.round(edge_weights[edge], 3) for edge in curved_edges}
    straight_edge_labels = {edge: np.round(edge_weights[edge], 3) for edge in straight_edges}

    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=straight_edges,
                        width=np.array([5 * v for v in straight_edge_labels.values()]))
    nx.draw_networkx_edges(g, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}',
                        width=np.array([5 * v for v in curved_edge_labels.values()]))
    my_draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=curved_edge_labels,rotate=True,rad=arc_rad)
    nx.draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=straight_edge_labels,rotate=True)
    if draw_loop:
         nx.draw_networkx_edges(g, pos, ax=ax, edgelist=loop_edges, connectionstyle=f'arc3, rad = {arc_rad}',
                        width=np.array([5 * v for v in loop_edge_labels.values()]))
         my_draw_networkx_edge_labels(g, pos, ax=ax, edge_labels=loop_edge_labels,rotate=True,rad=arc_rad)

    plt.box(False)
    plt.show()


def draw_kernel_on_graph(kernel, pos, node_labels=None, draw_loop=True):
    ker_vals = calc_graph_kernel(kernel)
    edge_list = np.where(~np.isclose(ker_vals, 0, atol=1e-3))
    edge_index = torch.tensor(edge_list, dtype=torch.long)
    edge_weight = torch.tensor(ker_vals[edge_list], dtype=torch.float)
    x = torch.tensor(np.arange(ker_vals.shape[0]).reshape(-1, 1), dtype=torch.long)
    G = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    g_directed = to_networkx(G, to_undirected=False, edge_attrs=['edge_weight'])
    if node_labels is not None:
        node_labels = {n:node_labels[i] for i, n in enumerate(g_directed)}
    draw_graph(g_directed, pos, node_labels=node_labels, arc_rad=0.2, draw_loop=draw_loop)

