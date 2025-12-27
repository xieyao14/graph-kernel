import numpy as np
import matplotlib.pyplot as plt
import itertools
from functools import partial
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

# Internal imports based on your package structure
from model import (
    BaseExponentialCosineBasis,
    TemporalParametricKernelChebnetLocalFilterOnGraph,
    TemporalPointProcessOnGraph,
    TemporalDeepBasisL3netLocalFilterOnGraphKernel
)
from utils import config_generate, eval_points_generate
from visualization import calc_graph_kernel, calc_graph_pointprocess
from synthetic_data_generator import GraphTemporalPointProcessGenerator

tqdm = partial(tqdm, position=0, leave=True)


"""# Evaluation Functions"""

def predict_next_events(model, seq, T_end, t_grid=100, true_model=False):
    """
    Predict the next event time and node.
    - model: learned_model
    - seq: a tensor of events, [ seq_len, data_dim ]
    """
    seq = seq[seq[:, 0] > 0]
    last_time = seq[-1, 0].item()
    ts = (np.linspace(last_time, T_end, t_grid+1) + (T_end-last_time) / t_grid / 2)[:-1]
    ms = np.arange(model.n_node)
    points = torch.Tensor(list(itertools.product(ts, ms)))

    if true_model:
        lams = model.sample_intensity_parametric(points, seq.unsqueeze(0), device="cpu")
    else:
        lams = model.sample_intensity(points, seq.unsqueeze(0), device="cpu")
    
    lams = lams.reshape(1, t_grid, len(ms))[0]                                  # [ t_grid, n_node ]
    tautau = ts
    # Probability density function f(t)
    f = (lams * np.exp(-(lams.cumsum(0) * ((T_end - last_time) / t_grid)).sum(1))[:, None])
                                                                                # [ t_grid, n_mark ]

    # Predicted time (expectation)
    t_p = (tautau * f.sum(-1) * (T_end - last_time) / t_grid).sum()
    # Predicted node (max probability)
    e_p = f.sum(0).argmax()

    return t_p, e_p

def predict_next_events_rolling(model, seq, T_plus=10, t_grid=100, n_pred=10, true_model=False):
    """
    Rolling prediction of next events.
    - model: learned_model
    - seq: a tensor of events, [ seq_len, data_dim ]
    """
    seq = seq[seq[:, 0] > 0]
    t_true = seq[-n_pred:, 0].numpy()
    e_true = seq[-n_pred:, 1].numpy()
    t_pred = []
    e_pred = []

    for i in range(n_pred):
        last_time = seq[-n_pred-1 + i, 0].item()
        T_end = last_time + T_plus
        ts = (np.linspace(last_time, T_end, t_grid+1) + (T_end-last_time) / t_grid / 2)[:-1]
        ms = np.arange(model.n_node)
        points = torch.Tensor(list(itertools.product(ts, ms)))

        if true_model:
            lams = model.sample_intensity_parametric(points, seq[:-n_pred+i].unsqueeze(0), device="cpu")
        else:
            lams = model.sample_intensity(points, seq[:-n_pred+i].unsqueeze(0), device="cpu")
        
        lams = lams.reshape(1, t_grid, len(ms))[0]                                  # [ t_grid, n_node ]
        tautau = ts
        f = (lams * np.exp(-(lams.cumsum(0) * ((T_end - last_time) / t_grid)).sum(1))[:, None])
                                                                                    # [ t_grid, n_node ]

        # Predicted time
        t_p = (tautau * f.sum(-1) * (T_end - last_time) / t_grid).sum()
        # Predicted node
        e_p = f.sum(0).argmax()

        t_pred.append(t_p)
        e_pred.append(e_p)

    t_pred = np.array(t_pred)
    e_pred = np.array(e_pred)

    return t_true, e_true, t_pred, e_pred


"""# Main Execution"""

if __name__ == "__main__":
    seed = 100
    torch.manual_seed(seed)
    np.random.seed(seed)

    ## Model Configuration
    T = [0., 50.]
    tau_max  = 5.
    mu       = .1
    n_basis_time = 1
    n_basis_loc  = 3
    data_dim = 2
    loss_type    = "likelihood"
    use_seq      = 500
    data_name    = "data-basecos_exp-ring_graph-V16-mu0.100-alpha1.500-beta2.000-freq0.200-nt1-nl3-n1000"
    model_name   = "Model_L3_%s_for_%s_nbasistime%d_nbasisloc%d_seqs%d" % (loss_type, data_name, n_basis_time, n_basis_loc, use_seq)
    
    # Paths assuming execution from root directory (Package/)
    save_path    = "results/saved_models/%s" % model_name
    graph_path   = "data/%s/graph.pt" % data_name
    data_path    = "data/%s/%s.npy" % (data_name, data_name)
    
    # Load Graph
    G = torch.load(graph_path)
    device = "cpu"

    ## 1. Define True Model (for comparison)
    timebasis = BaseExponentialCosineBasis(alpha=1.5, beta=2., freq=.2)
    basis_weight = np.array([[.2, -.3, .1]])
    kernel_true = TemporalParametricKernelChebnetLocalFilterOnGraph(
        device="cpu", T=T, tau_max=tau_max, G=G,
        time_basis_list=[timebasis], n_basis_loc=n_basis_loc,
        data_dim=data_dim, basis_weight=basis_weight, basis_dim=1,
        init_std=1e0, graph_normalize="sym"
    )
    trg_model = TemporalPointProcessOnGraph(
        device="cpu", T=T, G=G, mu=mu*np.ones(G.x.shape[0]), tau_max=tau_max,
        kernel=kernel_true, data_dim=data_dim
    )

    ## 2. Initialize and Load Learned Model
    order_list = [0, 1, 2]
    kernel_learned = TemporalDeepBasisL3netLocalFilterOnGraphKernel(
        T=T, G=G, tau_max=tau_max, device=device,
        n_basis_time=n_basis_time, loc_order_list=order_list,
        data_dim=data_dim, basis_dim=1, nn_width_basis_time=32,
        init_gain=5e-1, init_bias=1e-2, init_std=1e-2
    )
    init_model = TemporalPointProcessOnGraph(
        device=device, T=T, G=G, mu=mu*np.ones(G.x.shape[0]), tau_max=tau_max, loss=loss_type,
        kernel=kernel_learned, data_dim=data_dim, numerical_int=True,
        eval_res=100, int_res=100, int_res_loc=200,
        pen_res_time=50, l2_res_time=200
    )
    
    # Load weights
    # Note: Ensure the epoch number (299) matches a saved checkpoint you actually have
    checkpoint_path = "%s/%s-%d.pth" % (save_path, model_name, 299)
    print(f"Loading model from {checkpoint_path}...")
    try:
        init_model.load_state_dict(torch.load(checkpoint_path))
    except FileNotFoundError:
        print("Checkpoint not found. Please check the path or train the model first.")
        exit()

    ## 3. Data Loading
    data = np.load(data_path)
    data = torch.FloatTensor(data)
    # train_data = data[:int(use_seq*0.8)] # Unused in evaluation script
    test_data = data[900:1000]
    plot_points = test_data[0]

    ## 4. Evaluation: Log-Likelihood
    n_time_grid = 100
    test_loader = DataLoader(torch.utils.data.TensorDataset(test_data),
                             shuffle=False, batch_size=32, drop_last=False)
    ts = np.linspace(T[0], T[1], n_time_grid)
    ns = np.arange(G.x.shape[0])
    eval_points = eval_points_generate(ts, ns)

    learned_lams = []
    print("Evaluating intensity on test set...")
    for batch in test_loader:
        learned_lams.append(init_model.sample_intensity(eval_points, batch[0], device="cpu"))
    learned_lams = np.concatenate(learned_lams, axis=0) # [ n_test_seq, len(eval_points) ]
    learned_lams = learned_lams.reshape(-1, n_time_grid, 16)

    learned_lamvals = 0
    for seq in test_data:
        seq = seq[seq[:, 0] > 0]
        # Evaluate intensity at event times
        learned_lamvals += np.log(init_model.sample_intensity(seq, seq.unsqueeze(0), device="cpu")).sum()

    # Calculate Log-Likelihood per event
    test_llk = (learned_lamvals - (learned_lams[:, :-1, :] * (T[1] - T[0]) / (n_time_grid-1)).sum()) / (test_data[:, :, 0] > 0).sum()
    print(f"Test Log-Likelihood per event: {test_llk}")

    ## 5. Generation & Statistics (MAE / KLD)
    print("Generating synthetic data from learned model...")
    torch.manual_seed(1000)
    np.random.seed(1000)
    
    batch_size_gen = 100
    T_gen = [0., 50.]
    generator = GraphTemporalPointProcessGenerator(init_model, upper_bound=8e-1, T=T_gen, G=G)
    data_npp, size = generator.generate(batch_size=batch_size_gen, min_n_points=1, verbose=False)
    
    np.save(save_path + "/model_generated_data.npy", data_npp)

    # MAE of event count
    time_mae_fnn = (data_npp[:, :, 0] > 0).sum(-1).mean() / T_gen[1] - (test_data.numpy()[:, :, 0] > 0).sum(-1).mean().item() / T[1]
    print(f"Event Count MAE: {time_mae_fnn}")

    # KLD of event types
    all_event = data_npp[data_npp[:, :, 0] > 0]
    v1, c1 = np.unique(all_event[:, 1], return_counts=True)
    type_freq1 = c1 / len(all_event)

    all_event2 = test_data[test_data[:, :, 0] > 0]
    v2, c2 = np.unique(all_event2[:, 1], return_counts=True)
    test_type_dict = {v: c for v, c in zip(v2, c2)}
    c2_full = np.array([test_type_dict.get(v, 0) for v in v1])
    type_freq2 = c2_full / len(all_event2)

    # Ensure distributions are aligned (handle missing nodes in generation if any)
    # Simple check assuming all nodes appear in both sets for this example
    if len(type_freq1) == len(type_freq2):
        KLD_fnn = (type_freq1 * np.log(type_freq1 / type_freq2)).sum()
        print(f"Event Type KLD: {KLD_fnn}")
    else:
        print("Warning: Node mismatch in KLD calculation. Skipping KLD.")

    ## 6. Visualization
    print("Generating plots...")
    
    # 6a. Learned Model Visualization
    graph_kernel_vals = calc_graph_kernel(init_model.kernel).T
    events = plot_points[plot_points[:, 0] > 0]
    T_plot = [25., 40.]
    ngrid = 200
    lam_vals = calc_graph_pointprocess(init_model, events, T_plot=T_plot, ngrid=ngrid)

    # Dependency matrix calculation (example subset)
    select_event_idx = np.arange(0, min(76, len(events)), 1)
    n_dep = len(select_event_idx)
    idx_pair = np.array(list(itertools.product(select_event_idx, select_event_idx)))
    event_dep_1 = events[idx_pair[:, 0]]
    event_dep_2 = events[idx_pair[:, 1]]
    
    with torch.no_grad():
        dep_vals = init_model.kernel(event_dep_1, event_dep_2).numpy()
        dep_vals = dep_vals.reshape(n_dep, n_dep)
        upp_idx = np.bool_(np.triu(np.ones((n_dep, n_dep))))
        dep_vals[upp_idx] = 0.

    fig = plt.figure(figsize=(12, 4))
    
    # Kernel Plot
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(graph_kernel_vals, cmap="Blues", vmin=0.)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("node", fontsize=18)
    ax1.set_ylabel("node", fontsize=18)

    # Dependency Plot
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.log10(dep_vals+1e-5)+5, cmap="Greys", vmin=1.0)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("event", fontsize=18)
    ax2.set_ylabel("event", fontsize=18)

    # Intensity Plot
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(lam_vals, cmap="Reds", aspect=ngrid/len(ns)) # Fixed aspect ratio
    plot_events = events[(events[:, 0] >= T_plot[0]) * (events[:, 0] <= T_plot[1])]
    ax3.scatter((plot_events[:, 0] - T_plot[0]) / (T_plot[1] - T_plot[0]) * (ngrid-1), 
                plot_events[:, 1], marker="x", c="black", alpha=0.4, s=30, linewidth=1.0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("time", fontsize=18)
    ax3.set_ylabel("node", fontsize=18)

    fig.savefig("results/GraDK_16-node-graph.pdf")
    print("Saved results/GraDK_16-node-graph.pdf")

    # 6b. True Model Visualization
    graph_kernel_vals = calc_graph_kernel(trg_model.kernel).T
    lam_vals = calc_graph_pointprocess(trg_model, events, T_plot=T_plot, ngrid=ngrid)

    with torch.no_grad():
        dep_vals = trg_model.kernel(event_dep_1, event_dep_2).numpy()
        dep_vals = dep_vals.reshape(n_dep, n_dep)
        upp_idx = np.bool_(np.triu(np.ones((n_dep, n_dep))))
        dep_vals[upp_idx] = 0.

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.imshow(graph_kernel_vals, cmap="Blues")
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel("node", fontsize=18)
    ax1.set_ylabel("node", fontsize=18)

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.imshow(np.log10(dep_vals+1e-5)+5, cmap="Greys")
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("event", fontsize=18)
    ax2.set_ylabel("event", fontsize=18)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.imshow(lam_vals, cmap="Reds", aspect=ngrid/len(ns)) # Fixed aspect ratio
    plot_events = events[(events[:, 0] >= T_plot[0]) * (events[:, 0] <= T_plot[1])]
    ax3.scatter((plot_events[:, 0] - T_plot[0]) / (T_plot[1] - T_plot[0]) * (ngrid-1), 
                plot_events[:, 1], marker="x", c="black", alpha=0.4, s=30, linewidth=1.0, label="event")
    ax3.legend(loc="upper left", fontsize=15, handlelength=1.0)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel("time", fontsize=18)
    ax3.set_ylabel("node", fontsize=18)

    fig.savefig("results/True_16-node-graph.pdf")
    print("Saved results/True_16-node-graph.pdf")