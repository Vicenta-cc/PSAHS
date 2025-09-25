from torch_geometric.utils import to_scipy_sparse_matrix
import torch
def compute_homophily_list(edge_index, y, ignore_label=-1):
    num_nodes = y.size(0)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tolil()
    y = y.cpu().numpy()
    h_list = []

    for u in range(num_nodes):
        if y[u] == ignore_label:
            continue
        neighbors = adj.rows[u]
        if not neighbors:
            continue
        same = [v for v in neighbors if y[v] == y[u] and y[v] != ignore_label]
        h_u = len(same) / len(neighbors)
        h_list.append(h_u)

    return h_list

def nodewise_homophily(prob, edge_index):
    src, dst = edge_index
    sim = (prob[src] * prob[dst]).sum(dim=1)
    node_score = torch.zeros(prob.size(0), device=prob.device)
    deg = torch.zeros(prob.size(0), device=prob.device)
    node_score.index_add_(0, src, sim)
    deg.index_add_(0, src, torch.ones_like(sim))
    h_node = node_score / (deg + 1e-8)
    return h_node

def distribution_hist(h, bins=10):
    hist = torch.histc(h, bins=bins, min=0.0, max=1.0)
    hist = hist / hist.sum()
    return hist

def homophily_KL_loss(prob_src, edge_index_src, prob_tgt, edge_index_tgt, bins=10):
    h_src = nodewise_homophily(prob_src, edge_index_src)
    h_tgt = nodewise_homophily(prob_tgt, edge_index_tgt)

    p = distribution_hist(h_src, bins)
    q = distribution_hist(h_tgt, bins)

    kl = (p * (p / (q + 1e-8)).log()).sum()
    return kl

