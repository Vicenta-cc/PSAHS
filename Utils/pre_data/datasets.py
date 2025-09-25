from torch.utils.data import Dataset
import pandas as pd
import torch
from scipy.sparse import csr_matrix, lil_matrix, diags
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
from torch_geometric.data import Data
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import to_scipy_sparse_matrix
from sklearn.metrics.pairwise import cosine_similarity
import scipy.io as sio
from torch_geometric.transforms import OneHotDegree
import random
import math
import torch.nn.functional as F
from sparsebm import generate_SBM_dataset
import networkx as nx
import scipy
import csv
import json
import os
import sys
from torch_geometric.io import read_txt_array

sys.path.append('..')


def adjust_graph_structure_fast_source(graph, h_thresh=0.6):
    edge_index = graph.edge_index
    label = graph.y.cpu().numpy()
    num_nodes = graph.num_nodes

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    new_adj = adj.copy().astype(np.float32).tolil()

    node_homophily = np.zeros(num_nodes)
    for i in range(num_nodes):
        neighbors = adj[i].indices
        if len(neighbors) > 0:
            same_class = (label[neighbors] == label[i]).sum()
            node_homophily[i] = same_class / len(neighbors)

    label_to_nodes = {}
    for c in np.unique(label):
        nodes = np.where(label == c)[0]
        sorted_indices = np.argsort(node_homophily[nodes])[::-1]
        label_to_nodes[c] = nodes[sorted_indices]

    for u in range(num_nodes):
        neighbors = adj[u].indices
        if len(neighbors) == 0:
            continue

        h_u = node_homophily[u]

        if h_u >= h_thresh:
            continue

        alpha_u = (h_thresh - h_u) / (1 - h_thresh * h_u)

        same_mask = (label[neighbors] == label[u])
        diff_mask = ~same_mask

        same_neighbors = neighbors[same_mask]
        diff_neighbors = neighbors[diff_mask]
        new_adj[u, same_neighbors] = 1 + alpha_u
        new_adj[u, diff_neighbors] = 1 - alpha_u

        degree_u = len(neighbors)
        num_new = int(degree_u * (1 - h_u))
        if num_new <= 0:
            continue

        candidates = label_to_nodes[label[u]]


        mask = np.isin(candidates, neighbors, assume_unique=True, invert=True)
        mask &= (candidates != u)
        available = candidates[mask]


        if len(available) > 0:
            num_to_add = min(num_new, len(available))
            for v in available[:num_to_add]:
                new_adj[u, v] = alpha_u

    new_adj = new_adj.tocsr()
    row, col = new_adj.nonzero()
    edge_index_new = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(new_adj.data, dtype=torch.float)


    graph.mod_edge_index = edge_index_new
    graph.mod_edge_weight = edge_weight
    return graph


def adjust_graph_structure_fast_target(graph, h_thresh=0.6):

    edge_index = graph.edge_index
    label = graph.y_hat.cpu().numpy()
    num_nodes = graph.num_nodes


    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    new_adj = adj.copy().astype(np.float32).tolil()


    node_homophily = np.zeros(num_nodes)
    for i in range(num_nodes):
        neighbors = adj[i].indices
        if len(neighbors) > 0:
            same_class = (label[neighbors] == label[i]).sum()
            node_homophily[i] = same_class / len(neighbors)

    label_to_nodes = {}
    for c in np.unique(label):
        nodes = np.where(label == c)[0]

        sorted_indices = np.argsort(node_homophily[nodes])[::-1]
        label_to_nodes[c] = nodes[sorted_indices]

    for u in range(num_nodes):
        neighbors = adj[u].indices
        if len(neighbors) == 0:
            continue


        h_u = node_homophily[u]

        if h_u >= h_thresh:
            continue

        alpha_u = (h_thresh - h_u) / (1 - h_thresh * h_u)
        same_mask = (label[neighbors] == label[u])
        diff_mask = ~same_mask

        same_neighbors = neighbors[same_mask]
        diff_neighbors = neighbors[diff_mask]
        new_adj[u, same_neighbors] = 1 + alpha_u
        new_adj[u, diff_neighbors] = 1 - alpha_u

        degree_u = len(neighbors)
        num_new = int(degree_u * (1 - h_u))
        if num_new <= 0:
            continue

        candidates = label_to_nodes[label[u]]

        mask = np.isin(candidates, neighbors, assume_unique=True, invert=True)
        mask &= (candidates != u)
        available = candidates[mask]

        if len(available) > 0:
            num_to_add = min(num_new, len(available))
            for v in available[:num_to_add]:
                new_adj[u, v] = alpha_u


    new_adj = new_adj.tocsr()
    row, col = new_adj.nonzero()
    edge_index_new = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(new_adj.data, dtype=torch.float)


    graph.mod_edge_index = edge_index_new
    graph.mod_edge_weight = edge_weight
    return graph


def adjust_graph_structure_fast_target_Plabel(graph, h_thresh=0.6):

    edge_index = graph.edge_index
    label = graph.y_hat.cpu().numpy()
    num_nodes = graph.num_nodes

    valid_label_mask = (label != -1)


    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    new_adj = adj.copy().astype(np.float32).tolil()  # 使用浮点类型存储权重


    node_homophily = np.full(num_nodes, -1.0)  # 初始化为-1表示无效
    for i in range(num_nodes):

        if not valid_label_mask[i]:
            continue

        neighbors = adj[i].indices
        valid_neighbors = neighbors[valid_label_mask[neighbors]]

        if len(valid_neighbors) > 0:
            same_class = (label[valid_neighbors] == label[i]).sum()
            node_homophily[i] = same_class / len(valid_neighbors)

    label_to_nodes = {}
    unique_labels = np.unique(label[valid_label_mask])

    for c in unique_labels:
        nodes = np.where((label == c) & valid_label_mask)[0]
        sorted_indices = np.argsort(node_homophily[nodes])[::-1]
        label_to_nodes[c] = nodes[sorted_indices]


    for u in range(num_nodes):
        if not valid_label_mask[u] or node_homophily[u] < 0:
            continue
        neighbors = adj[u].indices
        if len(neighbors) == 0:
            continue
        h_u = node_homophily[u]
        if h_u >= h_thresh:
            continue

        alpha_u = (h_thresh - h_u) / (1 - h_thresh * h_u)
        valid_neighbors_mask = valid_label_mask[neighbors]
        same_mask = (label[neighbors] == label[u]) & valid_neighbors_mask
        diff_mask = (label[neighbors] != label[u]) & valid_neighbors_mask

        same_neighbors = neighbors[same_mask]
        diff_neighbors = neighbors[diff_mask]
        new_adj[u, same_neighbors] = 1 + alpha_u
        new_adj[u, diff_neighbors] = 1 - alpha_u

        degree_u = len(neighbors)
        num_new = int(degree_u * (1 - h_u))
        if num_new <= 0:
            continue


        if label[u] in label_to_nodes:
            candidates = label_to_nodes[label[u]]
        else:
            continue

        mask = np.isin(candidates, neighbors, assume_unique=True, invert=True)
        mask &= (candidates != u)
        available = candidates[mask]

        if len(available) > 0:
            num_to_add = min(num_new, len(available))
            for v in available[:num_to_add]:

                if valid_label_mask[v]:
                    new_adj[u, v] = alpha_u

    new_adj = new_adj.tocsr()
    row, col = new_adj.nonzero()
    edge_index_new = torch.tensor([row, col], dtype=torch.long)
    edge_weight = torch.tensor(new_adj.data, dtype=torch.float)

    graph.mod_edge_index = edge_index_new
    graph.mod_edge_weight = edge_weight
    return graph


def prepare_dblp_acm(raw_dir, name):
    docs_path = os.path.join(raw_dir, name, 'raw/{}_docs.txt'.format(name))
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)
    x = torch.from_numpy(x).to(torch.float)

    edge_path = os.path.join(raw_dir, name, 'raw/{}_edgelist.txt'.format(name))
    edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()

    num_node = x.size(0)
    data = np.ones(edge_index.size(1))
    adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                        shape=(num_node, num_node))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    label_path = os.path.join(raw_dir, name, 'raw/{}_labels.txt'.format(name))
    f = open(label_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        line = line.replace("\r", "").replace("\n", "")
        content_list.append(line)
    y = np.array(content_list, dtype=int)

    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)
    y = torch.from_numpy(y).to(torch.int64)
    graph = Data(edge_index=edge_index, x=x, y=y)
    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = torch.full_like(y, -1)
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)

    graph.mod_edge_index = edge_index
    graph.mod_edge_weight = torch.ones(graph.num_edges)

    return graph


def load_data_from_mat(raw_dir, name):

    docs_path = os.path.join(raw_dir, '{}.mat'.format(name))
    data_mat = sio.loadmat(docs_path)

    adj = data_mat['network']
    features = data_mat['attrb']
    labels = data_mat['group']
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    adj_dense = np.array(adj.todense())
    edges = np.vstack(np.where(adj_dense)).T
    edge_index = torch.tensor(edges, dtype=torch.long).t()

    features = torch.FloatTensor(np.array(features, dtype=np.float32))
    labels = np.argmax(labels, 1)
    labels = torch.LongTensor(labels)

    num_class = np.unique(labels.numpy())
    class_index = []
    for i in num_class:
        c_i = np.where(labels.numpy() == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])

    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)

    graph = Data(edge_index=edge_index, x=features, y=labels)

    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.y_hat = torch.full_like(labels, -1)
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)
    graph.mod_edge_index = edge_index
    graph.mod_edge_weight = torch.ones(graph.num_edges)

    def prepare_airport(raw_dir, name):
        label_path = os.path.join(raw_dir, name, 'raw/{}_labels.txt'.format(name))
        f = open(label_path, 'rb')
        content_list = []
        for line in f.readlines():
            line = str(line, encoding="utf-8")
            line = line.replace("\r", "").replace("\n", "")
            content_list.append(line)
        y = np.array(content_list, dtype=int)
        # y = torch.from_numpy(y).to(torch.int64)
        num_node = len(y)
        edge_path = os.path.join(raw_dir, name, 'raw/{}_edgelist.txt'.format(name))
        edge_index = read_txt_array(edge_path, sep=',', dtype=torch.long).t()
        data = np.ones(edge_index.size(1))
        adj = sp.csr_matrix((data, (edge_index[0], edge_index[1])),
                            shape=(num_node, num_node))
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        num_class = np.unique(y)
        class_index = []
        for i in num_class:
            c_i = np.where(y == i)[0]
            class_index.append(c_i)

        training_mask = np.array([])
        validation_mask = np.array([])
        testing_mask = np.array([])
        tgt_validation_mask = np.array([])
        tgt_testing_mask = np.array([])
        for idx in class_index:
            np.random.shuffle(idx)
            training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
            validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
            testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
            tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
            tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

        training_mask = training_mask.astype(int)
        testing_mask = testing_mask.astype(int)
        validation_mask = validation_mask.astype(int)

        y = torch.from_numpy(y).to(torch.int64)

        # Apply OneHotDegree
        graph = Data(edge_index=edge_index, y=y, x=torch.ones((num_node, 1)))
        one_hot_transform = OneHotDegree(241)
        graph = one_hot_transform(graph)

        graph.source_training_mask = training_mask
        graph.source_validation_mask = validation_mask
        graph.source_testing_mask = testing_mask
        graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
        graph.target_validation_mask = tgt_validation_mask
        graph.target_testing_mask = tgt_testing_mask
        graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
        graph.adj = adj
        graph.num_classes = len(num_class)
        graph.edge_weight = torch.ones(graph.num_edges)
        graph.y_hat = torch.full_like(y, -1)
        graph.num_classes = len(num_class)
        graph.mod_edge_index = edge_index
        graph.mod_edge_weight = torch.ones(graph.num_edges)

        return graph

    return graph

def prepare_Twitch(raw_dir, lang):
    # assert lang in ('DE', 'EN', 'FR'), 'Invalid dataset'
    label_path = os.path.join(raw_dir, lang, 'raw/musae_{}_target.csv'.format(lang))
    features_path = os.path.join(raw_dir, lang, 'raw/musae_{}_features.json'.format(lang))
    edges_path = os.path.join(raw_dir, lang, 'raw/musae_{}_edges.csv'.format(lang))
    label = []
    node_ids = []
    src = []
    targ = []
    uniq_ids = set()
    with open(label_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            node_id = int(row[5])
            # handle FR case of non-unique rows
            if node_id not in uniq_ids:
                uniq_ids.add(node_id)
                label.append(int(row[2] == "True"))
                node_ids.append(int(row[5]))

    node_ids = np.array(node_ids, dtype=int)

    with open(edges_path) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            src.append(int(row[0]))
            targ.append(int(row[1]))

    with open(features_path) as f:
        j = json.load(f)

    src = np.array(src)
    targ = np.array(targ)
    label = np.array(label)

    edge_index_np = np.vstack((src, targ))

    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    inv_node_ids = {node_id: idx for (idx, node_id) in enumerate(node_ids)}
    reorder_node_ids = np.zeros_like(node_ids)
    for i in range(label.shape[0]):
        reorder_node_ids[i] = inv_node_ids[i]

    n = label.shape[0]
    adj = sp.csr_matrix((np.ones(len(src)), (np.array(src), np.array(targ))), shape=(n, n))
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = np.zeros((n, 3170))
    for node, feats in j.items():
        if int(node) >= n:
            continue
        features[int(node), np.array(feats, dtype=int)] = 1
    features = np.array(features)
    x = torch.from_numpy(features).to(torch.float)

    new_label = label[reorder_node_ids]
    label = new_label
    y = torch.from_numpy(label).to(torch.int64)

    num_class = np.unique(y)
    class_index = []
    for i in num_class:
        c_i = np.where(y == i)[0]
        class_index.append(c_i)

    training_mask = np.array([])
    validation_mask = np.array([])
    testing_mask = np.array([])
    tgt_validation_mask = np.array([])
    tgt_testing_mask = np.array([])
    for idx in class_index:
        np.random.shuffle(idx)
        training_mask = np.concatenate((training_mask, idx[0:int(len(idx) * 0.6)]), 0)
        validation_mask = np.concatenate((validation_mask, idx[int(len(idx) * 0.6):int(len(idx) * 0.8)]), 0)
        testing_mask = np.concatenate((testing_mask, idx[int(len(idx) * 0.8):]), 0)
        tgt_validation_mask = np.concatenate((tgt_validation_mask, idx[0:int(len(idx) * 0.2)]), 0)
        tgt_testing_mask = np.concatenate((tgt_testing_mask, idx[int(len(idx) * 0.2):]), 0)

    training_mask = training_mask.astype(int)
    testing_mask = testing_mask.astype(int)
    validation_mask = validation_mask.astype(int)

    graph = Data(edge_index=edge_index, x=x, y=y)
    graph.source_training_mask = training_mask
    graph.source_validation_mask = validation_mask
    graph.source_testing_mask = testing_mask
    graph.source_mask = np.concatenate((training_mask, validation_mask, testing_mask), 0)
    graph.target_validation_mask = tgt_validation_mask
    graph.target_testing_mask = tgt_testing_mask
    graph.target_mask = np.concatenate((tgt_validation_mask, tgt_testing_mask), 0)
    graph.adj = adj
    graph.num_classes = len(num_class)
    graph.edge_weight = torch.ones(graph.num_edges)
    graph.y_hat = torch.full_like(y, -1)
    graph.num_classes = len(num_class)
    graph.mod_edge_index = edge_index
    graph.mod_edge_weight = torch.ones(graph.num_edges)

    return graph



