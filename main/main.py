import json
import torch
import argparse
from torch_geometric.data import DataLoader as DataLoader_graph
import matplotlib.pyplot as plt
import copy
import math
import numpy as np
import random
import os
from sklearn.metrics import accuracy_score
import sys
import torch.nn.functional as F
import models


sys.path.append('..')

from Utils.pre_data import datasets
from Utils import utils
import optuna
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    # utils.parse_optimizer(parser)
    parser.add_argument('-d', '--dataset', type=str, help='dataset used', default='Blog')
    parser.add_argument('-m', '--method', type=str, help='method used', default='DANN_rw')
    parser.add_argument('-b', '--backbone', type=str, help='backbone used', default='GCN')
    parser.add_argument('--conv_type', type=str, help='convolution type of GNNs', default='dir-gcn-weighted')
    parser.add_argument('--jk', type=str, help='jumping_knowledge', default='cat')
    parser.add_argument('--alpha', type=float, default=0.5,help='Convex combination weight between source-to-target and target-to-source aggregation.')
    parser.add_argument('--learn_alpha',type=bool,help='Whether to learn the alpha parameter during training.',default=False)
    parser.add_argument('--seed', type=int, help='note this should be number of seeds', default=5)
    parser.add_argument('--gpu_ratio', type=float, help='gpu memory ratio', default=None)


    parser.add_argument('--batch_size', type=int, help='Training batch size', default=1)
    parser.add_argument('--num_layers', type=int, help='Number of embedding layers', default=0)
    parser.add_argument('--dc_layers', type=int, help='Number of domain classification layers', default=2)
    parser.add_argument('--class_layers', type=int, help='Number of classification layers', default=2)
    parser.add_argument('--K', type=int, help='Number of GNN layers', default=2)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension for GNN', default=50)
    parser.add_argument('--conv_dim', type=int, help='hidden dimension for classification layer', default=50)
    parser.add_argument('--cls_dim', type=int, help='hidden dimension for feature extractor layer', default=50)
    parser.add_argument('--bn', type=bool, help='if use batch normalization', default=False)
    parser.add_argument("--resnet", type=bool, help='if we want to use resnet', default=False)
    parser.add_argument('--best_model', type=bool, help='if use best model from validation results', default=True)
    parser.add_argument('--valid_data', type=str,
                        help='src means use source valid to select model, tgt means use target valid', default='tgt')

    parser.add_argument('--epochs', type=int, help='Number of training epochs', default=300)
    parser.add_argument('--opt', type=str, help='optimizer', default='adam')
    parser.add_argument('--opt_scheduler', type=str, help='optimizer scheduler', default='step')
    parser.add_argument('--opt_decay_step', type=int, default=50)
    parser.add_argument('--opt_decay_rate', type=int, default=0.8)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--lr', type=float, help='Learning rate', default=0.007)
    parser.add_argument('--mlp_conv_dim', type=int, help='hidden dimension for classification layer', default=128)
    parser.add_argument('--mlp_cls_dim', type=int, help='hidden dimension for feature extractor layer', default=64)
    parser.add_argument('--mlp_class_layers', type=int, help='Number of classification layers', default=2)

    parser.add_argument('--num_nodes', type=int, help='number of nodes for each block of Stochastic block model',
                        default=1000)
    parser.add_argument("--ps", type=float, help="the source intraclass connection probability for CSBM dataset",
                        default=0.2)
    parser.add_argument("--qs", type=float, help="the source interclass connection probability for CSBM dataset",
                        default=0.02)
    parser.add_argument("--pt", type=float, help="the target intraclass connection probability for CSBM dataset",
                        default=0.02)
    parser.add_argument("--qt", type=float, help="the target interclass connection probability for CSBM dataset",
                        default=0.2)
    parser.add_argument('--sigma', type=float, help='sigma(std) in the stochastic block model', default=0.8)
    parser.add_argument('--theta', type=float, help='theta in the stochastic block model', default=0)
    parser.add_argument('--sbm_alpha', type=float, help='alpha in the stochastic block model', default=0)

    parser.add_argument('--num_events', type=int, help='number of events for Pileup dataset', default=100)
    parser.add_argument('--edge_feature', type=bool, help='if we want to use edge feature during convolution',
                        default=False)

    parser.add_argument('--reweight', type=bool, help='if we want to reweight the source graph', default=False)
    parser.add_argument('--rw_freq', type=int,
                        help='every number of epochs to compute the new weights based on psuedo-labels', default=20)
    parser.add_argument('--start_epoch', type=int, help='starting epoch for reweighting', default=200)
    parser.add_argument("--rw_lmda", type=float, help='lambda to control the rw', default=0.5)
    parser.add_argument("--use_valid_label", type=bool,
                        help='if we want to use target validation ground truth label in rw', default=True)
    parser.add_argument('--pseudo', type=bool, help='if use pseudo label for reweighting', default=False)
    parser.add_argument("--rw_average", type=bool,
                        help='if we want to average the edge weights when reweighting for multiple graphs',
                        default=False)
    parser.add_argument("--alphatimes", type=float, help='constant in front of the alpha for DANN', default=1)
    parser.add_argument("--alphamin", type=float, help='min of alpha for DANN', default=0.2)

    parser.add_argument("--h_threshold", type=float, help='threshold of homophily', default=0.6)

    parser.add_argument('--dir_name', type=str, default='')
    parser.add_argument("--src_name", type=str, help='specify for the source dataset name', default='Blog1')
    parser.add_argument("--tgt_name", type=str, help='specify for the target dataset name', default='Blog2')
    parser.add_argument("--domain_split", type=str, help='specify for the cora split', default='word')
    parser.add_argument('--start_year', type=int, help='training year start for arxiv', default=2005)
    parser.add_argument('--end_year', type=int, help='training year end for arxiv', default=2007)
    parser.add_argument("--train_sig", type=str, help='training signal for PU dataset', default='gg')
    parser.add_argument("--test_sig", type=str, help='testing signal for PU dataset', default='qq')
    parser.add_argument("--train_PU", type=int, help='training PU level for PU dataset', default=10)
    parser.add_argument("--test_PU", type=int, help='testing PU level for PU dataset', default=30)
    parser.add_argument("--balanced", type=bool, help='if we keep the label balanced in pileup dataset', default=True)
    parser.add_argument('--plt', type=str, help='plot using tsne or pca', default='pca')

    return parser.parse_args()


def train_one_epoch(model, mlp_model, src_data, tgt_data, args, opt, scheduler, epoch):
    src_data = src_data.to(device)
    tgt_data = tgt_data.to(device)
    mlp_model.eval()
    with torch.no_grad():
        tgt_mlp_embed, tgt_mlp_pred = mlp_model.forward(tgt_data)

    _, mlp_pred_tgt = tgt_mlp_pred.max(dim=1)
    if args.reweight and epoch >= (args.start_epoch - 1):

        _, [pred_tgt, _] = model.forward(tgt_data, 1)
        confidence, pred_tgt = pred_tgt.max(dim=1)
        mask = pred_tgt == mlp_pred_tgt
        tgt_data.y_hat[mask] = mlp_pred_tgt[mask]
        if (epoch - 1) % args.rw_freq == 0:

            datasets.adjust_graph_structure_fast_target_Plabel(tgt_data, h_thresh=args.h_threshold)
            tgt_data = tgt_data.to(device)


    DA_alpha = min((args.alphatimes * (epoch + 1) / args.epochs), args.alphamin)

    [GNN_embed_src, final_embed_src], [pred_src, pred_domain_src] = model.forward(src_data, DA_alpha)
    [GNN_embed_tgt, final_embed_tgt], [pred_tgt, pred_domain_tgt] = model.forward(tgt_data, DA_alpha)

    mask_src = src_data.source_training_mask
    label_src = src_data.y[mask_src]
    pred_src = pred_src[mask_src]
    pred_domain_src = pred_domain_src[src_data.source_mask]
    pred_domain_tgt = pred_domain_tgt[tgt_data.target_mask]
    domain_label_src = torch.zeros_like(pred_domain_src)
    domain_label_tgt = torch.ones_like(pred_domain_tgt)

    cls_loss_src = utils.CE_loss(pred_src, label_src)
    domain_loss_src = utils.BCE_loss(pred_domain_src, domain_label_src)
    domain_loss_tgt = utils.BCE_loss(pred_domain_tgt, domain_label_tgt)

    loss = cls_loss_src + 1 * (domain_loss_src + domain_loss_tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    scheduler.step()

    return loss


def train(source_dataset, target_dataset, args, seed):
    directory = args.dir_name
    parent_dir = "./"
    path = os.path.join(parent_dir, directory)
    isdir = os.path.isdir(path)

    if isdir == False:
        os.mkdir(path)
    sys.stdout = utils.Logger(path)

    if isinstance(source_dataset, list):
        input_dim = source_dataset[0].num_node_features
        output_dim = source_dataset[0].num_classes
    else:
        input_dim = source_dataset.num_node_features
        output_dim = source_dataset.num_classes

    model = models.directed_GNN_adv(input_dim, output_dim, args)
    model = model.to(device)
    best_mlp_model = models.MLPWithMLPClassifier(input_dim, output_dim, args).to(device)
    best_mlp_model.load_state_dict(torch.load(os.path.join(path, f'B1_B2_best_mlp_model_seed{seed}.pt')))
    scheduler, opt = utils.build_optimizer(args, model.parameters())

    epochs_train = []
    loss_src_train = []
    loss_src_valid = []
    loss_src_test = []
    loss_tgt_valid = []
    loss_tgt_test = []
    loss_dc_src = []
    loss_dc_tgt = []
    str_diff = []
    best_valid_score = 0
    best_epoch = 0
    rw_rec = 0

    for epoch in range(args.epochs):
        model.train()
        loss = train_one_epoch(model, best_mlp_model, source_dataset, target_dataset, args, opt, scheduler, epoch)

        if (epoch + 1) % 1 == 0:
            epochs_train.append(epoch)

            source_embed, target_embed, inter_report, loss_dict = evaluate(source_dataset, target_dataset, model)


            loss_dc_src.append(loss_dict['dc_loss_src'])
            loss_dc_tgt.append(loss_dict['dc_loss_tgt'])
            loss_src_train.append(loss_dict['loss_src_train'])
            loss_src_valid.append(loss_dict['loss_src_valid'])
            loss_src_test.append(loss_dict['loss_src_test'])
            loss_tgt_valid.append(loss_dict['loss_tgt_valid'])
            loss_tgt_test.append(loss_dict['loss_tgt_test'])

            if args.valid_data == "src":
                valid_score = inter_report['acc_src_valid']
            else:
                valid_score = inter_report['acc_tgt_valid']

            if valid_score > best_valid_score:
                best_epoch = epoch
                best_valid_score = valid_score
                best_model = model
                torch.save(model.state_dict(), path + '/best_valid_model_' + str(seed) + '.pt')



    if args.best_model == False:
        best_model = model
    else:
        best_model = models.directed_GNN_adv(input_dim, output_dim, args)
        best_model.load_state_dict(torch.load(path + '/best_valid_model_' + str(seed) + '.pt'))
        best_model = best_model.to(device)

    source_embed, target_embed, final_report, loss_dict = evaluate(source_dataset, target_dataset, best_model)

    return final_report


def evaluate(source_dataset, target_dataset, model):

    loss_src_train, dc_loss_src_train, GNN_embed_src_train, final_embed_src_train, acc_src_train, auc_src_train = evaluate_dataset(
        source_dataset, model, "src_train")
    loss_src_valid, dc_loss_src_valid, GNN_embed_src_valid, final_embed_src_valid, acc_src_valid, auc_src_valid = evaluate_dataset(
        source_dataset, model, "src_valid")
    loss_src_test, dc_loss_src_test, GNN_embed_src_test, final_embed_src_test, acc_src_test, auc_src_test = evaluate_dataset(
        source_dataset, model, "src_test")
    loss_tgt_valid, dc_loss_tgt_valid, GNN_embed_tgt_valid, final_embed_tgt_valid, acc_tgt_valid, auc_tgt_valid = evaluate_dataset(
        target_dataset, model, "tgt_valid")
    loss_tgt_test, dc_loss_tgt_test, GNN_embed_tgt_test, final_embed_tgt_test, acc_tgt_test, auc_tgt_test = evaluate_dataset(
        target_dataset, model, "tgt_test")
    report = {'acc_tgt_valid': acc_tgt_valid, 'auc_tgt_valid': auc_tgt_valid,
              'acc_tgt_test': acc_tgt_test, 'auc_tgt_test': auc_tgt_test,
              'acc_src_valid': acc_src_valid, 'auc_src_valid': auc_src_valid,
              'acc_src_train': acc_src_train, 'auc_src_train': auc_src_train,
              'acc_src_test': acc_src_test, 'auc_src_test': auc_src_test,
              'default': acc_tgt_valid}
    loss_dict = {'loss_src_train': loss_src_train, 'loss_src_valid': loss_src_valid,
                 'loss_src_test': loss_src_test, 'loss_tgt_valid': loss_tgt_valid,
                 'loss_tgt_test': loss_tgt_test, 'dc_loss_src': dc_loss_src_train,
                 'dc_loss_tgt': dc_loss_tgt_valid}

    source_embed = [[GNN_embed_src_train, GNN_embed_src_valid, GNN_embed_src_test],
                    [final_embed_src_train, final_embed_src_valid, final_embed_src_test]]
    target_embed = [[GNN_embed_tgt_valid, GNN_embed_tgt_test], [final_embed_tgt_valid, final_embed_tgt_test]]

    return source_embed, target_embed, report, loss_dict


def evaluate_dataset(data, model, phase):
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        if phase == "src_train":
            mask = data.source_training_mask
        elif phase == "src_valid":
            mask = data.source_validation_mask
        elif phase == "src_test":
            mask = data.source_testing_mask
        elif phase == "tgt_valid":
            mask = data.target_validation_mask
        else:
            mask = data.target_testing_mask

        label = data.y
        [GNN_embed, final_embed], [pred, pred_domain] = model.forward(data, 1)

        GNN_embed = GNN_embed[mask]
        final_embed = final_embed[mask]
        pred = pred[mask]
        if "src" in phase:
            pred_domain = pred_domain[data.source_mask]
            domain_label = torch.zeros_like(pred_domain)
        else:
            pred_domain = pred_domain[data.target_mask]
            domain_label = torch.ones_like(pred_domain)
        label = label[mask]

        loss = utils.CE_loss(pred, label).item()
        dc_loss = utils.BCE_loss(pred_domain, domain_label).item()
        acc, auc = utils.get_scores(pred, label)

    return loss, dc_loss, GNN_embed, final_embed, acc, auc


def evaluate_dataset_multi(datasets, model, phase):
    total_loss = 0
    total_dc_loss = 0
    total_GNN_embed = None
    total_final_embed = None
    total_acc = 0
    total_auc = 0
    num_events = len(datasets)
    num_auc = 0
    model.eval()
    for data in datasets:
        data = data.to(device)
        with torch.no_grad():
            label = data.y
            [GNN_embed, final_embed], [pred, pred_domain] = model.forward(data, 1)

            mask = data.training_mask
            GNN_embed = GNN_embed[mask]
            final_embed = final_embed[mask]
            pred = pred[mask]
            pred_domain = pred_domain[mask]
            if "src" in phase:
                domain_label = torch.zeros_like(pred_domain)
            else:
                domain_label = torch.ones_like(pred_domain)
            label = label[mask]

            loss = utils.CE_loss(pred, label).item()
            dc_loss = utils.BCE_loss(pred_domain, domain_label).item()
            acc, auc = utils.get_scores(pred, label)
            total_loss += loss
            total_dc_loss += dc_loss
            total_acc += acc
            if auc is not None:
                total_auc += auc
                num_auc += 1

            if total_GNN_embed == None:
                total_GNN_embed = GNN_embed
                total_final_embed = final_embed
            else:
                total_GNN_embed = torch.cat((total_GNN_embed, GNN_embed), dim=0)
                total_final_embed = torch.cat((total_final_embed, final_embed), dim=0)

    return total_loss / num_events, total_dc_loss / num_events, total_GNN_embed, total_final_embed, total_acc / num_events, total_auc / num_auc


def get_avg_std_report(reports):
    all_keys = {k: [] for k in reports[0]}
    avg_report, avg_std_report = {}, {}
    for report in reports:
        for k in report:
            if report[k]:
                all_keys[k].append(report[k])
            else:
                all_keys[k].append(0)
    avg_report = {k: np.mean(v) for k, v in all_keys.items()}
    avg_std_report = {k: f'{np.mean(v):.5f} +/- {np.std(v):.5f}' for k, v in all_keys.items()}
    return avg_report, avg_std_report



def main():
    optimized_params = \
        {
            'epochs': 100,
            'opt_decay_step': 50,
            'opt_decay_rate': 0.8,
            "lr": 0.003,
            "alphamin": 1,
            "alphatimes": 1.5,
            "start_epoch": 0,
            "hidden_dim": 128,
            "conv_dim": 128,
            "cls_dim": 64,
            "rw_freq": 20,
            "h_threshold": 0.4,
            "K": 4
        }
    print(optimized_params)
    args = arg_parse()
    if args.gpu_ratio is not None:
        torch.cuda.set_per_process_memory_fraction(args.gpu_ratio)

    args.reweight = False
    if 'rw' in args.method:
        args.reweight = True
        args.rw_freq = optimized_params['rw_freq']
        args.start_epoch = optimized_params['start_epoch']

    args.lr = optimized_params['lr']
    args.opt_decay_rate = optimized_params['opt_decay_rate']
    args.opt_decay_step = optimized_params['opt_decay_step']
    args.hidden_dim = optimized_params['hidden_dim']
    args.conv_dim = optimized_params['conv_dim']
    args.cls_dim = optimized_params['cls_dim']
    #args.rw_lmda = optimized_params['rw_lmda']
    args.alphamin = optimized_params['alphamin']
    args.alphatimes = optimized_params['alphatimes']
    args.epochs = optimized_params['epochs']
    # args.num_layers = optimized_params['num_layers']
    # args.dc_layers = optimized_params['dc_layers']
    # args.class_layers = optimized_params['class_layers']
    # args.K = optimized_params['K']
    # args.dropout = optimized_params['dropout']

    print(args)
    num_seeds = args.seed
    # all_seeds = np.random.choice(range(5), num_seeds, replace=False)
    all_seeds = [1, 3, 5, 6, 8]
    reports = []
    for seed in all_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        if args.dataset == "Blog":
            source_dataset = datasets.load_data_from_mat("../dataset/Blog",args.src_name)
            datasets.adjust_graph_structure_fast_source(source_dataset, h_thresh=0.6)

            target_dataset = datasets.load_data_from_mat("../dataset/Blog",args.target_name)

        elif args.dataset == "Twitch":
            source_dataset = datasets.prepare_Twitch("../dataset/Twitch/",args.src_name)
            datasets.adjust_graph_structure_fast_source(source_dataset, h_thresh=0.6)
            target_dataset = datasets.prepare_Twitch("../dataset/Twitch/",args.tgt_name)

        elif args.dataset == "dblp_acm":
            source_dataset = datasets.prepare_dblp_acm("../dataset", args.src_name)
            datasets.adjust_graph_structure_fast_source(source_dataset, h_thresh=0.9)
            target_dataset = datasets.prepare_dblp_acm("../dataset",args.tgt_name)
        elif args.dataset == "Airport":
            source_dataset = datasets.prepare_airport("../dataset/Airport",args.src_name)
            target_dataset = datasets.prepare_airport("../dataset/Airport",args.tgt_name)

        report_dict = train(source_dataset, target_dataset, args, seed)
        reports.append(report_dict)
        print('-' * 80), print('-' * 80), print(f'[Seed {seed} done]: ', json.dumps(report_dict, indent=4)), print(
            '-' * 80), print('-' * 80)

    avg_report, avg_std_report = get_avg_std_report(reports)

    print(f'[All seeds done], Results: ', json.dumps(avg_std_report, indent=4))
    print('-' * 80), print('-' * 80), print('-' * 80), print('-' * 80)
    print("ggg")


if __name__ == '__main__':
    #torch.set_num_threads(5)
    main()
