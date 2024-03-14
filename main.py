import argparse
import torch
import torch.nn as nn
import numpy as np
import warnings
from sklearn.utils import class_weight
from utils import setup_seed, print_metrics_binary, get_median, device
from model import Model
from data_construction_MIMIC3 import data_construction


def train(train_loader,
          valid_loader,
          demographic_data,
          idx_list,
          input_dim,
          demo_dim,
          gru_dim,
          graph_head,
          phi,
          mask_p,
          walks_per_node,
          walk_length,
          gcn_dim,
          n_clusters,
          ffn_dim,
          del_rate,
          add_rate,
          tau,
          lambda_kl,
          lambda_cl,
          dropout,
          lambda2,
          lr,
          seed,
          epochs,
          file_name,
          device):

        model = Model(input_dim, demo_dim, gru_dim, graph_head, phi, mask_p, walks_per_node,
                      walk_length, gcn_dim, n_clusters, ffn_dim, del_rate, add_rate,
                      tau, lambda_kl, lambda_cl, dropout).to(device)
        opt_model = torch.optim.Adam(model.parameters(), lr=lr)

        setup_seed(seed)
        best_epoch = 0
        max_auroc = 0

        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(precision=2)
        np.set_printoptions(suppress=True)

        for each_epoch in range(epochs):
            batch_loss = []
            model.train()

            for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(train_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.to(device)

                batch_demo = []
                for i in range(len(batch_name)):
                    cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                    cur_idx = cur_id + '_' + cur_ep
                    idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                    cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)

                    batch_demo.append(cur_demo)

                batch_demo = torch.stack(batch_demo).to(device)
                x_mean = torch.stack(get_median(batch_x)).to(device)
                x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
                batch_x = torch.where(batch_x == -1, x_mean, batch_x)

                output, loss_hybrid = model(batch_x, batch_demo, sorted_length)

                batch_y = batch_y.long()
                y_out = batch_y.cpu().numpy()
                class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_out),
                                                                  y=y_out)
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
                criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

                loss_ce = criterion(output, batch_y)
                loss = lambda2 * loss_ce + (1-lambda2) * loss_hybrid
                batch_loss.append(loss.cpu().detach().numpy())

                opt_model.zero_grad()
                loss.backward()
                opt_model.step()


            y_true = []
            y_pred = []
            with torch.no_grad():
                model.eval()

                for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(valid_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.to(device)

                    batch_demo = []
                    for i in range(len(batch_name)):
                        cur_id, cur_ep, _ = batch_name[i].split('_', 2)
                        cur_idx = cur_id + '_' + cur_ep
                        idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

                        cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)

                        batch_demo.append(cur_demo)

                    batch_demo = torch.stack(batch_demo).to(device)
                    x_mean = torch.stack(get_median(batch_x)).to(device)
                    x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
                    batch_x = torch.where(batch_x == -1, x_mean, batch_x)

                    output, _ = model(batch_x, batch_demo, sorted_length)

                    batch_y = batch_y.long()
                    y_pred.append(output)
                    y_true.append(batch_y)

            y_pred = torch.cat(y_pred, 0)
            y_true = torch.cat(y_true, 0)
            test_y_pred = y_pred.cpu().detach().numpy()
            test_y_true = y_true.cpu().detach().numpy()
            ret = print_metrics_binary(test_y_true, test_y_pred)

            cur_auroc = ret['auroc']

            if cur_auroc > max_auroc:
                max_auroc = cur_auroc
                best_epoch = each_epoch
                state = {
                    'net': model.state_dict(),
                    'optimizer': opt_model.state_dict(),
                    'epoch': each_epoch
                }
                torch.save(state, file_name)

        return best_epoch


def test(test_loader,
         demographic_data,
         idx_list,
         input_dim,
         demo_dim,
         gru_dim,
         graph_head,
         phi,
         mask_p,
         walks_per_node,
         walk_length,
         gcn_dim,
         n_clusters,
         ffn_dim,
         del_rate,
         add_rate,
         tau,
         lambda_kl,
         lambda_cl,
         dropout,
         seed,
         epochs,
         file_name,
         device):

    setup_seed(seed)
    model = Model(input_dim, demo_dim, gru_dim, graph_head, phi, mask_p, walks_per_node,
                  walk_length, gcn_dim, n_clusters, ffn_dim, del_rate, add_rate,
                  tau, lambda_kl, lambda_cl, dropout).to(device)
    checkpoint = torch.load(file_name)
    model.load_state_dict(checkpoint['net'])
    model.eval()

    y_true = []
    y_pred = []
    for step, (batch_x, batch_y, sorted_length, batch_ts, batch_name) in enumerate(test_loader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.to(device)

        batch_demo = []
        for i in range(len(batch_name)):
            cur_id, cur_ep, _ = batch_name[i].split('_', 2)
            cur_idx = cur_id + '_' + cur_ep
            idx = idx_list.index(cur_idx) if cur_idx in idx_list else None

            cur_demo = torch.tensor(demographic_data[idx], dtype=torch.float32)

            batch_demo.append(cur_demo)

        batch_demo = torch.stack(batch_demo).to(device)
        x_mean = torch.stack(get_median(batch_x)).to(device)
        x_mean = x_mean.unsqueeze(0).expand(batch_x.shape[0], batch_x.shape[1], batch_x.shape[2])
        batch_x = torch.where(batch_x == -1, x_mean, batch_x)

        output, _ = model(batch_x, batch_demo, sorted_length)

        batch_y = batch_y.long()
        y_pred.append(output)
        y_true.append(batch_y)

    y_pred = torch.cat(y_pred, 0)
    y_true = torch.cat(y_true, 0)
    test_y_pred = y_pred.cpu().detach().numpy()
    test_y_true = y_true.cpu().detach().numpy()
    ret = print_metrics_binary(test_y_true, test_y_pred)

    cur_auroc = ret['auroc']
    cur_auprc = ret['auprc']
    cur_acc = ret['acc']
    cur_precision = ret['prec1']
    cur_recall = ret['rec1']
    cur_f1 = ret['f1_score']
    cur_minpse = ret['minpse']

    results = {'AUROC': cur_auroc, 'AUPRC': cur_auprc, 'Accuracy':cur_acc, 'Precision':cur_precision, 'Recall':cur_recall, 'F1':cur_f1, 'Minpse':cur_minpse}

    return results


if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dim", type=int)
    parser.add_argument("--demo_dim", type=int)
    parser.add_argument("--gru_dim", type=int)
    parser.add_argument("--graph_head", type=int)
    parser.add_argument("--phi", type=float)
    parser.add_argument("--mask_p", type=float)
    parser.add_argument("--walks_per_node", type=int)
    parser.add_argument("--walk_length", type=int)
    parser.add_argument("--gcn_dim", type=int)
    parser.add_argument("--n_clusters", type=int)
    parser.add_argument("--ffn_dim", type=int)
    parser.add_argument("--del_rate", type=float)
    parser.add_argument("--add_rate", type=float)
    parser.add_argument("--tau", type=float)
    parser.add_argument("--lambda_kl", type=float)
    parser.add_argument("--lambda_cl", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lambda2", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_demo_path", type=str)
    parser.add_argument("--file_name", type=str)
    args = parser.parse_args()

    input_dim = args.input_dim
    demo_dim = args.demo_dim
    gru_dim = args.gru_dim
    graph_head = args.graph_head
    phi = args.phi
    mask_p = args.mask_p
    walks_per_node = args.walks_per_node
    walk_length = args.walk_length
    gcn_dim = args.gcn_dim
    n_clusters = args.n_clusters
    ffn_dim = args.ffn_dim
    del_rate = args.del_rate
    add_rate = args.add_rate
    tau = args.tau
    lambda_kl = args.lambda_kl
    lambda_cl = args.lambda_cl
    dropout = args.dropout
    lambda2 = args.lambda2
    lr = args.lr
    seed = args.seed
    epochs = args.epochs
    data_path = args.data_path
    data_demo_path = args.data_demo_path
    file_name = args.file_name

    train_loader, valid_loader, test_loader, demographic_data, los_data, idx_list = data_construction(data_path, data_demo_path)
    best_epoch = train(train_loader, valid_loader, demographic_data, idx_list, input_dim, demo_dim, gru_dim, graph_head, phi, mask_p, walks_per_node, walk_length, \
                       gcn_dim, n_clusters, ffn_dim, del_rate, add_rate, tau, lambda_kl, lambda_cl, dropout, lambda2, lr, seed, epochs, file_name, device)
    results = test(test_loader, demographic_data, idx_list, input_dim, demo_dim, gru_dim, graph_head, phi, mask_p, walks_per_node, walk_length, \
                   gcn_dim, n_clusters, ffn_dim, del_rate, add_rate, tau, lambda_kl, lambda_cl, dropout, seed, epochs, file_name, device)
    print(results)

