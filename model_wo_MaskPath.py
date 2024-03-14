import math
import torch
import numpy as np
import copy
from typing import Optional
from torch import nn, Tensor
from torch.nn import functional as F
from torch.nn.modules import MultiheadAttention, Linear, Dropout, BatchNorm1d, TransformerEncoderLayer
from torch_cluster import random_walk
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv import GCNConv
from sklearn.cluster import KMeans
from utils import device


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def padding_mask(lengths, max_len=None):
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise ValueError("activation should be relu/gelu, not {}".format(activation))


# From https://github.com/pytorch/examples/blob/master/word_language_model/model.py
class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = src.permute(1, 2, 0)
        src = self.norm1(src)
        src = src.permute(2, 0, 1)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = src.permute(1, 2, 0)
        src = self.norm2(src)
        src = src.permute(2, 0, 1)
        return src


class TSTransformerEncoder(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, num_layers, dim_feedforward, dropout=0.1,
                 pos_encoding='fixed', activation='gelu', norm='BatchNorm', freeze=False):
        super(TSTransformerEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads

        self.project_inp = nn.Linear(feat_dim, d_model)
        self.pos_enc = get_pos_encoder(pos_encoding)(d_model, dropout=dropout * (1.0 - freeze), max_len=max_len)

        if norm == 'LayerNorm':
            encoder_layer = TransformerEncoderLayer(d_model, self.n_heads, dim_feedforward, dropout * (1.0 - freeze),
                                                    activation=activation)
        else:
            encoder_layer = TransformerBatchNormEncoderLayer(d_model, self.n_heads, dim_feedforward,
                                                             dropout * (1.0 - freeze), activation=activation)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_layer = nn.Linear(d_model, feat_dim)

        self.act = _get_activation_fn(activation)

        self.dropout1 = nn.Dropout(dropout)

        self.feat_dim = feat_dim

    def forward(self, X, padding_masks):
        inp = X.permute(1, 0, 2)
        inp = self.project_inp(inp) * math.sqrt(
            self.d_model)
        inp = self.pos_enc(inp)
        output = self.transformer_encoder(inp, src_key_padding_mask=~padding_masks)
        output = self.act(output)
        output = output.permute(1, 0, 2)

        return output


class TransformerEncoderFinal(nn.Module):
    def __init__(self, feat_dim, max_len, d_model, n_heads, n_layers, ff_dim, dropout):
        super(TransformerEncoderFinal, self).__init__()
        self.feat_dim = feat_dim
        self.max_len = max_len + 1
        self.tst = TSTransformerEncoder(feat_dim, self.max_len, d_model, n_heads, n_layers, ff_dim, dropout,
                                        activation='relu')

    def forward(self, x, sorted_length):
        batch_size = x.shape[0]
        x_star = torch.randn(batch_size, 1, self.feat_dim).to(device)
        x_c = torch.cat((x, x_star), 1)

        mask_pad = padding_mask(torch.from_numpy(np.array(sorted_length)).to(device), self.max_len)
        output = self.tst(x_c, mask_pad)

        return output[:, -1, :]


class GraphConstructor(nn.Module):
    def __init__(self, input_dim, h, phi, dropout=0):
        super(GraphConstructor, self).__init__()
        assert input_dim % h == 0

        self.d_k = input_dim // h
        self.h = h
        self.linears = clones(nn.Linear(input_dim, self.d_k * self.h), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.Wo = nn.Linear(h, 1)
        self.phi = nn.Parameter(torch.tensor(phi), requires_grad=True)

    def forward(self, query, key):
        query, key = [l(x).view(query.size(0), -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attns = self.attention(query.squeeze(2), key.squeeze(2))
        adj = torch.where(attns >= self.phi, torch.ones(attns.shape).to(device), torch.zeros(attns.shape).to(device))

        return adj

    def attention(self, query, key):
        d_k = query.size(-1)
        scores = torch.bmm(query.permute(1, 0, 2), key.permute(1, 2, 0)) \
                 / math.sqrt(d_k)
        scores = self.Wo(scores.permute(1, 2, 0)).squeeze(2)
        p_attn = F.softmax(scores, dim=1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)
        return p_attn


def DataAug(x, adj, prob_feature, prob_edge):
    batch_size = x.shape[0]
    input_dim = x.shape[1]

    tensor_p = torch.ones((batch_size, input_dim)) * (1 - prob_edge)
    mask_feature = torch.bernoulli(tensor_p).to(device)

    tensor_p = torch.ones((batch_size, batch_size)) * (1 - prob_edge)
    mask_edge = torch.bernoulli(tensor_p).to(device)

    return mask_feature * x, mask_edge * adj


class Clustering(nn.Module):
    def __init__(self, n_clusters, in_dim, alpha=1.0):
        super(Clustering, self).__init__()
        self.alpha = alpha
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=20)
        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, in_dim))

    def forward(self, z):
        _ = self.kmeans.fit_predict(z.data.cpu().numpy())
        self.cluster_layer.data.copy_(torch.tensor(self.kmeans.cluster_centers_, dtype=torch.float32))

        q = self.soft_assign(z)
        p = self.target_distribution(q).data

        cluster_label = torch.argmax(q, dim=1)
        cluster_loss = self.cluster_loss(p, q)

        return cluster_label, cluster_loss

    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer) ** 2, dim=2) / self.alpha)
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, dim=1)).t()

        return q

    def target_distribution(self, q):
        p = q ** 2 / q.sum(0)

        return (p.t() / p.sum(1)).t()

    def cluster_loss(self, p, q):
        def kld(target, pred):
            return torch.mean(torch.sum(target * torch.log(target / (pred + 1e-6)), dim=-1))

        kldloss = kld(p, q)

        return kldloss


def add_edge(z, edge_index, cluster_label, add_rate):
    edge_start, edge_end = edge_index[0].cpu().detach().numpy().tolist(), edge_index[1].cpu().detach().numpy().tolist()
    num_edges = edge_index.size(1)
    num_nodes = maybe_num_nodes(edge_index)
    cluster_label_np = cluster_label.cpu().detach().numpy()

    for k in np.unique(cluster_label_np):
        same_cluster = np.array(cluster_label_np == k)
        same_cluster_idx = np.nonzero(same_cluster)[0]
        nodes = same_cluster_idx

        num_nodes_k = len(nodes)
        add_num = int(add_rate * num_edges * (num_nodes_k / num_nodes))
        cluster_embeds = z[nodes]

        sim = torch.mm(cluster_embeds, cluster_embeds.T).cpu().detach()
        sim_top = sim.view(-1).topk(add_num).indices.numpy()

        row_idx = sim_top // num_nodes_k
        col_idx = sim_top % num_nodes_k

        new_edge_start = nodes[row_idx]
        new_edge_end = nodes[col_idx]

        edge_start.extend(new_edge_start.tolist())
        edge_end.extend(new_edge_end.tolist())

    new_edge_index = torch.vstack([torch.tensor(edge_start).to(device), torch.tensor(edge_end).to(device)])
    new_edge_index = torch.unique(new_edge_index, dim=1)

    return new_edge_index


def del_edge(z, edge_index, cluster_label, del_rate):
    edge_start, edge_end = edge_index[0].cpu().detach().numpy(), edge_index[1].cpu().detach().numpy()
    cluster_label_np = cluster_label.cpu().detach().numpy()

    diff_class_bool = np.array(cluster_label_np[edge_start] != cluster_label_np[edge_end], dtype=np.bool)
    diff_class_idx = np.nonzero(diff_class_bool)[0]
    diff_class_edge_len = len(diff_class_idx)

    nodes = np.concatenate((edge_start[diff_class_idx], edge_end[diff_class_idx]))
    nodes = np.sort(np.unique(nodes))
    node_embeds = z[nodes]
    del_num = int(del_rate * diff_class_edge_len)

    sim = torch.mm(node_embeds, node_embeds.T).cpu().detach()
    edge_ = np.vstack((edge_start[diff_class_idx], edge_end[diff_class_idx])).T

    edge_0 = edge_[:, 0]
    edge_1 = edge_[:, 1]
    edge_0_idx = [np.where(nodes == edge_0[i])[0][0] for i in range(len(edge_0))]
    edge_1_idx = [np.where(nodes == edge_1[i])[0][0] for i in range(len(edge_1))]

    del_edge = (-sim[edge_0_idx, edge_1_idx]).topk(del_num).indices.numpy()
    del_edge_idx = edge_[del_edge].T
    edge_index_tuple = tuple(map(tuple, edge_index.T.cpu().detach().numpy()))
    del_edge_idx_tuple = tuple(map(tuple, del_edge_idx.T))
    remain_edge_index = set(edge_index_tuple) - set(del_edge_idx_tuple)
    remain_edge_index = torch.tensor(np.array(list(remain_edge_index))).T

    return remain_edge_index


def sim(z1, z2, hidden_norm=True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.T)


def clloss(z, z_aug, adj, cluster_label, tau, hidden_norm=True):
    adj = adj - torch.diag_embed(adj.diag())

    f = lambda x: torch.exp(x / tau)
    intra_view_sim = f(sim(z, z, hidden_norm))
    inter_view_sim = f(sim(z, z_aug, hidden_norm))

    cluster_label = cluster_label.unsqueeze(1)
    one_hot_labels = torch.zeros(len(cluster_label), len(cluster_label)).to(device).scatter_(1, cluster_label, 1)
    cluster_adj = torch.mm(one_hot_labels, one_hot_labels.T)
    cluster_adj = cluster_adj - torch.diag_embed(cluster_adj.diag())
    cluster_adj_new = torch.where(adj == 1, torch.zeros(cluster_adj.shape).to(device), cluster_adj)

    positive = inter_view_sim.diag() + intra_view_sim.mul(adj).sum(1) + intra_view_sim.mul(cluster_adj_new).sum(1)

    loss = positive / (intra_view_sim.sum(1) + inter_view_sim.sum(1) - intra_view_sim.diag())

    adj_count = torch.sum(adj, 1) + torch.sum(cluster_adj_new, 1) + 1
    loss = torch.log(loss) / adj_count

    return -torch.sum(loss, 0)


class Model(nn.Module):
    def __init__(self, input_dim, demo_dim, gru_dim, graph_head, phi, mask_p, walks_per_node, walk_length,
                 gcn_dim, n_clusters, ffn_dim, del_rate, add_rate, tau, lambda_kl, lambda_cl, dropout):
        super(Model, self).__init__()
        self.del_rate = del_rate
        self.add_rate = add_rate
        self.tau = tau
        self.lambda_kl = lambda_kl
        self.lambda_cl = lambda_cl

        self.gru = nn.GRU(input_size=input_dim, hidden_size=gru_dim,
                          batch_first=True, bidirectional=False, num_layers=1)
        self.h_0 = nn.Parameter(torch.zeros(1, 1, gru_dim), requires_grad=True).to(device)

        self.graphconstructor = GraphConstructor(gru_dim + demo_dim, graph_head, phi, dropout=0)
        self.clustering = Clustering(n_clusters, gru_dim + demo_dim)
        # self.maskpath = MaskPath(mask_p, walks_per_node, walk_length, start='node', num_nodes=None)
        self.gcn = GCNConv(gru_dim + demo_dim, gcn_dim)
        self.ffn = nn.Linear(gcn_dim, ffn_dim)

        self.pre = nn.Linear(gcn_dim, 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_demo, sorted_length):
        batch_size = x.shape[0]
        h_0_contig = self.h_0.expand(1, batch_size, self.gru.hidden_size).contiguous()
        out, _ = self.gru(x, h_0_contig)
        out_list = []
        for i in range(out.shape[0]):
            idx = sorted_length[i] - 1
            out_list.append(out[i, idx, :])

        x_gru = torch.stack(out_list)
        x_star = torch.cat((x_gru, x_demo), 1)

        adj_init = self.graphconstructor(x_star, x_star)
        adj_init = adj_init - torch.diag_embed(adj_init.diag())
        edge_index = torch.nonzero(adj_init == 1).T

        cluster_label, kl_loss = self.clustering(x_star)

        new_edge_index = del_edge(x_star, edge_index, cluster_label, self.del_rate)
        new_edge_index = add_edge(x_star, new_edge_index, cluster_label, self.add_rate)
        new_edge_index_T = new_edge_index.T
        new_adj = torch.zeros(x.shape[0], x.shape[0]).to(device)
        new_adj[new_edge_index_T[:, 0], new_edge_index_T[:, 1]] = 1

        _, adj_aug = DataAug(x_star, new_adj, prob_feature=0, prob_edge=self.mask_p)
        edge_index_aug = torch.nonzero(adj_aug == 1).T

        z = self.gcn(x_star, new_edge_index)
        z_aug = self.gcn(x_star, edge_index_aug)

        z_ffn = self.ffn(z)
        z_ffn_aug = self.ffn(z_aug)
        cl_loss = clloss(z_ffn, z_ffn_aug, new_adj, cluster_label, self.tau)
        loss = self.lambda_kl * kl_loss + self.lambda_cl * cl_loss
        output = self.pre(z)

        return output, loss

