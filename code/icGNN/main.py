import argparse
import os.path as osp
import time
from tqdm import tqdm

import torch_geometric.transforms as T
from torch.autograd import Variable
from torch_geometric.datasets import Planetoid

import torch, math
import numpy as np
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, dense_to_sparse
from torch_geometric.utils import to_dense_adj

from GCNLFR import GCNLFR
from attack.graph_attack import PGDAttack, Metattack, RandomAttack
from attack.target_attack import Nettack
from util import sparse_mx_to_torch_sparse_tensor, select_nodes

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Pubmed',help='Cora Pubmed citeseer')
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--use_gdc', action='store_true', help='Use GDC')
parser.add_argument('--wandb', action='store_true', help='Track experiment')
parser.add_argument('--n_perturbations', type=int, default=4)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
# dataset = Planetoid("./data/Planetoid", args.dataset, transform=T.NormalizeFeatures())
dataset = Planetoid(root='./data/', name=args.dataset)
data = dataset[0]

if args.use_gdc:
    transform = T.GDC(
        self_loop_weight=1,
        normalization_in='sym',
        normalization_out='col',
        diffusion_kwargs=dict(method='ppr', alpha=0.05),
        sparsification_kwargs=dict(method='topk', k=128, dim=0),
        exact=True,
    )
    data = transform(data)


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class VAE(nn.Module):
    def __init__(self, input_dim, inter_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, latent_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inter_dim),
            nn.ReLU(),
            nn.Linear(inter_dim, input_dim),
            nn.Sigmoid(),
        )

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x):
        org_size = x.size()
        batch = org_size[0]
        x = x.view(batch, -1)
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        z = self.reparameterise(mu, logvar)
        recon_x = self.decoder(z).view(size=org_size)
        return recon_x, mu, logvar


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False,
                 bias=True, normalize=True, **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize

        self.encoder = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * out_channels),
        )

        self.decoder = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Linear(32, out_channels),
            nn.Sigmoid(),
        )

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        x = torch.matmul(x, self.weight)
        # print(x.size())
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            if self.normalize:
                edge_index, norm = self.norm(edge_index, x.size(
                    self.node_dim), edge_weight, self.improved, x.dtype)
            else:
                norm = edge_weight
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result
        out = self.propagate(edge_index, x=x, norm=norm)
        org_size = out.size()
        # print(org_size)
        h = self.encoder(out)
        mu, logvar = h.chunk(2, dim=1)
        recon_out = self.reparameterise(mu, logvar)
        # print(recon_out.size())
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # recon_loss = F.binary_cross_entropy(recon_out, out)
        return out, mu, logvar, recon_out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j if norm is not None else x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * hidden_channels, cached=False,
                             normalize=not args.use_gdc)
        self.conv2 = GCNConv(2 * hidden_channels, out_channels, cached=False,
                             normalize=not args.use_gdc)
        #self.liner = nn.Linear(3 * out_channels, out_channels)
        # self.conv3 = GCNConv(hidden_channels, out_channels, cached=True,
        #                      normalize=not args.use_gdc)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=0.5, training=self.training)
        x, mu, logvar, recon_x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x.relu(), p=0.5, training=self.training)
        recon_x = F.dropout(recon_x.relu(), p=0.5, training=self.training)
        x1, mu, logvar, recon_x1 = self.conv2(recon_x, edge_index, edge_weight)
        x2, mu, logvar, recon_x2 = self.conv2(x, edge_index, edge_weight)
        out = 0.95 * x2 + 0.05 * x1
        # out=self.liner(torch.cat((x,x2,x1),1))
        # print(out.size())
        return x2, mu, logvar, recon_x2


### node 2708  feature 1433  32
model = GCN(dataset.num_features, args.hidden_channels, dataset.num_classes)
model, data = model.to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# optimizer = torch.optim.Adam([
#     dict(params=model.conv1.parameters(), weight_decay=5e-4),
#     dict(params=model.conv2.parameters(), weight_decay=5e-4),
#     # dict(params=model.conv3.parameters(), weight_decay=5e-4)
# ], lr=args.lr)  # Only perform weight-decay on first convolution.

recon_loss_fn = nn.MSELoss()


def train():
    model.train()
    optimizer.zero_grad()
    data.x.requires_grad = True
    # data.edge_index.requires_grad=True
    # data.edge_attr.requires_grad=True
    out, mu, logvar, recon_x = model(data.x, data.edge_index, data.edge_attr)
    # print(recon_x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(out,recon_x)
    recon_loss = recon_loss_fn(out, recon_x)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # print("loss:", loss, "recon_loss", recon_loss, "kl_loss", recon_loss)
    loss = loss + 0.01 * recon_loss  # + 0.01 * kl_loss
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred, mu, logvar, recon_x = model(data.x, data.edge_index, data.edge_attr)
    pred = pred.argmax(dim=-1)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((pred[mask] == data.y[mask]).sum()) / int(mask.sum()))
    return accs



def multi_test_poison():
    # test on 40 nodes on poisoining attack
    from deeprobust.graph.defense import GCN
    adj=to_dense_adj(data.edge_index)[0]
    surrogate = GCN(nfeat=dataset.num_features,
                    nhid=args.hidden_channels,
                    nclass=dataset.num_classes,
                    dropout=0.5, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(data.x, adj, data.y, data.train_mask, data.val_mask, patience=30)
    cnt = 0
    degrees = torch.flatten(adj.sum(0))
    node_list = select_nodes(surrogate,data.test_mask,data.y)
    num = len(node_list)
    print('=== [Poisoning] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):
        # n_perturbations = int(degrees[target_node])
        attack_model = Nettack(surrogate, nnodes=data.x.shape[0], attack_features=True, device=device)
        attack_model = attack_model.to(device)
        attack_model.attack(data.x, adj, data.y, target_node, args.n_perturbations, verbose=False)
        #modified_adj = attack_model.modified_adj
        modified_adj = sparse_mx_to_torch_sparse_tensor(attack_model.modified_adj).to_dense().to(device)
        modified_features = sparse_mx_to_torch_sparse_tensor(attack_model.modified_features).to_dense().to(device)
        #print("modified_features:", modified_features)
        edge_index, edge_weight = dense_to_sparse(modified_adj)
        acc = single_test(edge_index, edge_weight, modified_features, target_node, model=model)
        if acc >= 0.5:
            cnt += 1
    print('poison classification acc rate : %s' % (cnt/num))
def multi_test_evasion():
    from deeprobust.graph.defense import GCN
    # test on 40 nodes on evasion attack
    adj = to_dense_adj(data.edge_index)[0]
    cnt = 0
    surrogate = GCN(nfeat=dataset.num_features,
                    nhid=args.hidden_channels,
                    nclass=dataset.num_classes,
                    dropout=0.5, device=device)
    surrogate = surrogate.to(device)
    surrogate.fit(data.x, adj, data.y, data.train_mask, data.val_mask, patience=30)
    #surrogate.fit(data.x, adj, data.y, data.train_mask, data.val_mask, patience=30)
    degrees = torch.flatten(adj.sum(0))
    node_list = select_nodes(surrogate,data.test_mask,data.y)
    num = len(node_list)

    print('=== [Evasion] Attacking %s nodes respectively ===' % num)
    for target_node in tqdm(node_list):
        #n_perturbations = int(degrees[target_node])
        #print("n_perturbations:",n_perturbations)
        attack_model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device=device)
        attack_model = attack_model.to(device)
        attack_model.attack(data.x, adj, data.y, target_node, args.n_perturbations, verbose=False)
        #modified_adj = attack_model.modified_adj
        modified_adj = sparse_mx_to_torch_sparse_tensor(attack_model.modified_adj).to_dense().to(device)
        #modified_features =attack_model.modified_features#torch.tensor(attack_model.modified_features).to(device)
        modified_features = sparse_mx_to_torch_sparse_tensor(attack_model.modified_features).to_dense().to(device)
        #print("modified_features:",modified_features)
        edge_index, edge_weight = dense_to_sparse(modified_adj)
        acc = single_test(edge_index, edge_weight, modified_features, target_node, model=model)
        #acc = single_test(edge_index,edge_weight, modified_features, target_node, model=model)
        if acc >= 0.5:
            cnt += 1
    print('Evasion classification acc: %s' % (cnt/num))


def single_test(edge_index, edge_weight, modified_features, target_node, model=model):
    pred, mu, logvar, recon_x = model(modified_features, edge_index,edge_weight)
    probs = torch.exp(pred[[target_node]])
    # acc_test = accuracy(output[[target_node]], labels[target_node])
    acc_test = (pred.argmax(1)[target_node] == data.y[target_node])
    return acc_test.item()

best_val_acc = final_test_acc = 0
times = []
for epoch in range(1, args.epochs + 1):
    start = time.time()
    loss = train()
    train_acc, val_acc, tmp_test_acc = test()
    print("loss: {:.4f}",loss)
    # train_acc, val_acc, attack_test_acc = test_attack()  # 0,0,0 #
    if final_test_acc < tmp_test_acc:
        final_test_acc = tmp_test_acc
    # print('Epoch {:03d} loss: {:.4f} test_acc: {:.4f} pgd_acc: {:.4f}'.format(
    #     epoch, loss, final_test_acc, 0))
    times.append(time.time() - start)
    torch.cuda.empty_cache()
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")
#multi_test_evasion()
#multi_test_poison()
