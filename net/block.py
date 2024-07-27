import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, MessagePassing, GraphNorm, CGConv, GENConv, GATv2Conv, GATConv, PDNConv, GeneralConv

from typing import List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    InstanceNorm1d,
    LayerNorm,
    ReLU,
    Sequential,
)

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.nn.norm import MessageNorm
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)
from torch_geometric.utils import is_torch_sparse_tensor, to_edge_index


class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super().__init__(*m)


class gcnblock(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = 'softmax',
        t: float = 1.0,
        learn_t: bool = False,
        p: float = 1.0,
        learn_p: bool = False,
        msg_norm: bool = False,
        learn_msg_scale: bool = False,
        norm: str = 'batch',
        num_layers: int = 2,
        expansion: int = 2,
        eps: float = 1e-7,
        bias: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):

        # Backward compatibility:
        semi_grad = True if aggr == 'softmax_sg' else False
        aggr = 'softmax' if aggr == 'softmax_sg' else aggr
        aggr = 'powermean' if aggr == 'power' else aggr

        # Override args of aggregator if `aggr_kwargs` is specified
        if 'aggr_kwargs' not in kwargs:
            if aggr == 'softmax':
                kwargs['aggr_kwargs'] = dict(t=t, learn=learn_t,
                                             semi_grad=semi_grad)
            elif aggr == 'powermean':
                kwargs['aggr_kwargs'] = dict(p=p, learn=learn_p)

        super().__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.eps = eps

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if in_channels[0] != out_channels:
            self.lin_src = Linear(in_channels[0], out_channels, bias=bias)

        if edge_dim is not None and edge_dim != out_channels:
            self.lin_edge = Linear(edge_dim, out_channels, bias=bias)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(out_channels)
        else:
            aggr_out_channels = out_channels

        if aggr_out_channels != out_channels:
            self.lin_aggr_out = Linear(aggr_out_channels, out_channels,
                                       bias=bias)

        if in_channels[1] != out_channels:
            self.lin_dst = Linear(in_channels[1], out_channels, bias=bias)

        channels = [out_channels]
        for i in range(num_layers - 1):
            channels.append(out_channels * expansion)
        channels.append(out_channels)
        self.mlp = MLP(channels, norm=norm, bias=bias)

        if msg_norm:
            self.msg_norm = MessageNorm(learn_msg_scale)

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp)
        if hasattr(self, 'msg_norm'):
            self.msg_norm.reset_parameters()
        if hasattr(self, 'lin_src'):
            self.lin_src.reset_parameters()
        if hasattr(self, 'lin_edge'):
            self.lin_edge.reset_parameters()
        if hasattr(self, 'lin_aggr_out'):
            self.lin_aggr_out.reset_parameters()
        if hasattr(self, 'lin_dst'):
            self.lin_dst.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                edge_attr: OptTensor = None, size: Size = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        if hasattr(self, 'lin_src'):
            x = (self.lin_src(x[0]), x[1])

        if isinstance(edge_index, SparseTensor):
            edge_attr = edge_index.storage.value()
        elif is_torch_sparse_tensor(edge_index):
            _, value = to_edge_index(edge_index)
            if value.dim() > 1 or not value.all():
                edge_attr = value

        if edge_attr is not None and hasattr(self, 'lin_edge'):
            edge_attr = self.lin_edge(edge_attr)

        # Node and edge feature dimensionalites need to match.
        if edge_attr is not None:
            assert x[0].size(-1) == edge_attr.size(-1)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        if hasattr(self, 'lin_aggr_out'):
            out = self.lin_aggr_out(out)

        if hasattr(self, 'msg_norm'):
            h = x[1] if x[1] is not None else x[0]
            assert h is not None
            out = self.msg_norm(h, out)

        x_dst = x[1]
        if x_dst is not None:
            if hasattr(self, 'lin_dst'):
                x_dst = self.lin_dst(x_dst)
            out = out + x_dst

        return self.mlp(out)

    def message(self, x_j: Tensor, edge_attr: OptTensor) -> Tensor:
        msg = x_j if edge_attr is None else x_j + edge_attr
        return msg.relu() + self.eps

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')



class Encoder(nn.Module):
    def __init__(self, in_size, hidden_size,  out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.relu = nn.ReLU()
        self.bn1 = GraphNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_size)

    def forward(self, data, batch):
        condition = self.relu(self.bn1(self.fc1(data), batch))
        return self.fc2(condition)


class EdgesPointsEncoder(MessagePassing):
    def __init__(self, p_channel, e_channel):
        super().__init__(aggr='mean')
        self.fc1 = nn.Linear(p_channel, e_channel)
        self.fc2 = nn.Linear(p_channel, e_channel)
        self.fc3 = nn.Linear(e_channel, e_channel)
        self.bn1 = nn.BatchNorm1d(e_channel)

        self.relu = nn.LeakyReLU()

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)

        return edge_attr

    def edge_update(self, x_j, x_i, edge_attr):
        edge_attr = self.fc1(x_i) + self.fc2(x_j) + self.fc3(edge_attr)
        edge_attr = self.relu(self.bn1(edge_attr))
        return edge_attr


class TransBlock(nn.Module):
    def __init__(self, p_e, p_d, number_layers):
        super(TransBlock, self).__init__()
        self.TransGcn1 = TransformerConv(in_channels=p_e, out_channels=p_e,
                                        heads=8, dropout=0.1,
                                        concat=False, edge_dim=p_d)
        self.gn1 = GraphNorm(p_e)

        self.TransGcn2 = TransformerConv(in_channels=p_e, out_channels=p_e,
                                        heads=8, dropout=0.1,
                                        concat=False, edge_dim=p_d)
        self.gn2 = GraphNorm(p_e)

        self.relu = nn.LeakyReLU()

    def forward(self, x, index, attr, batch):
        x = self.relu(self.gn1(self.TransGcn1(x, index, attr), batch))
        encoder_out = self.relu(self.gn2(self.TransGcn2(x, index, attr), batch))  # (t_h_s, t_o_s)

        return encoder_out


class GATv1(nn.Module):
    def __init__(self, p_e, p_d, number_layers):
        super(GATv1, self).__init__()
        self.gat1 = GATConv(in_channels=p_e, out_channels=p_e,
                                        heads=8, dropout=0.1,
                                        concat=False, edge_dim=p_d)
        self.gn1 = GraphNorm(p_e)

        self.gat2 = GATConv(in_channels=p_e, out_channels=p_e,
                                        heads=8, dropout=0.1,
                                        concat=False, edge_dim=p_d)
        self.gn2 = GraphNorm(p_e)
        self.relu = nn.LeakyReLU()

    def forward(self, x, index, attr, batch):
        x = self.relu(self.gn1(self.gat1(x, index, attr), batch))
        encoder_out = self.relu(self.gn2(self.gat2(x, index, attr), batch))  # (t_h_s, t_o_s)

        return encoder_out


class cgc(nn.Module):
    def __init__(self, p_e, p_d, number_layers):
        super(cgc, self).__init__()
        self.cgc1 = CGConv(channels=p_e, dim=p_d)
        self.gn1 = GraphNorm(p_e)

        self.cgc2 = CGConv(channels=p_e, dim=p_d)
        self.gn2 = GraphNorm(p_e)
        self.relu = nn.LeakyReLU()

    def forward(self, x, index, attr, batch):
        x = self.relu(self.gn1(self.cgc1(x, index, attr), batch))
        x = self.relu(self.gn2(self.cgc2(x, index, attr), batch))  # (t_h_s, t_o_s)

        return x

class PdnBLOCK(nn.Module):
    def __init__(self, p_e, p_d, number_layers):
        super(PdnBLOCK, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(2):
            conv = GeneralConv(in_channels=p_e, out_channels=p_e, in_edge_channels=p_d)
            self.convs.append(conv)
            self.batch_norms.append(GraphNorm(p_e))

        self.relu = nn.LeakyReLU()

    def forward(self, x, index, attr, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.relu(batch_norm(conv(x, index, attr), batch))

        return x


class Generalblock(nn.Module):
    def __init__(self, p_e, p_d, number_layers):
        super(Generalblock, self).__init__()

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(2):
            conv = PDNConv(in_channels=p_e, out_channels=p_e, edge_dim=p_d, hidden_channels=p_d)
            self.convs.append(conv)
            self.batch_norms.append(GraphNorm(p_e))

        self.relu = nn.LeakyReLU()

    def forward(self, x, index, attr, batch):
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = self.relu(batch_norm(conv(x, index, attr), batch))

        return x


class gatv2(nn.Module):
    def __init__(self, p_e, p_d, number_layers):
        super(gatv2, self).__init__()
        self.gat1 = GATv2Conv(in_channels=p_e, out_channels=p_e,
                                        heads=8, dropout=0.1,
                                        concat=False, edge_dim=p_d)
        self.gn1 = GraphNorm(p_e)

        self.gat2 = GATv2Conv(in_channels=p_e, out_channels=p_e,
                                        heads=8, dropout=0.1,
                                        concat=False, edge_dim=p_d)
        self.gn2 = GraphNorm(p_e)

        self.relu = nn.LeakyReLU()

    def forward(self, x, index, attr, batch):
        x = self.relu(self.gn1(self.gat1(x, index, attr), batch))
        encoder_out = self.relu(self.gn2(self.gat2(x, index, attr), batch))  # (t_h_s, t_o_s)

        return encoder_out


class DeepMeshGcnLayer(nn.Module):
    def __init__(self, p_d, e_d):
        super().__init__()
        self.passing = nn.Linear(p_d, e_d)
        self.conv = gcnblock(p_d, p_d, aggr='softmax', t=1.0, edge_dim=e_d, learn_t=True, num_layers=2, norm='layer')
        self.gn = GraphNorm(p_d)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(e_d)

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        attr = edge_attr
        h = self.act(self.gn(h, batch))
        attr = self.act(self.bn(attr))
        attr = self.passing(attr)

        h = self.conv(h, edge_index, attr)

        return h + x, attr + edge_attr


class DeeperMeshGcn(nn.Module):
    def __init__(self, p_d, e_d, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(0, num_layers):
            layer = DeepMeshGcnLayer(p_d, e_d)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_attr, batch):
        edge_attr = self.layers[0].passing(edge_attr)
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x, edge_attr = layer(x, edge_index, edge_attr, batch)

        x = self.layers[0].act(self.layers[0].gn(x, batch))
        return x


class DeepMeshGcnLayerNoEdge(nn.Module):
    def __init__(self, p_d, e_d):
        super().__init__()
        self.conv = gcnblock(p_d, p_d, aggr='softmax', t=1.0, edge_dim=e_d, learn_t=True, num_layers=2, norm='layer')
        self.gn = GraphNorm(p_d)
        self.act = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(e_d)

    def forward(self, x, edge_index, edge_attr, batch):
        h = x
        h = self.act(self.gn(h, batch))
        h = self.conv(h, edge_index, edge_attr)

        return h + x


class DeeperMeshGcnNoEdge(nn.Module):
    def __init__(self, p_d, e_d, num_layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(0, num_layers):
            layer = DeepMeshGcnLayerNoEdge(p_d, e_d)
            self.layers.append(layer)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr, batch)

        x = self.layers[0].act(self.layers[0].gn(x, batch))
        return x


