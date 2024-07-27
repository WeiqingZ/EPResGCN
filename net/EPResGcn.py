import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GraphNorm
from net.block import Encoder, EPResGCN


class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gcn1 = GCNConv(in_channels=args.p_e, out_channels=args.v_h1_s)
        self.gn1 = GraphNorm(args.v_h1_s)
        self.relu = nn.ReLU()
        self.gcn2 = GCNConv(in_channels=args.v_h1_s, out_channels=args.v_h2_s)
        self.gn2 = GraphNorm(args.v_h2_s)
        self.fc1 = nn.Linear(args.v_h2_s, 2)

    def forward(self, x, index, batch):
        x = self.relu(self.gn1(self.gcn1(x, index), batch))
        x = self.relu(self.gn2(self.gcn2(x, index), batch))
        out = self.fc1(x)
        return out


class DnnCfD(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.point_encoder = Encoder(args.p_i, args.p_h, args.p_e)
        self.edges_encoder = nn.Linear(args.o_edge_dim, args.edge_dim)

        self.transformer_encoder = EPResGCN(args.p_e, args.edge_dim, args.num_layers)

        self.decoder = Decoder(args)

    def forward(self, point, condition, index, attr, batch, time):

        attr = self.edges_encoder(attr)
        point = torch.cat((point, condition, time), dim=1)
        point = self.point_encoder(point, batch)

        trans_encoder = self.transformer_encoder(point, index, attr, batch)

        out = self.decoder(trans_encoder, index, batch)
        return out
