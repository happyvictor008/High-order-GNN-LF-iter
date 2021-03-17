from __future__ import absolute_import

import torch.nn as nn
from models.gnn_lf_iter_conv import GNN_LF
from models.gnn_lf_iter_conv_0 import GNN_LF_0
import torch 




class MLP(nn.Module):

    def __init__(self,in_features,out_features):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.model = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, out_features),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.model(x)

        return x



class _GraphConv0(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv0, self).__init__()

        self.gconv = GNN_LF_0(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim*4)
        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):

        x = self.gconv(x).transpose(1, 2)##################
        x = self.bn(x).transpose(1, 2)
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x






class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = GNN_LF(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim*4)
        self.relu = nn.ReLU()
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, x0):

        x = self.gconv(x, x0).transpose(1, 2)##################
        x = self.bn(x).transpose(1, 2)
        
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x



class _ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()



        self.mlp=MLP(input_dim, hid_dim*4)
        self.gconv1 = _GraphConv0(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv3 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv4 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv5 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv6 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv7 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv8 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv9 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)
        self.gconv10 = _GraphConv(adj, hid_dim*4, hid_dim, p_dropout)#13
        self.gconv11 = GNN_LF(hid_dim*4,output_dim,adj)



    def forward(self, x):
        initial = self.mlp(x)

        out = self.gconv1(initial)
        out = self.gconv2(out,initial)
        out = self.gconv3(out,initial)
        out = self.gconv4(out,initial)



        out = self.gconv5(out,initial)
        out = self.gconv6(out,initial)
        out = self.gconv7(out,initial)
        out = self.gconv8(out,initial)
        out = self.gconv9(out,initial)
        out = self.gconv10(out,initial)
        out = self.gconv11(out,initial)


        return  out


class GNN_LF_Iter(nn.Module):
    def __init__(self, adj, hid_dim, coords_dim=(2, 3), num_layers=4,nodes_group=None, p_dropout=None):
        super(GNN_LF_Iter, self).__init__()

        #self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim, p_dropout=p_dropout,l=1)
        self.model = _ResGraphConv(adj, coords_dim[0],  coords_dim[1],hid_dim, p_dropout=p_dropout)
        #self.gconv_layers = _ResGraphConv(adj, hid_dim*4, hid_dim, hid_dim, p_dropout=p_dropout,l=1)
        #self.gconv_output = HGraphConvII(hid_dim*4, coords_dim[1], adj,l=1)#layer_index+2
    def forward(self, x): 

        out=self.model(x)
        
        return out



