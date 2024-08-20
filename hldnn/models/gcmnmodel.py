import torch
import torch_geometric
import torch.nn.functional as F
import random
import models.mlps as mlps
import torch.nn as nn
from config import config
import torch_scatter
from torch.nn.functional import normalize
from models.encoders import AtomEncoder, BondEncoder
from torch_geometric.nn import aggr

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = config.NODE_ENCODER_NUM_TYPES
        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x.to(dtype=torch.int64)[:, 0])
        return batch


class TypeDictEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = config.EDGE_ENCODER_NUM_TYPES
        if num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, edge_features):
        edge_features = edge_features.to(dtype=torch.int64)
        edge_features = torch.flatten(self.encoder(edge_features), start_dim=1)
        return edge_features


class EncodeModule(nn.Module):
    def __init__(self):
        super(EncodeModule, self).__init__()
        self.node_encoder = nn.Identity()

        if config.ATOM_ENCODER:
            self.node_encoder = AtomEncoder(config.HIDDEN_SIZE)
            self.encoder = mlps.WideMLP(config.HIDDEN_SIZE+1, config.HIDDEN_SIZE, config.HIDDEN_SIZE, 1)
        else:
            self.encoder = mlps.WideMLP(config.IN_SIZE+1, config.HIDDEN_SIZE, config.HIDDEN_SIZE, 1)

        if config.BOND_ENCODER:
            self.edge_encoder = BondEncoder(config.HIDDEN_SIZE)
        else:
            self.edge_encoder = mlps.WideMLP(config.EDGE_SIZE, config.HIDDEN_SIZE, config.HIDDEN_SIZE, 1)

        if config.EDGE_ENCODER_NUM_TYPES is not None:
            self.edge_encoder = TypeDictEdgeEncoder(config.HIDDEN_SIZE)
        if config.NODE_ENCODER_NUM_TYPES is not None:
            self.encoder = mlps.WideMLP(config.HIDDEN_SIZE+1, config.HIDDEN_SIZE, config.HIDDEN_SIZE, 1)
            self.node_encoder = TypeDictNodeEncoder(config.HIDDEN_SIZE)

    def forward(self, data):
        data = self.node_encoder(data)
        if config.BOND_ENCODER:
            if data.edge_features.shape[1] == 0:
                #print("NO.")
                data.edge_features = torch.zeros((data.edge_features.shape[0], config.HIDDEN_SIZE)).to(config.device)
            else:
                #print(data.edge_features.shape)
                #print(data.edge_features)
                data = self.edge_encoder(data)
        else:
            if data.edge_features.shape[1] == 0:
                #print("NO.")
                data.edge_features = torch.zeros((data.edge_features.shape[0], config.HIDDEN_SIZE)).to(config.device)
            else:
                data.edge_features = self.edge_encoder(data.edge_features).reshape((data.edge_features.shape[0], -1))
        data.x = self.encoder(torch.cat([data.x,torch.rand(data.x.shape[0],1,device=getDevice())],dim=-1))
        return data


class ProcessModule(nn.Module):
    def __init__(self, ):
        super(ProcessModule, self).__init__()

        self.merger = mlps.WideMLP(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.HIDDEN_LAYERS)
        self.node_edge_merger = mlps.WideMLP(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.HIDDEN_LAYERS)

        self.merger_rev = mlps.WideMLP(config.HIDDEN_SIZE + 1, config.HIDDEN_SIZE, config.HIDDEN_SIZE, config.HIDDEN_LAYERS)
        self.node_edge_merger_rev = mlps.WideMLP(config.HIDDEN_SIZE * 2, config.HIDDEN_SIZE, config.HIDDEN_SIZE,
                                             config.HIDDEN_LAYERS)
        self.activation=nn.Tanh()

    def forward_up(self, data):
        u_nodes_depths=torch.index_select(data.depths,0,data.edge_index[0])
        v_nodes_depths=torch.index_select(data.depths,0,data.edge_index[1])

        # merge edges and nodes to corresponding nodes in higher layer
        starting_node_features = data.x[data.edge_index[0][u_nodes_depths==0]]
        merged_node_edge_features = self.node_edge_merger(torch.cat([starting_node_features, data.edge_features], dim=1))
        data.x[data.edge_index[1][v_nodes_depths==1]] = merged_node_edge_features

        #print(data.x[31])

        # merge nodes through
        for depth in range(1,config.GCMN_DEPTH):
            left_features = data.x[data.edge_index[0][(v_nodes_depths==depth+1) & (data.edge_states == 0)]]
            right_features = data.x[data.edge_index[0][(v_nodes_depths==depth+1) & (data.edge_states == 1)]]
            
            #print("up: "+str(depth))
            #print(data.x[31])

            merged_features = self.merger(torch.cat([left_features, right_features], dim=1))

            data.x[data.edge_index[1][(v_nodes_depths == depth+1 ) & (data.edge_states == 1)]] = merged_features

            #print(data.x[31])

        return data

    def forward_down(self, data):
        v_nodes_depths=torch.index_select(data.depths,0,data.edge_index[1])
        #print(v_nodes_depths)
        for depth in range(config.GCMN_DEPTH,0,-1):
            top_features = data.x[data.edge_index[1][(v_nodes_depths==depth)]]
            bottom_edge_mask = (v_nodes_depths==depth)
            #print(bottom_edge_mask)
            bottom_features = data.x[data.edge_index[0][bottom_edge_mask]]
            merged_features = self.merger_rev(torch.cat([top_features, 
                torch.unsqueeze(data.edge_states[v_nodes_depths==depth],dim=1)],dim=1))

            #print("down: "+str(depth))
            #print(data.x[31])

            data.x[data.edge_index[0][bottom_edge_mask]]=torch.zeros((data.x[data.edge_index[0][bottom_edge_mask]].shape[0],data.x.shape[1]),device=getDevice())
            #print(data.x[31])

            #torch_scatter.scatter_max(merged_features,data.edge_index[0][bottom_edge_mask],dim=0,out=data.x)
            out = torch_scatter.scatter_sum(merged_features,data.edge_index[0][bottom_edge_mask],dim=0)
            data.x[data.edge_index[0][bottom_edge_mask]]=torch.add(out[data.edge_index[0][bottom_edge_mask]],bottom_features)

            #print(data.x[31])

        return data

    def forward(self, data):
        return self.forward_down(self.forward_up(data))

class DecodeModule(nn.Module):
    def __init__(self, ):
        super(DecodeModule, self).__init__()
        self.decoder = mlps.WideMLP(config.HIDDEN_SIZE, config.OUT_SIZE, config.HIDDEN_SIZE, 1,
                                    last_activation=nn.Identity)

    def forward(self, x):
        pred = self.decoder(x)
        if config.BINARY_OUTPUT:
            pred = torch.sigmoid(pred)
        else:
            pred = pred  # nn.LeakyReLU()(pred)
        return pred

# Different from hldmodel
class NodeReadoutModule(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, data):
        return data.x[data.depths == 0]

class ReadoutModule(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.node_readout = NodeReadoutModule()
        self.aggregation=config.FINAL_AGGR()#aggr.SoftmaxAggregation(learn=True)

    def forward(self, data):
        out=self.aggregation(self.node_readout(data),data.batch[data.depths == 0])
        return out

class GCMNModel(nn.Module):
    def __init__(self, node_readout_module=False):
        super(GCMNModel, self).__init__()
        self.encoding_module = EncodeModule()
        self.decode_module = DecodeModule()

        self.up_down_modules = nn.ModuleList([
                ProcessModule() for i in range(config.UP_DOWN_MODULES)
        ])

        if node_readout_module:
            self.readout_module = NodeReadoutModule()
        else:
            self.readout_module = ReadoutModule()

    def forward(self, data):
        # Encode input features in leafs
        data = self.encoding_module(data)

        # Up down modules
        for up_down_module in self.up_down_modules:
            data = up_down_module(data)

        # Final decoding from heads
        data.y_computed = self.decode_module(
            self.readout_module(data)
        )
        return data