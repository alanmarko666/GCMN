import torch
import torch_geometric
import torch.nn.functional as F
import random
import models.mlps as mlps
import torch.nn as nn
from config import config
import torch_scatter
from torch.nn.functional import normalize
from models.encoders import AtomEncoder,BondEncoder

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
        batch.x = self.encoder(batch.x[:, 0])

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

    def forward(self, batch):
        batch.parent_edge_features=batch.parent_edge_features.to(torch.int64)
        batch.parent_light_edge_features=batch.parent_light_edge_features.to(torch.int64)
        batch.parent_edge_features = torch.flatten(self.encoder(batch.parent_edge_features),start_dim=1)
        batch.parent_light_edge_features = torch.flatten(self.encoder(batch.parent_light_edge_features),start_dim=1)
        return batch

class EncodeModule(nn.Module):
    def __init__(self):
        super(EncodeModule,self).__init__()
        self.node_encoder=nn.Identity()

        if config.ATOM_ENCODER:
            self.node_encoder=AtomEncoder(config.HIDDEN_SIZE)
            self.encoder=mlps.WideMLP(config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)
        else:
            self.encoder=mlps.WideMLP(config.IN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)
            
        if config.BOND_ENCODER:
            self.edge_encoder=BondEncoder(config.HIDDEN_SIZE)
        else:
            self.edge_encoder=mlps.WideMLP(config.EDGE_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)


        if config.EDGE_ENCODER_NUM_TYPES is not None:
            self.edge_encoder=TypeDictEdgeEncoder(config.HIDDEN_SIZE)
        if config.NODE_ENCODER_NUM_TYPES is not None:
            self.encoder=mlps.WideMLP(config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,1)
            self.node_encoder=TypeDictNodeEncoder(config.HIDDEN_SIZE)

    def forward(self,data):
        data=self.node_encoder(data)
        if config.BOND_ENCODER:
            data=self.edge_encoder(data)
        else:
            if data.parent_edge_features.shape[1]==0:
                data.parent_edge_features=torch.zeros((data.x.shape[0], config.HIDDEN_SIZE)).to(config.device)
                data.parent_light_edge_features=torch.zeros((data.x.shape[0], config.HIDDEN_SIZE)).to(config.device)
            else:
                data.parent_edge_features=self.edge_encoder(data.parent_edge_features).reshape((data.x.shape[0],-1))
                data.parent_light_edge_features=self.edge_encoder(data.parent_light_edge_features).reshape((data.x.shape[0],-1))
        data.x=self.encoder(data.x)
        return data


class ProcessModule(nn.Module):
    def __init__(self,):
        super(ProcessModule,self).__init__()
        if config.ATTENTION:
            self.merger=mlps.AttentionMLP(config.HIDDEN_SIZE*2+config.HIDEEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        else:
            self.merger=mlps.WideMLP(config.HIDDEN_SIZE*2+config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        self.light_edge_processor=mlps.WideMLP(config.HIDDEN_SIZE+config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        self.light_edge_merger=mlps.WideMLP(config.HIDDEN_SIZE*2,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)

    def forward(self,data):
        parents = torch.zeros(data.x.shape[0],dtype=torch.long,device=config.device)
        parents[data.edge_index[0]] = data.edge_index[1]

        max_depth=torch.max(data.depths)
        for depth in range(max_depth,0,-1):
            #Interval tree
            mask_depth=data.depths==depth

            left_sons_mask=mask_depth & (data.states==0)
            right_sons_mask=mask_depth & (data.states==1)

            left_parents=parents[left_sons_mask]
            right_parents=parents[right_sons_mask]

            left=torch.zeros_like(data.x,device=config.device)
            right=torch.zeros_like(data.x,device=config.device)

            left = torch_scatter.scatter_add(data.x[left_sons_mask], left_parents, out=left,dim=0)
            right = torch_scatter.scatter_add(data.x[right_sons_mask], right_parents, out=right,dim=0)

            #print(left.shape, right.shape, data.parent_edge_features.shape, config.EDGE_SIZE, config.HIDDEN_SIZE)
            x_parents=torch.cat([left,right, data.parent_edge_features],dim=1)
            x_parents=self.merger(x_parents)

            parents_mask=torch.zeros((data.x.shape[0],1),device=config.device)
            parents_mask=torch_scatter.scatter_add(torch.ones((data.x.shape[0],1),device=config.device)[left_sons_mask], left_parents, out=parents_mask,dim=0).long()!=0


            #Merging two heavy paths
            heads_mask=mask_depth & (data.states==3)
            heads_parents=parents[heads_mask]

            processed_heads=self.light_edge_processor(torch.cat([data.x, data.parent_light_edge_features], dim=1)[heads_mask])

            merged_heads_sum = torch.zeros_like(data.x,device=config.device)
            torch_scatter.scatter_add(processed_heads, heads_parents, out=merged_heads_sum,dim=0)
            if config.MAX_LIGHT_EDGES:
                merged_heads_max = torch.zeros_like(data.x,device=config.device)
                torch_scatter.scatter_add(processed_heads, heads_parents, out=merged_heads_max,dim=0)

            designated_leafs_mask=torch.zeros((data.x.shape[0],1),device=config.device)
            designated_leafs_mask=torch_scatter.scatter_add(torch.ones((data.x.shape[0],1),device=config.device)[heads_mask], heads_parents, out=designated_leafs_mask,dim=0).long()!=0

            x_designated_leafs=torch.where(designated_leafs_mask, data.x,torch.zeros_like(data.x,device=config.device))
            x_merged=torch.cat([x_designated_leafs,merged_heads_sum]+([merged_heads_max] if config.MAX_LIGHT_EDGES else []),dim=1)
            x_merged=self.light_edge_merger(x_merged)
            
            data.x = torch.where(parents_mask, x_parents, 
                torch.where(designated_leafs_mask, x_merged, data.x))
            
        return data
    
    def regularizing_loss(self,data):
        def normalize_tensor(t):
            mean, std, var = torch.mean(t,dim=0,keepdim=True), torch.std(t,dim=0,keepdim=True), torch.var(t,dim=0,keepdim=True)
            t  = (t-mean)/std
            return t

        n=data.x.shape[0]
        a=data.x[torch.randperm(n,device=config.device)]
        b=data.x[torch.randperm(n,device=config.device)]
        c=data.x[torch.randperm(n,device=config.device)]

        edges_ab=data.parent_edge_features[torch.randperm(n,device=config.device)]
        edges_bc=data.parent_edge_features[torch.randperm(n,device=config.device)]

        res1=self.merger(
          torch.cat([self.merger(torch.cat([a,b,edges_ab],dim=1)),c,edges_bc],dim=1)
        )

        res2=self.merger(
          torch.cat([a,self.merger(torch.cat([b,c,edges_bc],dim=1)),edges_ab],dim=1)
        )

        res1=normalize_tensor(res1)
        res2=normalize_tensor(res2)

        return  F.mse_loss(res1,res2)

class DecodeModule(nn.Module):
    def __init__(self,):
        super(DecodeModule,self).__init__()
        self.decoder=mlps.WideMLP(config.HIDDEN_SIZE,config.OUT_SIZE,config.HIDDEN_SIZE,1,last_activation=nn.Identity)

    def forward(self,x):
        pred=self.decoder(x)
        if config.BINARY_OUTPUT:
            pred = torch.sigmoid(pred)
        else:
            pred = pred#nn.LeakyReLU()(pred)
        return pred


class ReadoutModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,data):
        return data.x[data.depths==0]

class NodeReadoutModule(nn.Module):
    def __init__(self,is_final=True):
        super().__init__()
        self.processor=mlps.WideMLP(config.HIDDEN_SIZE*2+1,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        self.is_final=is_final

    def forward(self, data):
        parents = torch.zeros(data.x.shape[0],dtype=torch.long,device=config.device)
        parents[data.edge_index[0]] = data.edge_index[1]

        max_depth=torch.max(data.depths)

        for depth in range(1, max_depth):
            current_vert_mask = data.depths==depth
            #print(current_vert_mask.shape, data.x.shape, data.x[parents].shape, data.states.shape, data.states.reshape((-1,1)).shape)
            #print(self.processor(torch.cat([data.x, data.x[parents], data.states.reshape((-1,1))],dim=1)).shape)

            merged=self.processor(torch.cat([data.x, data.x[parents], data.states.reshape((-1,1))],dim=1))
            if config.ADD_PARENTS:
                merged=torch.add(data.x,data.x[parents])

            data.x = torch.where(current_vert_mask.reshape((-1,1)), 
                merged,
                data.x)
        
        if self.is_final:
            return self.get_original_nodes(data,parents)

        return data
    
    def get_original_nodes(self, data, parents):
        original_nodes = torch.ones(data.x.shape[0],dtype=torch.bool)
        original_nodes[parents[data.states==1]] = False
        return data.x[original_nodes]



class HLDGNNModel(nn.Module):
    def __init__(self,node_readout_module = False):
        super(HLDGNNModel,self).__init__()
        self.encoding_module=EncodeModule()
        self.process_module=ProcessModule()
        self.decode_module=DecodeModule()
        
        self.up_down_modules=nn.ModuleList([
            nn.Sequential(
                ProcessModule(),
                NodeReadoutModule(is_final=False),
            ) for i in range(config.UP_DOWN_MODULES)
        ])

        if not node_readout_module:
            self.readout_module=ReadoutModule()
        else:
            self.readout_module = NodeReadoutModule()

    def forward(self,data):
        # Encode input features in leafs
        data=self.encoding_module(data)
        
        #Up down modules
        for up_down_module in self.up_down_modules:
            data=up_down_module(data)

        # Process HLD trees from bottom to top
        data=self.process_module(data)

        # Final decoding from heads
        data.y_computed=self.decode_module(
            self.readout_module(data)
        )
        return data

    def regularizing_loss(self,data):
        return self.process_module.regularizing_loss(data)
