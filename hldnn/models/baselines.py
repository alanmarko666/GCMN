import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing,global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import add_self_loops, degree
import models.mlps as mlps
from config import config
#from tutorial https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
class MPNNLayer(MessagePassing):
    def __init__(self, HIDDEN_SIZE, hidden_edge_channels):
        super().__init__(aggr='add')
        self.edge_MLP = Linear(3*HIDDEN_SIZE, HIDDEN_SIZE, bias=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        # Step 4-5: Start propagating messages.
        
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.shape[1],config.HIDDEN_SIZE)).to(config.device)
        data.x = self.propagate(edge_index, x=x, e=edge_attr)
        
        return data

    def message(self, x_i, x_j, e):
        return self.edge_MLP(torch.cat([x_i, x_j, e], dim=-1))

class NodeModel(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.model = model(*args, **kwargs)
    
    def forward(self, data):
        data.x = self.model(data.x)
        return data
    
    


class MPNNModel(torch.nn.Module):
    def __init__(self,node_readout_module = False, activation = torch.nn.LeakyReLU):
        super().__init__()
        self.encode_module = NodeModel(mlps.WideMLP, config.IN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        self.process_modules = torch.nn.Sequential(*(
            [
                    torch.nn.Sequential(
                        MPNNLayer(config.HIDDEN_SIZE, config.HIDDEN_SIZE),
                        NodeModel(mlps.WideMLP, config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
                    )
            ]*(config.BASELINE_MESSAGE_PASSING)))

        
        if node_readout_module:
            self.decode_module = NodeModel(mlps.WideMLP, config.HIDDEN_SIZE,config.OUT_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        else:
            self.decode_module = mlps.WideMLP(3*config.HIDDEN_SIZE,config.OUT_SIZE,config.HIDDEN_SIZE,config.HIDDEN_LAYERS)
        self.node_readout_module = node_readout_module

    def forward(self,data):
        # Encode input features in leafs
        data.x=data.x.float()
        data=self.encode_module(data)
        
        # Process HLD trees from bottom to top
        data=self.process_modules(data)

        # Final decoding from heads
        if self.node_readout_module:
            data.y_computed=self.decode_module(data).x
        else:
            #print(self.decode_module)
            #print(torch.cat([f(data.x, data.batch) for f in [global_mean_pool, global_max_pool, global_add_pool]]).shape)
            data.y_computed=self.decode_module(torch.cat([f(data.x, data.batch) for f in [global_mean_pool, global_max_pool, global_add_pool]], dim=-1))
        if config.BINARY_OUTPUT:
            data.y_computed = torch.sigmoid(data.y_computed)
        return data

    def regularizing_loss(self,data):
        return torch.zeros((1,)).to(config.device)
    