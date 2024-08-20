import torch
from datasets.dataset import Dataset
import torch
import torch_geometric
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import random
from utils.visualize_data import visualize_data
from config import config

from datasets.datasets_utils import get_adjacency_list_with_features, edge_index_to_tensor #, getDevice

def build_rooted_gcmn_subtree(adj_list, root, offset, edge_feat_tensor):
    n_nodes=len(adj_list)
    
    traversal = []
    visited = set()
    queue = [(root, -1, None)]  # u,v,feature

    parent_links = [-1 for _ in
                    range(len(adj_list) * config.GCMN_DEPTH)]  # for each node GCMN_DEPTH different chains up

    parent_features = [torch.unsqueeze(torch.zeros_like(edge_feat_tensor),dim=0) for _ in range(n_nodes)] #, device=getDevice()

    gcmn_edges = []

    while queue:
        node, prev, feature = queue.pop(0)
        if node not in visited:
            visited.add(node)
            if prev != -1:
                traversal.append((node, prev, feature))
                parent_features[node]=torch.unsqueeze(feature,dim=0)

                merging_node = prev * config.GCMN_DEPTH
                for d in range(1, config.GCMN_DEPTH):
                    parent_links[node * config.GCMN_DEPTH + d] = merging_node

                    gcmn_edges.append((merging_node, node * config.GCMN_DEPTH + d, 0))
                    gcmn_edges.append((node * config.GCMN_DEPTH + d - 1, node * config.GCMN_DEPTH + d, 1))

                    if merging_node != -1:
                        merging_node = parent_links[merging_node + 1] 
                        if merging_node != -1:
                            merging_node = merging_node + 1
            else:
                for d in range(1, config.GCMN_DEPTH):
                    gcmn_edges.append((-1, node * config.GCMN_DEPTH + d, 0))
                    gcmn_edges.append((node * config.GCMN_DEPTH + d - 1, node * config.GCMN_DEPTH + d, 1))

            neighbors = adj_list[node]

            # if there are multiple edges to choose from, select one randomly
            if len(neighbors) > 1:
                random.shuffle(neighbors)
            for neighbor, neighbor_feature in neighbors:
                queue.append((neighbor, node, neighbor_feature))


    gcmn_edges=[(t[0] + offset if t[0]!=-1 else 0, t[1] + offset if t[1]!=-1 else 0, t[2]) for t in gcmn_edges]
    for v in range(len(adj_list)):
        gcmn_edges.append((v+1, config.GCMN_DEPTH * v + offset, 2))

    edges_tensor = edge_index_to_tensor([(t[0],t[1]) for t in gcmn_edges])
    edge_states_tensor = torch.tensor([t[2] for t in gcmn_edges], dtype=torch.long) #,device=getDevice()
    edge_features_tensor = torch.cat(parent_features, dim=0)

    node_depths=torch.tensor([i for i in range(1,config.GCMN_DEPTH+1)]) #,device=getDevice()
    depths=torch.cat([node_depths for i in range(n_nodes)],dim=0)
    
    return edges_tensor, edge_states_tensor, edge_features_tensor, depths, n_nodes*config.GCMN_DEPTH
    

def build_gcmn_graph(data):
    adj_list = get_adjacency_list_with_features(data)
    n = 1 if data.edge_index.shape[1] == 0 else int(torch.max(data.edge_index) + 1) #in case that we have no edges

    #Loop through all nodes, create single GCMN subtree and then merge all subtrees together
    n_gcmn_nodes = 1 + len(adj_list)

    edge_index=torch.tensor([[],[]], dtype=torch.long) #,device=getDevice()
    edge_states=torch.tensor([], dtype=torch.long) #,device=getDevice()
    edge_features=torch.tensor([]) #,device=getDevice()
    depths = torch.tensor([-1]+[0 for i in range(n_gcmn_nodes-1)]) #,device=getDevice()

    for i in range(0,n,config.GCMN_FREQ):
        new_edges, new_states, new_features, new_depths, n_new_nodes = build_rooted_gcmn_subtree(adj_list,i,n_gcmn_nodes, torch.zeros(config.INIT_EDGE_SIZE) if data.edge_attr is None or data.edge_attr.shape[0] == 0 else data.edge_attr[0])
        edge_index=torch.cat([edge_index,new_edges],dim=1)
        edge_states=torch.cat([edge_states,new_states],dim=0)
        edge_features=torch.cat([edge_features,new_features],dim=0)
        depths=torch.cat([depths,new_depths],dim=0)

        n_gcmn_nodes+=n_new_nodes

    return edge_index, depths, edge_states, edge_features


#Takes in task data and transforms them to GCMN structure
data_created_counter=0
def transform_to_gcmn(data):
    global data_created_counter
    data_created_counter += 1
    if data_created_counter % 100 == 0:
        print("Created: "+str(data_created_counter))
    #visualize_data(data,0,data.x)
    #visualize_data(data,1,data.x)
    #visualize_data(data,2,data.x)

    #print(data.edge_index)

    edge_index,depths, edge_states, edge_features = build_gcmn_graph(data)
    data.edge_index=edge_index

    #print(data.edge_index)

    node_features = torch.cat([torch.zeros((1, data.x.shape[1])), data.x, torch.zeros((depths.shape[0]-data.x.shape[0]-1, data.x.shape[1]))])

    #visualize_data(data,0,node_features)
    #visualize_data(data,1,node_features)
    #visualize_data(data,2,node_features)

    #print(node_features.shape)
    #print(edge_states.shape)
    #print(depths.shape)
    #print(edge_features.shape)
    #print(edge_index.shape)

    return torch_geometric.data.Data.from_dict({**(data.to_dict()),
                                    'x':node_features, # dummy 0 node, original nodes, gcmn nodes build from each node
                                    'edge_index':edge_index, # gcmn edges+edges to original nodes
                                    'edge_states':edge_states, # 0-connection with upper merging chain 1 - with lower 2 - connection to original nodes
                                    'depths':depths, # depths of each node
                                    'edge_features':edge_features,
                                    'edge_attr':torch.zeros(1,1) #edge attr not needed after transformation
                                    })