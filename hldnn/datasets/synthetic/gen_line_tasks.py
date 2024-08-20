import sys
sys.path.append('.')
import torch
import torch_geometric
import torch.nn.functional as F
import random
import datasets.line_dataset
from config import config
from ..line_dataset import transform_to_binary_tree
from ..dataset import Dataset

def gen_line_task1_data(n):
    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    parity = torch.sum(bin_features[beg:end+1])%2
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=parity, edge_index=edge_index) #v edge index zoradene ako v x?

def gen_line_task2_data(n): #is something inside of an interval
    bin_features = F.one_hot(torch.randint(0,n,(1,)), num_classes=n)#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    parity = torch.sum(bin_features[beg:end+1])%2
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=parity, edge_index=edge_index) #v edge index zoradene ako v x?

def gen_line_task3_data(n): #count 1s inside of an interval
    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    result = torch.sum(bin_features[beg:end+1])
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=result, edge_index=edge_index) #v edge index zoradene ako v x?

def gen_line_task4_data(n): #shortest path length
    bin_features = torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    result = torch.sum(bin_features[beg:end+1])
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=result, edge_index=edge_index) #v edge index zoradene ako v x?

#find minimum
def gen_line_task5_data(n): #find minimum
    bin_features = torch.randint(torch.randint(0,20,(1,)).item(),40,(n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    if end<beg: beg,end=end,beg
    result = torch.min(bin_features[beg:end+1])
    edge_index = torch.vstack([torch.arange(n-1).long(), torch.arange(1,n).long()])
    edge_index = torch.cat([edge_index, edge_index[[1,0]]], dim=1)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    return torch_geometric.data.Data(x=node_features, y=result, edge_index=edge_index) #v edge index zoradene ako v x?

def get_line_dataset(count=10000,min_n=2,max_n=128,generator=gen_task1_data,name="",force_download=True):
    def gen_f():
        #n=random.choices([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],weights=(30,30,30,5,5,5,3,3,3,1,1,1,1,1,1,1,1,1,1),k=1)[0]
        n=random.randint(min_n,max_n)#random.choices([2,4,8,16,32,64,128],weights=(30,30,30,30,30,30,30),k=1)[0]
        ret = generator(n)
        if conifg.HLD_Transform:
            ret = transform_to_hld_tree(ret)
        return ret    return Dataset(count, gen_f,force_download=force_download, root="{}-{}-{}-{}".format(name,count, min_n, max_n))