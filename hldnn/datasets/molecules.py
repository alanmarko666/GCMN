import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import datasets.line_dataset as line_dataset
import datasets.peptides_structural
from torch_geometric.datasets import ZINC
from torch_geometric.datasets import AQSOL, TUDataset, QM9
from config import config
import torch_geometric.datasets
from datasets.transform_gcmn import transform_to_gcmn

from torch.utils.data import ConcatDataset,Subset

def get_adjacency_list_with_features(data):
    n = int(torch.max(data.edge_index)+1)
    adj_list = [[] for i in range(n)]
    for i in range(data.edge_index.T.shape[0]):
        a,b=data.edge_index.T[i]
        feature=data.edge_attr[i]
        adj_list[a].append((int(b),feature))
    return adj_list

def get_spanning_tree(data):
    adj_list=get_adjacency_list_with_features(data)
    used=set()
    new_edge_index=[[],[]]
    new_edge_attr=[]
    def dfs(v):
        used.add(v)
        for child,feature in adj_list[v]:
            if child in used:
                continue
            new_edge_index[0].append(v)
            new_edge_index[1].append(child)
            new_edge_index[0].append(child)
            new_edge_index[1].append(v)
            new_edge_attr.append(feature)
            new_edge_attr.append(feature)
            dfs(child)
    dfs(0)
    return new_edge_index,new_edge_attr

class FilterOneComponent(object):
    def __call__(self, data):
        if data.edge_attr.dim()==1:
            data.edge_attr=torch.unsqueeze(data.edge_attr,1)
        
        if len(data.edge_index[0])==0:
            return False

        if data.x.dim()==1:
            data.x=torch.unsqueeze(data.x,1)
        
        new_edge_index,new_edge_attr=get_spanning_tree(data)

        if len(new_edge_index[0])==0:
            return False

        if ((len(new_edge_index[0])//2)+1) != data.x.shape[0]:
            return False
        return True

class SpanningTreeTransform(object):
    def __call__(self, data):
        if data.edge_attr.dim()==1:
            data.edge_attr=torch.unsqueeze(data.edge_attr,1)
        
        if data.x.dim()==1:
            data.x=torch.unsqueeze(data.x,1)
        
        new_edge_index,new_edge_attr=get_spanning_tree(data)
        #print("dei:",data.edge_index)
        #print("ei: ",new_edge_index)

        

        data.edge_index=torch.tensor(new_edge_index,dtype=torch.int64)
        if len(new_edge_attr)==0:
            data.edge_attr=torch.rand((0,config.EDGE_SIZE))
        else: 
            data.edge_attr=torch.vstack(new_edge_attr).float()#None
        #print(data.edge_attr)
        n_nodes=(len(new_edge_index[0])//2)+1
        data.x=data.x[0:n_nodes]

        if config.HLD_Transform:
            data=line_dataset.transform_to_hld_tree(data)
        elif config.GCMN_Transform:
            data=transform_to_gcmn(data)
        return data

def get_molhiv_tree_dataset(split="train"):
    if config.RANDOM_ROOT:
        datasets=[]
        for i in range(config.REPEAT_COUNT):
            dataset = PygGraphPropPredDataset(root="dataset/molhiv"+str(i),name = "ogbg-molhiv",pre_transform=SpanningTreeTransform())
            split_idx = dataset.get_idx_split()
            dataset=dataset[split_idx[split]]
            datasets.append(dataset)
        return ConcatDataset(datasets)
    else:
        dataset = PygGraphPropPredDataset(root="dataset/molhiv2",name = "ogbg-molhiv",pre_transform=SpanningTreeTransform())#,pre_filter=FilterOneComponent()
        split_idx = dataset.get_idx_split()
        dataset=dataset[split_idx[split]]
        return dataset

def get_peptides_tree_dataset(split="train"):
    dataset = datasets.peptides_functional.PeptidesFunctionalDataset(pre_transform=SpanningTreeTransform())
    split_idx = dataset.get_idx_split()
    dataset=dataset[split_idx[split]]
    return dataset

def get_peptides_structural_tree_dataset(split="train"):
    if config.RANDOM_ROOT:
      dataset_list=[]
      for i in range(config.REPEAT_COUNT):
          dataset = datasets.peptides_structural.PeptidesStructuralDataset(root="dataset/pepstruct"+str(i),pre_transform=SpanningTreeTransform())
          split_idx = dataset.get_idx_split()
          dataset=dataset[split_idx[split]]
          dataset_list.append(dataset)
      return ConcatDataset(datasets)
    else:  
      dataset = datasets.peptides_structural.PeptidesStructuralDataset(root="dataset/pepstruct2",pre_transform=SpanningTreeTransform())
      split_idx = dataset.get_idx_split()
      dataset=dataset[split_idx[split]]
      return dataset

def get_zinc_tree_dataset(split="train"):
    dataset = ZINC(root='dataset', split=split, subset=True,pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent())
    return dataset

def get_aqsol_tree_dataset(split="train"):
    if config.RANDOM_ROOT:
      dataset=ConcatDataset([AQSOL(root='dataset/fullrand'+str(i), split=split,pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent()) for i in range(config.REPEAT_COUNT)])
    else:
      dataset = AQSOL(root='dataset', split=split,pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent())
    return dataset

def get_mutag_tree_dataset(split="train"):
    if config.RANDOM_ROOT:
      dataset=ConcatDataset([TUDataset(root='dataset/MUTAG/fullrand'+str(i), name="MUTAG", split=split,pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent(), use_node_attr=True, use_edge_att=True) for i in range(config.REPEAT_COUNT)])
    else:
      dataset = TUDataset(root='dataset/MUTAG/', name="MUTAG", split=split,pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent(), use_node_attr=True, use_edge_att=True)
    return dataset


class SetTarget(object):
    def __init__(self, target=0):
        self.target = target

    def __call__(self, data):
        data.y = data.y[:, self.target]
        return data

def get_qm9_tree_dataset(split="train", target=0):
    transform = torch_geometric.transforms.Compose([SpanningTreeTransform(), SetTarget(config.TARGET_Y)])
    len_dataset = len(QM9(root='dataset/QM9/fullrand'+str(i),pre_transform=transform,pre_filter=FilterOneComponent(), use_node_attr=True, use_edge_att=True))
    subset= {
            'train':range(int(0.8*len_dataset)),
            'test':range(int(0.8*len_dataset),int(0.9*len_dataset)),
            'val':range(int(0.9*len_dataset),len_dataset),
            }[split]
    if config.RANDOM_ROOT:
        dataset=ConcatDataset([Subset(QM9(root='dataset/QM9/fullrand'+str(i),pre_transform=transform,pre_filter=FilterOneComponent(), use_node_attr=True, use_edge_att=True), subset) for i in range(config.REPEAT_COUNT)])
    else:
        dataset = Subset(QM9(root='dataset/QM9/',pre_transform=transform,pre_filter=FilterOneComponent(), use_node_attr=True, use_edge_att=True),subset)
    return dataset


def get_split(dataset,split="train"):
    len_dataset = len(dataset)
    subset= {
            'train':range(int(0.8*len_dataset)),
            'test':range(int(0.8*len_dataset),int(0.9*len_dataset)),
            'val':range(int(0.9*len_dataset),len_dataset),
            }[split]
    return Subset(dataset,subset)

def get_esol_tree_dataset(split="train"):
    if config.RANDOM_ROOT:
      dataset=ConcatDataset([
        get_split(torch_geometric.datasets.MoleculeNet(root="dataset/esol"+str(i),name="ESOL", pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent()),split)
        for i in range(config.REPEAT_COUNT)])
    else:
      dataset = get_split(torch_geometric.datasets.MoleculeNet(root="dataset/esolgcmn",name="ESOL", pre_transform=SpanningTreeTransform(),pre_filter=FilterOneComponent()),split)
    return dataset
