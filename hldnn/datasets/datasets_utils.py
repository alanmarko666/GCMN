import torch

def getDevice():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_adjacency_list_with_features(data):
    n = 1 if data.edge_index.shape[1]==0 else int(torch.max(data.edge_index)+1)
    adj_list = [[] for i in range(n)]
    for i in range(data.edge_index.T.shape[0]):
        a,b=data.edge_index.T[i]
        feature= torch.zeros(0) if data.edge_attr is None else data.edge_attr[i] #,device=getDevice()
        adj_list[a].append((int(b),feature))
    return adj_list

def get_adjacency_list(edge_index):
    n = 1 if edge_index.shape[1]==0 else int(torch.max(edge_index)+1)
    adj_list = [[] for i in range(n)]
    for a,b in edge_index.T:
        adj_list[int(a)].append(int(b))
    #print(adj_list)
    return adj_list

def edge_index_to_tensor(edge_index):
    #device=getDevice()
    rol, col = zip(*edge_index)
    edge_index = [rol, col]
    edge_index = torch.tensor(edge_index, dtype=torch.long)#, device=device
    return edge_index#.to(device).type(torch.long)