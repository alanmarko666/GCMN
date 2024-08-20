import sys
sys.path.append('.')
import torch
from ..line_dataset import transform_to_hld_tree,get_adjacency_list
from ..transform_gcmn import transform_to_gcmn
from ..dataset import Dataset
import torch.nn.functional as F
import random
import torch_geometric
import datasets.line_dataset
from config import config
import networkx as nx
#import torch_cluster

def get_edge_index_tensor(edges):
    res = torch.Tensor(list(map(list,edges))).T.long()
    res = torch.hstack([res, res[[1,0]]])
    return res

def get_knn_component_graph(n):
    dimensions=2
    n_neighbors=3
    pos=torch.rand([n,dimensions],dtype=torch.float)
    ei=torch_cluster.knn_graph(pos,n_neighbors)
    g=nx.Graph()
    g.add_edges_from(zip(*(ei.tolist())))
    g.add_edges_from(list(nx.k_edge_augmentation(g, k=1)))
    return g
    #g=nx.waxman_graph(n)
    #g.add_edges_from(list(nx.k_edge_augmentation(g, k=1)))
    #print(g.edges, flush=True)
    #return g

def generate_synthetic_graph_edges(n):
    g=get_knn_component_graph(n)
    return get_edge_index_tensor(g.edges)
    
def generate_synthetic_graph_edges2(n):
    count = random.randint(3,n-3)
    g=get_knn_component_graph(count)
    h=get_knn_component_graph(n-count)
    res=nx.disjoint_union(g,h)
    return get_edge_index_tensor(res.edges)

def generate_graph_task_1_data(n):
    def find_answer_task1(data, start, endIndex=2):
        adj_list = get_adjacency_list(data.edge_index)
        queue = [(start,0)]
        visited = set()
        while queue:
            current,count = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            count += data.x[current][0].item()
            if data.x[current][endIndex].item() == 1:
                return count
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    queue.append((neighbor,count))
        return None

    bin_features = torch.randint(1,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    beg2, end2 = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n),F.one_hot(beg2, num_classes=n),F.one_hot(end2, num_classes=n)]).T
    edge_index = generate_synthetic_graph_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    len1=find_answer_task1(data,beg,endIndex=2)
    len2=find_answer_task1(data,beg2,endIndex=4)
    data.y = 0 if len1<len2 else 1
    return  data

def generate_graph_task_2_data(n): 
    def find_answer_task1(data, start, endIndex=2):
        adj_list = get_adjacency_list(data.edge_index)
        queue = [(start,0)]
        visited = set()
        while queue:
            current,count = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            count += data.x[current][0].item()
            if data.x[current][endIndex].item() == 1:
                return count
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    queue.append((neighbor,count))
        return None

    bin_features = torch.randint(1,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n-1, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_graph_edges2(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    len1=find_answer_task1(data,beg,endIndex=2)
    data.y = 0 if len1 is None else 1
    return  data

def generate_graph_task_3_data(n):
    bin_features = torch.randint(0,1, (n,))
    def find_answer_task1(data, start, endIndex=2, alpha=10):
        adj_list = get_adjacency_list(data.edge_index)
        queue = [(start,(0,0))]
        visited = set()
        res_length=-1
        while queue:
            current,(count,count_c) = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            if count == alpha:
              bin_features[current]=1
              count_c += 1
            count += 1
            if data.x[current][endIndex].item() == 1:
                res_length=count_c
            for neighbor in adj_list[current]:
                if neighbor not in visited:
                    queue.append((neighbor,(count,count_c)))
        return res_length

    #torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_graph_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    alpha=random.randint(3, max(3,n//3))
    len1=find_answer_task1(data,beg,endIndex=2, alpha = alpha)
    data.x=torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    data.y = 1 if len1>0 else 0
    #print(alpha)
    #print(beg)
    #print(end)
    return  data

def generate_synthetic_tree_edges(n):
    import networkx as nx
    tree = nx.random_tree(n=n)
    return get_edge_index_tensor(tree.edges)
    
def generate_tree_task_1_data(n):
    def find_answer_task1(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(1,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task1(data,beg)
    data.y=result
    return  data

def generate_tree_task_2_data(n):#parity, random dots
    def find_answer_task1(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task1(data,beg)%2
    data.y=result
    return  data

def generate_tree_task_8order_data(n):#order
    def find_answer_task1(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    beg, end,mid = torch.randint(0, n, (3,))
    bin_features = F.one_hot(mid, num_classes=n)
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task1(data,beg)%2
    data.y=result
    return  data

def generate_tree_task_3_data(n):#not, random dots
    def find_answer_task(data,start):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result=find_answer_task(data,beg)
    data.y=result
    return  data

def generate_tree_task_4_data(n):#vertices on shortest path
    def find_answer_task(data,start,result):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=1#data.x[v][0].item()
            found=data.x[v][2].item() == 1
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            result[v] = found
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.randint(0,2, (n,))#torch.ones(n)
    beg, end = torch.randint(0, n, (2,))
    node_features = torch.vstack([bin_features,F.one_hot(beg, num_classes=n),F.one_hot(end, num_classes=n)]).T
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    result = torch.zeros((n,))
    find_answer_task(data,beg, result)
    data.y=result
    return  data

def generate_tree_task_5_data(n):#lca
    def find_answer_task(data,start,end, result):
        adj_list=get_adjacency_list(data.edge_index)
        def dfs(v,parent):
            count=1#data.x[v][0].item()
            found= v==end
            for child in adj_list[v]:
                if child==parent:
                    continue
                count_child,found_child =  dfs(child,v)
                if found_child:
                    count += count_child
                found = found or found_child
            result[v] = found
            return count, found
        count,_=dfs(start,-1)
        return count

    bin_features = torch.zeros((n))
    a, b, c = torch.randint(0, n, (3,))
    bin_features[a] = bin_features[b] = bin_features[c] = 1
    bin_features = bin_features.reshape((n,1))
    node_features = torch.cat([bin_features,bin_features,bin_features],dim=1)
     
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, y=-1, edge_index=edge_index)
    res1 = torch.zeros((n,1), dtype=torch.bool)
    res2 = torch.zeros((n,1), dtype=torch.bool)
    res3 = torch.zeros((n,1), dtype=torch.bool)
    find_answer_task(data,a, b, res1)
    find_answer_task(data,b, c, res2)
    find_answer_task(data,a, c, res3)
    data.y=res1&res2&res3
    return data

def generate_tree_task_6_data(n,scaling=1.0):#minimum vertex cover on tree
    def find_answer_task(data):
        adj_list=get_adjacency_list(data.edge_index)
        memo = torch.zeros((n,2))
        ans = torch.zeros((n,1)).bool()
        def dfs(v,parent):
            node_included, node_not_included = data.x[v][0].item(),0
            for child in adj_list[v]:
                if child==parent:
                    continue
                child_included, child_not_included =  dfs(child,v)
                node_included += child_not_included
                node_not_included += max(child_included, child_not_included)
            memo[v][0], memo[v][1] = node_included, node_not_included
            return node_included, node_not_included
        poss = dfs(0,-1)
        def dfs2(v, parent, can_include):
            can_include_children = True
            ans[v] = False
            if can_include and memo[v][0]>memo[v][1]:
                can_include_children = False
                ans[v] = True
            for child in adj_list[v]:
                if child == parent: continue
                dfs2(child, v, can_include_children)
        
        dfs2(0, -1, True)
        return max(poss), ans

    
    node_features = torch.randint(0,10,(n,1)).float()*scaling
    
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, edge_index=edge_index)
    maxi, ans = find_answer_task(data)
    data.y = torch.sum(node_features)-maxi #remove n- for maximum independent set
    data.y_vert = ans
    return data

def generate_tree_task_7_data(n,scaling=1.0):#minimum vertex cover on tree with vertices
    r = 0#random.randrange(0,n)
    def find_answer_task(data):
        adj_list=get_adjacency_list(data.edge_index)
        memo = torch.zeros((n,2))
        ans = torch.zeros((n,1)).bool()
        def dfs(v,parent):
            node_included, node_not_included = data.x[v][0].item(),0
            for child in adj_list[v]:
                if child==parent:
                    continue
                child_included, child_not_included =  dfs(child,v)
                node_included += child_not_included
                node_not_included += max(child_included, child_not_included)
            memo[v][0], memo[v][1] = node_included, node_not_included
            return node_included, node_not_included
        poss = dfs(0,-1)
        def dfs2(v, parent, can_include):
            can_include_children = True
            ans[v] = False
            if can_include and memo[v][0]>memo[v][1]:
                can_include_children = False
                ans[v] = True
            for child in adj_list[v]:
                if child == parent: continue
                dfs2(child, v, can_include_children)
        
        dfs2(r, -1, True)
        return max(poss), ans

    root_feat = torch.zeros((n,1))
    root_feat[r,0] = 1
    node_features = torch.hstack([torch.randint(0,10,(n,1)).float()*scaling, root_feat])
    
    edge_index = generate_synthetic_tree_edges(n)
    data=torch_geometric.data.Data(x=node_features, edge_index=edge_index)
    maxi, ans = find_answer_task(data)
    #data.y = torch.sum(node_features)-maxi #remove n- for maximum independent set
    data.y = ans
    return data


def get_tree_dataset(count=10000,min_n=2,max_n=128,generator=generate_tree_task_1_data,name="",force_download=True):
    def gen_f():
        #n=random.choices([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],weights=(30,30,30,5,5,5,3,3,3,1,1,1,1,1,1,1,1,1,1),k=1)[0]
        n=random.randint(min_n,max_n)#random.choices([2,4,8,16,32,64,128],weights=(30,30,30,30,30,30,30),k=1)[0]
        ret = generator(n)
        if config.HLD_Transform:
            ret = transform_to_hld_tree(ret)
        elif config.GCMN_Transform:
            ret = transform_to_gcmn(ret)
        return ret
    return Dataset(count, gen_f,force_download=force_download, root="{}-{}-{}-{}{}{}".format(name,count, min_n, max_n, "" if config.HLD_Transform else "raw","GCMN" if config.GCMN_Transform else ""))