import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

import sys
sys.path.append('.')

def visualize_data(data,feature,coloring,edge_attr=None):
    g=nx.DiGraph()
    edges=zip(*(data.edge_index.tolist()))
    g.add_edges_from(zip(*(data.edge_index.tolist())))
    edge_color=[0 for u,v in edges]
    if edge_attr is not None:
        edge_color_map={}
        for i in range(edge_attr.shape[0]):
            edge_color_map[list(zip(*(data.edge_index.tolist())))[i]]=edge_attr[i][0].item()
        edge_color=[edge_color_map[(u,v)] for u,v in g.edges()]
    
    pos = graphviz_layout(g, prog='twopi')
    color_map={
        0:'blue',
        1:'black'
    }
    nx.draw(g,
        pos, 
        node_color=[coloring[node][feature].item()
                    for node in g],
        with_labels=True,
        edge_color=edge_color,
        vmin=0,
        vmax=1)
    plt.show()