import numpy as np
import torch
import torch.nn as nn
from torch.nn import LeakyReLU

from config import config

def ProgressiveSequence(start, end, layers):
    return np.linspace(
        start, end,
        layers + 1,
        dtype='int'
    )


def ProgressiveMLP(start, end, layers_n, homogeneous=False,last_activation=config.default_activation, activation=config.default_activation):
    if (layers_n == 0):
        return nn.Identity()

    size_list = ProgressiveSequence(start, end, layers_n)
    return CreateMLP(
        size_list=size_list,
        last_activation=last_activation, activation=activation, bias=not homogeneous
    )

def WideMLP(in_size,out_size,middle,n_mid_layers,homogeneous=False,last_activation=config.default_activation, activation=config.default_activation):
    size_list=[in_size]+[middle for i in range(n_mid_layers)]+[out_size]
    return CreateMLP(
        size_list=size_list,
        last_activation=last_activation, activation=activation, bias=not homogeneous
    )


def CreateMLP(size_list, bias=False,last_activation=config.default_activation, activation=config.default_activation, dropout_p=0):
    return nn.Sequential(
        *(
            nn.Sequential(
                nn.Linear(size_list[i_lay], size_list[i_lay + 1], bias=(bias if i_lay != len(size_list) - 2 else bias)),
                #nn.Dropout(dropout_p) if i_lay != len(size_list) - 2 else nn.Identity(),
                #nn.BatchNorm1d(num_features=size_list[i_lay + 1]),
                activation() if i_lay != len(size_list) - 2 else last_activation()
            )
            for i_lay in range(len(size_list) - 1)
        )
    )

class AttentionMLP(nn.Module):
    def __init__(self,in_size,out_size,middle,n_mid_layers,n_attention_layers=1,splits=12,homogeneous=False,last_activation=config.default_activation, activation=config.default_activation):
        super(AttentionMLP,self).__init__()
        self.processor=WideMLP(in_size,out_size,middle,n_mid_layers,homogeneous,last_activation, activation)
        self.attention=WideMLP(in_size//splits,1,middle,n_attention_layers,last_activation=nn.Identity)
        self.in_size=in_size
        self.out_size=out_size
        self.splits=splits
    
    def forward(self,x):
        x=x.view(-1,self.splits,self.in_size//self.splits)
        attention_values=self.attention(x)
        alphas=nn.Softmax(dim=1)(attention_values)
        x=(torch.mul(x,attention_values.repeat(1,1,self.in_size//self.splits))).view(-1,self.in_size)
        x=self.processor(x)
        return x