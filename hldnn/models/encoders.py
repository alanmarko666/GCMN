import torch
from config import config

#From https://github.com/bayer-science-for-a-better-life/molnet-geometric-lightning/blob/main/molnet_geometric_lightning/mol_encoder.py

x_map_default = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}
 
e_map_default = {
    'bond_type': [
        'misc',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        'AROMATIC',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
        'STEREOANY',
    ],
    'is_conjugated': [False, True],
}

def get_atom_feature_dims():
    allowable_features = x_map_default
    return list(map(len, [
        allowable_features['atomic_num'],
        allowable_features['chirality'],
        allowable_features['degree'],
        allowable_features['formal_charge'],
        allowable_features['num_hs'],
        allowable_features['num_radical_electrons'],
        allowable_features['hybridization'],
        allowable_features['is_aromatic'],
        allowable_features['is_in_ring']
        ]))


def get_bond_feature_dims():
    allowable_features = e_map_default
    return [i+10 for i in list(map(len, [
        allowable_features['bond_type'],
        allowable_features['stereo'],
        allowable_features['is_conjugated']
        ]))]


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        full_atom_feature_dims = get_atom_feature_dims()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, data):
        data.x=data.x.to(torch.int64)
        x_embedding = 0
        for i in range(data.x.shape[1]):
            x_embedding += self.atom_embedding_list[i](data.x[:, i])

        data.x=x_embedding
        return data

class BondEncoderOnFeatures(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoderOnFeatures, self).__init__()
        full_bond_feature_dims = get_bond_feature_dims()
        #print("fbfd: "+str(full_bond_feature_dims))
        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        #print(torch.max(edge_attr,dim=0))
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class BondEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.encoder=BondEncoderOnFeatures(emb_dim)

    def forward(self, batch):
        if config.MODEL == "GCMN":
            batch.edge_features=batch.edge_features.to(torch.int64)
            batch.edge_features = torch.flatten(self.encoder(batch.edge_features),start_dim=1)
        else:
            batch.parent_edge_features=batch.parent_edge_features.to(torch.int64)
            batch.parent_light_edge_features=batch.parent_light_edge_features.to(torch.int64)
        
            batch.parent_edge_features = torch.flatten(self.encoder(batch.parent_edge_features),start_dim=1)
            batch.parent_light_edge_features = torch.flatten(self.encoder(batch.parent_light_edge_features),start_dim=1)
        return batch