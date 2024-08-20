import os.path as osp
import torch_geometric
import torch
from torch_geometric.data.makedirs import makedirs
import sys
sys.path.append('.')

class ASTDataset(torch_geometric.data.Dataset):
    def __init__(self, force_download=False, root='.', transform=None, pre_transform=None, pre_filter=None):
        self.n_inputs =
        
        super().__init__(root, transform, pre_transform, pre_filter)
        if force_download:
            makedirs(self.raw_dir)
            self.download()
        else:
            self._download()
        
    

    @property
    def raw_file_names(self):
        r"""The name of the files in the :obj:`self.raw_dir` folder that must
        be present in order to skip downloading."""
        return []

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        r"""The name of the files in the :obj:`self.processed_dir` folder that"""
        #return []
        return ["sam{}.pt".format(i) for i in range(self.n_inputs)]

    def download(self):
        for i in range(self.n_inputs):
            torch.save(self.generate(), osp.join(self.raw_dir, "sam{}.pt".format(i)))
        


    #def process(self):
    #    r"""Processes the dataset to the :obj:`self.processed_dir` folder."""
    #    raise NotImplementedError

    def len(self) -> int:
        r"""Returns the number of graphs stored in the dataset."""
        return self.n_inputs

    def get(self, idx: int):
        r"""Gets the data object at index :obj:`idx`."""
        return torch.load(osp.join(self.raw_dir, "sam{}.pt".format(idx)))