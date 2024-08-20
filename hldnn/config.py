import torch
import torch.nn.functional as F
import torch.nn as nn
import wandb
from pipeline.metrics import binary_MSE_loss, binary_MAE_loss, binary_int_precision
from torch_geometric.nn import aggr

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__dict__.update({key: value})
    
    def set_config(self, config):
        for key,val in config.items():
            self.__setitem__(key,val)
    

class DefaultConfig(Config):
    def __init__(self,):
        super().__init__()
        self.NUM_WORKERS=2
        self.ATOM_ENCODER=False
        self.BOND_ENCODER=False
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.wandb = True
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=3
        self.IN_SIZE=3
        self.OUT_SIZE=1
        self.EDGE_SIZE = 0
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.REGUL_LOSS_BETA=0.1
        self.criterion=F.mse_loss
        self.METRICS=["LOSS"]
        self.default_activation=nn.LeakyReLU
        self.SCHEDULER_LAMBDA=None
        self.MAX_LIGHT_EDGES = False
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.default_activation=nn.LeakyReLU
        self.SCHEDULER_LAMBDA=None
        self.SCHEDULER_PLATEAU=True
        self.PATIENCE=500
        self.ATTENTION=False
        self.UP_DOWN_MODULES=0
        self.RANDOM_ROOT=False
        self.MEAN_AGGR=False
        self.ADD_PARENTS=False
        self.REPEAT_COUNT=1
        self.DATASET_TRAIN_SIZE = 10000
        self.DATASET_TRAIN_MIN_N = 2
        self.DATASET_TRAIN_MAX_N = 100
        self.DATASET_VALID_SIZE = 300
        self.DATASET_VALID_MIN_N = 2
        self.DATASET_VALID_MAX_N = 100
        self.DATASET_TESTS = [(100, 2, 100),(100, 100, 10000)]
        self.NODE_READOUT = False
        self.DATASET=None
        self.TEST_SMALL_DATASET = None
        self.TARGET_Y = 0
        self.epoch_num = 1000
        self.HLD_Transform = True
        self.BASELINE_MESSAGE_PASSING=30
        self.MODEL="hldgnn"
        self.BOND_ENCODER=False
        self.SEED = 0
        self.BATCH_SIZE=32
        self.EVAL_BATCH_SIZE=32
        self.LR=1e-3
        self.TEST_EVAL_FREQ=1
        self.GCMN_FREQ=1
        self.INIT_EDGE_SIZE=3
        self.FINAL_AGGR=aggr.SumAggregation

class MoleculesConfig(Config):
    def __init__(self,):
        super().__init__()
        self.HIDDEN_SIZE=64
        self.IN_SIZE=9
        self.OUT_SIZE=1
        self.MOLECULES=True #Use OGB Evaluator and loss

class PeptidesConfig(Config):
    def __init__(self,):
        super().__init__()
        self.ATOM_ENCODER=True
        self.HIDDEN_SIZE=200
        self.HIDDEN_LAYERS=3
        self.EDGE_SIZE=3
        self.IN_SIZE=9
        self.OUT_SIZE=10
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.METRICS=["LOSS","AVERAGE_PRECISION",]
        self.REGUL_LOSS_BETA=0.0
        self.criterion=nn.BCEWithLogitsLoss()

class PeptidesStructuralConfig(Config):
    def __init__(self,):
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=200
        self.HIDDEN_LAYERS=3
        self.EDGE_SIZE=3
        self.IN_SIZE=9
        self.OUT_SIZE=11
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","MAE"]
        self.REGUL_LOSS_BETA=0.1
        self.criterion=F.mse_loss
        #self.default_activation=nn.ELU
        self.UP_DOWN_MODULES=0
        self.ATTENTION=False
        self.RANDOM_ROOT=True
        self.MEAN_AGGR=False
        self.REPEAT_COUNT=30
        self.ADD_PARENTS=False
        self.BATCH_SIZE=512
        self.DATASET="pepstruct"

class ZincConfig(Config):
    def __init__(self,):
        super().__init__()
        self.ATOM_ENCODER=False
        self.NODE_ENCODER_NUM_TYPES=28
        self.EDGE_ENCODER_NUM_TYPES=4
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.EDGE_SIZE=64
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","MAE"]
        self.REGUL_LOSS_BETA=15.0
        self.criterion=F.mse_loss
        self.default_activation=nn.ELU
        self.SCHEDULER_LAMBDA = lambda epoch: (1 if epoch < 25 else (0.1 if epoch < 100 else 0.01))

class AqsolConfig(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.EDGE_SIZE=64
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","MAE"]
        self.REGUL_LOSS_BETA=0.1
        self.criterion=F.mse_loss
        self.default_activation=nn.ELU
        self.UP_DOWN_MODULES=2
        self.ATTENTION=False
        self.RANDOM_ROOT=True
        self.MEAN_AGGR=True
        self.REPEAT_COUNT=30
        self.ADD_PARENTS=True
        self.DATASET="aqsol"

class EsolConfig(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.EDGE_SIZE=64
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","MAE"]
        self.REGUL_LOSS_BETA=100.0
        self.criterion=F.mse_loss
        self.default_activation=nn.ELU
        self.UP_DOWN_MODULES=2
        self.ATTENTION=False
        self.RANDOM_ROOT=True
        self.MEAN_AGGR=True
        self.REPEAT_COUNT=30
        self.ADD_PARENTS=True
        self.DATASET="esol"

class MolhivConfig(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=3
        self.EDGE_SIZE=3
        self.IN_SIZE=-1
        self.OUT_SIZE=1
        self.MOLECULES=True
        self.BINARY_OUTPUT=True
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","MAE","AUCM"]
        self.REGUL_LOSS_BETA=100.0
        self.criterion=F.mse_loss
        #self.default_activation=nn.ELU
        self.UP_DOWN_MODULES=0
        self.ATTENTION=False
        self.RANDOM_ROOT=True
        self.MEAN_AGGR=False
        self.REPEAT_COUNT=30
        self.ADD_PARENTS=False
        self.DATASET="molhiv"

class EdgeTestConfig(Config):
    def __init__(self,):
        super(Config,self).__init__()
        self.HIDDEN_SIZE=64
        self.EDGE_SIZE=1
        self.IN_SIZE=3
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=2
        self.METRICS=["LOSS","AVERAGE_PRECISION",]
        self.criterion=nn.BCEWithLogitsLoss()
        
class MinimumVertexCoverConfig(Config):
    def __init__(self,):
        super().__init__()
        self.IN_SIZE = 1


class TreeTask1Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask1"
        self.METRICS=["LOSS","MAE","INT_PRECISION"]
        self.PATIENCE=50
        self.BATCH_SIZE=1024

class TreeTask2Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask2"
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]
        self.BINARY_OUTPUT=True
        self.LR=1e-4
        #self.criterion=nn.BCELoss()

class TreeTask3Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask3"
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]

class TreeTask4Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask4"
        self.NODE_READOUT = True
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]
        self.BINARY_OUTPUT=True
        self.criterion=nn.BCELoss()

class TreeTask5Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask5"
        self.NODE_READOUT = True
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_INT_PRECISION", "BIN_MSE_LOSS"]
        self.BINARY_OUTPUT=True
        self.criterion=nn.BCELoss()
        self.PATIENCE=50


class TreeTask6Config(Config):
    def __init__(self,):
        super().__init__()
        self.IN_SIZE=1
        self.DATASET="treetask6"
        self.METRICS=["LOSS","MAE","INT_PRECISION"]

class TreeTask7Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask7"
        self.IN_SIZE=2
        self.NODE_READOUT = True
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_INT_PRECISION", "BIN_MSE_LOSS"]
        self.BINARY_OUTPUT=True
        self.criterion=nn.BCELoss()

class TreeTask8Config(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask8"
        self.IN_SIZE=3
        #self.NODE_READOUT = True
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_INT_PRECISION", "BIN_MSE_LOSS"]
        self.BINARY_OUTPUT=True
        #self.criterion=nn.BCELoss()

class TreeTask4ConfigGCMN(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask4"
        self.NODE_READOUT = True
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]
        self.BINARY_OUTPUT=True
        #self.criterion=nn.BCELoss()
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.DATASET_TRAIN_SIZE = 304
        self.DATASET_TRAIN_MIN_N = 2
        self.DATASET_TRAIN_MAX_N = 20
        self.DATASET_VALID_SIZE = 304
        self.DATASET_VALID_MIN_N = 2
        self.DATASET_VALID_MAX_N = 200
        self.DATASET_TESTS = [(101, 2, 1000)]
        self.NUM_WORKERS = 0
        self.REGUL_LOSS_BETA=None
        self.PATIENCE=10
        self.UP_DOWN_MODULES=1
        self.BATCH_SIZE=32
        self.EVAL_BATCH_SIZE=1
        self.TEST_EVAL_FREQ=10

class TreeTask8ConfigGCMN(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask8"
        self.NODE_READOUT = False
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.BINARY_OUTPUT=True
        #self.criterion=nn.BCELoss()
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.DATASET_TRAIN_SIZE = 10004
        self.DATASET_TRAIN_MIN_N = 2
        self.DATASET_TRAIN_MAX_N = 20
        self.DATASET_VALID_SIZE = 304
        self.DATASET_VALID_MIN_N = 2
        self.DATASET_VALID_MAX_N = 200
        self.DATASET_TESTS = [(101, 2, 1000)]
        self.NUM_WORKERS = 0
        self.REGUL_LOSS_BETA=None
        self.PATIENCE=10
        self.UP_DOWN_MODULES=1
        self.BATCH_SIZE=32
        self.EVAL_BATCH_SIZE=1
        self.TEST_EVAL_FREQ=1
        self.INIT_EDGE_SIZE=0

class TreeTask2ConfigGCMN(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="treetask2"
        self.NODE_READOUT = False
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.BINARY_OUTPUT=True
        #self.criterion=nn.BCELoss()
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.DATASET_TRAIN_SIZE = 10004
        self.DATASET_TRAIN_MIN_N = 2
        self.DATASET_TRAIN_MAX_N = 20
        self.DATASET_VALID_SIZE = 304
        self.DATASET_VALID_MIN_N = 2
        self.DATASET_VALID_MAX_N = 200
        self.DATASET_TESTS = [(101, 2, 1000)]
        self.NUM_WORKERS = 0
        self.REGUL_LOSS_BETA=None
        self.PATIENCE=10
        self.UP_DOWN_MODULES=1
        self.BATCH_SIZE=32
        self.EVAL_BATCH_SIZE=1
        self.TEST_EVAL_FREQ=1
        self.INIT_EDGE_SIZE=0
        self.FINAL_AGGR=aggr.MaxAggregation

class AqsolConfigGCMN(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.EDGE_SIZE=64
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=0 #2
        self.METRICS=["LOSS","MAE","INT_PRECISION"]
        self.criterion=F.mse_loss
        self.UP_DOWN_MODULES=2
        self.NODE_READOUT = False
        self.REGUL_LOSS_BETA=None
        #self.MEAN_AGGR=True
        self.DATASET="aqsol"
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.PATIENCE=20

class EsolGCMNConfig(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=128
        self.HIDDEN_LAYERS=4
        self.EDGE_SIZE=128
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=0
        self.METRICS=["LOSS","MAE","INT_PRECISION"]

        self.criterion=F.mse_loss
        self.UP_DOWN_MODULES=4
        self.NODE_READOUT = False
        self.REGUL_LOSS_BETA=None
        #self.MEAN_AGGR=True
        self.DATASET="esol"
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.PATIENCE=20
        self.LR=1e-3

class ZincConfigGCMN(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=False
        self.BOND_ENCODER=False
        self.NODE_ENCODER_NUM_TYPES=28
        self.EDGE_ENCODER_NUM_TYPES=4
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=2
        self.EDGE_SIZE=64
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=0 #2
        self.METRICS=["LOSS","MAE","INT_PRECISION"]
        self.criterion=F.mse_loss
        self.UP_DOWN_MODULES=3
        self.NODE_READOUT = False
        self.REGUL_LOSS_BETA=None
        #self.MEAN_AGGR=True
        self.DATASET="zinc"
        self.MODEL="GCMN"
        self.GCMN_DEPTH=5
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.PATIENCE=25
        self.LR=1e-3

class MolhivConfigGCMN(Config):
    def __init__(self,): 
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=3
        self.EDGE_SIZE=64
        self.IN_SIZE=1
        self.OUT_SIZE=1
        self.MOLECULES=True
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=0 #2
        self.METRICS=["AUCM","LOSS","MAE","INT_PRECISION"]
        self.criterion=F.mse_loss
        self.UP_DOWN_MODULES=2
        self.NODE_READOUT = False
        self.REGUL_LOSS_BETA=None
        #self.MEAN_AGGR=True
        self.DATASET="molhiv"
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.PATIENCE=20
        self.GCMN_FREQ=3

class PepStructGCMNConfig(Config):
    def __init__(self,):
        super().__init__()
        self.ATOM_ENCODER=True
        self.BOND_ENCODER=True
        self.NODE_ENCODER_NUM_TYPES=None
        self.EDGE_ENCODER_NUM_TYPES=None
        self.HIDDEN_SIZE=64
        self.HIDDEN_LAYERS=3
        self.EDGE_SIZE=64
        self.IN_SIZE=9
        self.OUT_SIZE=11
        self.MOLECULES=False
        self.BINARY_OUTPUT=False
        self.NUM_WORKERS=0
        self.METRICS=["LOSS","MAE","INT_PRECISION"]
        self.criterion=F.mse_loss
        self.UP_DOWN_MODULES=1
        self.REGUL_LOSS_BETA=None
        self.DATASET="pepstruct"
        self.MODEL="GCMN"
        self.GCMN_DEPTH=7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.PATIENCE=20
        self.LR=1e-3
        self.GCMN_FREQ=20

class GraphTaskConfigGCMN(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="gt7"
        self.NODE_READOUT = False
        self.METRICS=["LOSS","MAE","INT_PRECISION", "BIN_MSE_LOSS"]
        self.HIDDEN_SIZE=64 #128
        self.HIDDEN_LAYERS=2
        self.BINARY_OUTPUT=True
        #self.criterion=nn.BCELoss()
        self.MODEL="GCMN"
        self.IN_SIZE=5
        self.GCMN_DEPTH=5 # 7
        self.HLD_Transform = False
        self.GCMN_Transform = True
        self.DATASET_TRAIN_SIZE = 20004 #10004
        self.DATASET_TRAIN_MIN_N = 2
        self.DATASET_TRAIN_MAX_N = 50 #100
        self.DATASET_VALID_SIZE = 504
        self.DATASET_VALID_MIN_N = 2
        self.DATASET_VALID_MAX_N = 200
        self.DATASET_TESTS = [(273, 2, 2000)]
        self.NUM_WORKERS = 0
        self.REGUL_LOSS_BETA=None
        self.PATIENCE=10
        self.UP_DOWN_MODULES=1
        self.BATCH_SIZE=32
        self.EVAL_BATCH_SIZE=1
        self.TEST_EVAL_FREQ=1
        self.INIT_EDGE_SIZE=0
        self.FINAL_AGGR=aggr.SumAggregation
        
        

class MUTAGConfig(Config):
    def __init__(self,):
        super().__init__()
        self.DATASET="mutag"
        self.METRICS=["LOSS","MAE","INT_PRECISION"]

class MPNNConfig(Config):
    def __init__(self,):
        super().__init__()
        self.MODEL = "MPNN"
        self.HLD_Transform = False

config = DefaultConfig()

