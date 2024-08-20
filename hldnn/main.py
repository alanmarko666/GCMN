from config import config as global_config
from datasets.get_dataset import get_dataset
import models.hldmodel as hldmodel
import models.gcmnmodel as gcmnmodel
import models.baselines as baselines
import pipeline.pipeline as pipeline
import torch.nn.functional as F
import datasets.line_dataset as line_dataset
from config import DefaultConfig
import wandb
import os
import torch
import config


def run(config = {}):
    global_config.set_config(DefaultConfig())
    global_config.set_config(config)
    print(global_config)
    D = get_dataset()
    if global_config.MODEL == "hldgnn":
        network=hldmodel.HLDGNNModel(node_readout_module=global_config.NODE_READOUT)
    elif global_config.MODEL == "MPNN":
        network=baselines.MPNNModel(node_readout_module=global_config.NODE_READOUT)
    elif global_config.MODEL == "GCMN":
        network=gcmnmodel.GCMNModel(node_readout_module=global_config.NODE_READOUT) 

    torch.autograd.set_detect_anomaly(True)
    p= pipeline.Pipeline(network=network,
                train_dataset=D["train"],
                test_datasets=D["test"],
                validation_dataset=D["valid"],
                batch_size=global_config.BATCH_SIZE,
                lr=global_config.LR)
    p.train()
    return (D, network, p)

def reevaluate_run(config, run="hldnn/hldnn/rju12q9b", model_name="model-99.h5"):
    os.environ["WANDB_API_KEY"] = "8523288184de2d3d62b4ab6590c0cf60b918da8c"

    api = wandb.Api()
    run = api.run(run)

    global_config.set_config(DefaultConfig())
    #global_config.set_config(run.config)
    run.file(model_name).download(replace=True)
    global_config.set_config(config)
    D = get_dataset()
    network=hldmodel.HLDGNNModel(node_readout_module=global_config.NODE_READOUT)
    network.load_state_dict(torch.load(model_name))
    p= pipeline.Pipeline(network=network,
                train_dataset=D["train"],
                test_datasets=D["test"],
                validation_dataset=D["valid"],
                batch_size=config.BATCH_SIZE,
                lr=1e-3)
    p.evaluate(epoch=0)
    return (D, network, p)

def unit_test():
    
    conifgs = [
        config.TreeTask1Config(),
        config.TreeTask2Config(),
        config.TreeTask3Config(),
        config.TreeTask4Config(),
        config.TreeTask5Config(),
        config.TreeTask6Config(),
        config.TreeTask7Config(),
    ]

    
    
    for model, hld in [("hldgnn", True), ("MPNN",False)]:
        for c in conifgs:
            c.MODEL = model
            c.HLD_Transform = hld
            c.TEST_SMALL_DATASET = 50
            c.epoch_num = 2
            c.wandb=False
            print(c)
            run(c)
    

from config import TreeTask1Config, TreeTask2Config, TreeTask3Config,TreeTask4Config,TreeTask5Config,TreeTask6Config,TreeTask7Config, MPNNConfig, MolhivConfig, AqsolConfig
import sys, itertools
def final_runs(c):
    combs = []
    def combiner(lis):
        return [{k: v for z in l for k, v in z.items()} for l in itertools.product(*lis)]
    #combs += combiner([[MPNNConfig(),{}],
    #[TreeTask1Config(), TreeTask2Config(), TreeTask3Config(),TreeTask4Config(),TreeTask5Config(),TreeTask6Config(),TreeTask7Config()]])
    #combs += combiner([[MPNNConfig(),{}],
    #            [MolhivConfig(), AqsolConfig()]])
    combs += combiner([[MPNNConfig(),{}],
    [TreeTask2Config(),TreeTask8Config()]])
    combs = combiner([[{"SEED": i } for i in range(5)],combs])
    print(len(combs))
    print(combs)
    run(combs[c])
    sys.exit(0)

if __name__ == '__main__':
    if len(sys.argv) >=2 and sys.argv[1] == "--hyperparm":
        run(config={})
        sys.exit(0)
    elif len(sys.argv) >= 2 and sys.argv[1] == "--finalruns":
        final_runs(int(sys.argv[2]))
        sys.exit(0)