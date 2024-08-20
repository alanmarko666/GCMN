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

def create_config(conf):
    config = Config()
    config.batch_size = 10  # input batch size for training (default: 64)
    config.test_batch_size = 10  # input batch size for testing (default: 64)
    config.limit_dataset = -1
    config.ignore_labels = False
    config.ignore_classified_labels = False
    config.epochs = 50  # number of epochs to train (default: 100)
    config.lr = 1e-3  # learning rate (default: 0.01)
    config.seed = 42
    config.device = "cuda"
    config.num_hid_neurons = 128  # num hid neurons
    config.knn_neighbours = 20
    config.dataset = "bpq"
    config.lr_reduce_factor = 0.5
    config.lr_schedule_patience = 5
    config.weight_decay = 1e-6
    config.min_lr = 1e-6
    config.num_layers = 3
    config.edge_dim = 111
    config.num_scales = 3
    config.debug = False
    config.num_workers = 0 if config.debug else 2
    config.downscale = 5
    config.batch_norm = True
    config.wandb = True
    config.time_limit = 30 * 60
    config.pointnet_include_features = True
    config.k_fold = 0
    config.gated = False
    config.residual = False
    config.multi_scale_subsampler = []

    config.HIDDEN_SIZE=64
    config.IN_SIZE=3
    config.OUT_SIZE=1


    for key,val in conf.items():
        config[key] = val

    return config