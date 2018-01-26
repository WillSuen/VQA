import numpy as np
from easydict import EasyDict as edict
import yaml

cfg = edict()
cfg.batch_size = 64
cfg.gpus = '0'
cfg.frequent = 100
cfg.kv_store = 'device'
cfg.memonger = False
cfg.retrain = False
cfg.model_load_epoch = 0
cfg.num_epoch = 300
cfg.model_path = "./model/"

# network
cfg.network = edict()
cfg.network.rnn_type = 'gru'
cfg.network.dropout = 0.0                    # good for training
cfg.network.gru_dropout = 0.25               # dropout for gru
cfg.network.hidden_size = 512
cfg.network.vocab_size = 15031


# Train
cfg.train = edict()
cfg.train.bn_mom = 0.9
cfg.train.lr = 0.1
cfg.train.mom = 0.9
cfg.train.wd = 0.0001
cfg.train.workspace = 512
cfg.train.lr_factor_epoch = 13   # decay learning rate after # of epoch
cfg.train.lr_factor = 0.25    # learning rate decay ratio
cfg.train.beta1 = 0.9
cfg.train.beta2 = 0.999
cfg.train.uni_mag = 1
cfg.train.skip_thought_dict = './vqa-sva/rundata/skip-argdict.pkl'

# dataset
cfg.dataset = edict()
cfg.dataset.img_feature_path = './'
cfg.dataset.qa_path = './'
cfg.dataset.sent_gru_hsize = 2400
cfg.dataset.w = 14 
cfg.dataset.h = 14
cfg.dataset.seed = 1234
cfg.dataset.max_seq_length = 25


def read_cfg(cfg_file):
    with open(cfg_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in cfg:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in cfg[k]:
                            cfg[k][vk] = vv
                        else:
                            raise ValueError("key {} not exist in config.py".format(vk))
                else:
                    cfg[k] = v
            else:
                raise ValueError("key {} exist in config.py".format(k))