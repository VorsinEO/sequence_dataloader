import pickle
import random, os
import numpy as np
import torch


def seed_everything(seed: int):

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class Args(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "<{cl_name}> \n @{id:x} {attrs}".format(
            cl_name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )


base_conf = {
    "emb_params": {"feat": {"vocabulary_size": 5, "dim": 2}},
    "cont_features": ["feat_2"],
    "rnn": {
        "type": "GRU",
        "hidden_size": 128,
        "rnn_num_layers": 2,
        "rnn_bias": True,
        "bidirectional": True,
    },
    "ffn": {"hid_linear": 128},
    "seq_len": 256,
    "batch_size": 128,
    "num_workers": 0,
    "device": "cuda:0",
    "lr": 0.001,
    "weight_decay": 0.01,
    "l1_weight": 0,
    "step_size": 2,
    "gamma": 0.5,
    "fp16": False,
    "clip_value": 1.5,
    "batch_accum": 1,
    "smooth_factor": None,
    "early_stop": {"more_best": True, "patience": 5},
}


def save_pickle(path="filename.pickle", your_data=None):
    with open(path, "wb") as handle:
        pickle.dump(your_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path="filename.pickle"):
    with open(path, "rb") as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data
