from .base import BaseModel

# import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from typing import Dict, Optional, List, Union


def get_input_size(params) -> int:
    """
    Parameters:
    __________
    params: Args -  config of model and pipeline in general

    Output:
    __________
    input_size: int - its value for embedding size per event in sequense
    """
    input_size = sum(
        [params.emb_params[feat]["dim"] for feat in params.emb_params.keys()]
    ) + len(params.cont_features)

    input_size = input_size + params.__dict__.get("text_embeddings", False) * 768

    return input_size


class RNNClsReceipts(BaseModel):
    def __init__(self, params):
        super().__init__()
        self.params = params
        # d_out = params.__dict__.get("d_out", 1)
        d_out_bi = 1
        d_out_multy = 6

        self.input_size = get_input_size(params)

        self.embeds = nn.ModuleList(
            [
                nn.Embedding(
                    params.emb_params[feat]["vocabulary_size"] + 2,
                    params.emb_params[feat]["dim"],
                )
                for feat in params.emb_params.keys()
            ]
        )
        self.embeds_const = nn.ModuleList(
            [
                nn.Embedding(
                    params.emb_params_const[feat]["vocabulary_size"] + 2,
                    params.emb_params_const[feat]["dim"],
                )
                for feat in params.emb_params_const.keys()
            ]
        )
        # RNN
        self.rnn = self._get_rnn()

        # FFN classifier
        hidden_input = (
            (params.rnn["bidirectional"] * 1 + 1) * 2 * params.rnn["hidden_size"]
            + sum(
                [
                    params.emb_params_const[feat]["dim"]
                    for feat in params.emb_params_const.keys()
                ]
            )
            + len(params.cont_features_const)
        )

        self.ffn_head = nn.Sequential(
            nn.Linear(hidden_input, params.ffn["hid_linear"]),
            nn.ReLU(),
            nn.Linear(params.ffn["hid_linear"], params.ffn["hid_linear"] // 2),
            nn.ReLU(),
            nn.Linear(params.ffn["hid_linear"] // 2, d_out_bi),
        )
        # FFN classifier multy

        self.ffn_head_multy = nn.Sequential(
            nn.Linear(hidden_input, params.ffn["hid_linear"]),
            nn.ReLU(),
            nn.Linear(params.ffn["hid_linear"], params.ffn["hid_linear"] // 2),
            nn.ReLU(),
            nn.Linear(params.ffn["hid_linear"] // 2, d_out_multy),
        )
        # DROPOUT and Normalization
        self.spatial_dropout = nn.Dropout2d(params.spatial_dropout)
        self.layer_norm = nn.LayerNorm(self.input_size)
        self.batch_norm = nn.BatchNorm1d(hidden_input)
        self.dropout = nn.Dropout(params.dropout)

    def forward(self, batch):
        params = self.params
        if len(params.cont_features) > 0:
            cont_feats = [batch[feat].unsqueeze(-1) for feat in params.cont_features]

        embedded = [
            self.embeds[i](batch[feat])
            for i, feat in enumerate(params.emb_params.keys())
        ]
        if len(params.cont_features_const) > 0:
            cont_feats_const = [batch[feat] for feat in params.cont_features_const]

        embedded_const = [
            self.embeds_const[i](batch[feat]).squeeze()
            for i, feat in enumerate(params.emb_params_const.keys())
        ]

        # if params.__dict__.get("fasttesx_embeddings", False):
        #    cont_feats.append(batch["fasttesx_embeddings"])
        if len(params.cont_features) > 0:
            stacked = torch.cat(cont_feats + embedded, dim=-1)
        else:
            stacked = torch.cat(embedded, dim=-1)
        if len(params.cont_features_const) > 0:
            stacked_const = torch.cat(cont_feats_const + embedded_const, dim=-1)
        else:
            stacked_const = torch.cat(embedded_const, dim=-1)

        stacked = self.layer_norm(stacked)

        stacked = stacked.permute(0, 2, 1).unsqueeze(3)
        stacked = self.spatial_dropout(stacked)
        stacked = stacked.squeeze(3).permute(0, 2, 1)

        stacked = pack_padded_sequence(
            stacked, lengths=batch["lengths"], batch_first=True, enforce_sorted=False
        )

        rnn_d, _ = self.rnn(stacked)
        rnn_d, _ = pad_packed_sequence(
            rnn_d, batch_first=True, total_length=params.seq_len
        )

        max_pool = rnn_d.max(1)[0]
        # avg_pool = rnn_d.sum(dim=1) / rnn_d.shape[1]
        den = (
            torch.tensor(batch["lengths"]).unsqueeze(-1).to(torch.device(params.device))
        )
        avg_pool = rnn_d.sum(dim=1) / den
        # rnn_out = torch.cat([max_pool, avg_pool], dim=-1)
        out_cat = torch.cat([max_pool, avg_pool, stacked_const], dim=-1)
        # out_cat = torch.cat([avg_pool, stacked_const], dim=-1)
        out_cat = self.batch_norm(out_cat)
        out_cat = self.dropout(out_cat)

        logit_bi = self.ffn_head(out_cat)
        logit_multy = self.ffn_head_multy(out_cat)
        return logit_bi, logit_multy

    def _get_rnn(self):
        params = self.params

        if params.rnn["type"] == "LSTM":
            rnn = nn.LSTM(
                self.input_size,
                hidden_size=params.rnn["hidden_size"],
                num_layers=params.rnn["rnn_num_layers"],
                bias=params.rnn["rnn_bias"],
                batch_first=True,
                bidirectional=params.rnn["bidirectional"],
            )

        elif params.rnn["type"] == "GRU":
            rnn = nn.GRU(
                self.input_size,
                hidden_size=params.rnn["hidden_size"],
                num_layers=params.rnn["rnn_num_layers"],
                bias=params.rnn["rnn_bias"],
                batch_first=True,
                bidirectional=params.rnn["bidirectional"],
            )
        else:
            print(params.rnn["type"])
            raise NotImplementedError
        return rnn
