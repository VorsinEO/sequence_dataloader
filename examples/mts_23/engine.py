from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# from typing import Dict, Optional, List, Union, Tuple, Type
import torch

# from torch import nn
import json
import pandas as pd
import numpy as np

# from .criterion import loss_fn, SoftBCEWithLogitsLoss, FocalLoss
# from .model import RNNEncoder

# from .dataloader import inf_loop

from sklearn.metrics import f1_score, roc_auc_score


def get_metrics(true_sex, pred_sex, true_age, pred_age):
    sex_gini = (2 * roc_auc_score(true_sex, pred_sex)) - 1
    age_f1 = f1_score(true_age, pred_age, average="weighted")
    val_score = 2 * age_f1 + sex_gini
    return sex_gini, age_f1, val_score


def train_fn(
    model,
    dataloader,
    optimizer,
    scaler,
    params,
    device,
    total=500,
    scheduler=None,
):
    train_loss, train_loss_bi, train_loss_mc = [], [], []
    crit_bi = torch.nn.BCEWithLogitsLoss()
    crit_mc = torch.nn.CrossEntropyLoss()

    model.train()
    # target_col = params.__dict__.get("target_col", "target")

    for i, batch in tqdm(enumerate(dataloader), total=total):

        y_bi = (
            batch[params.target_col_bin]
            .flatten()
            .float()
            .to(device, non_blocking=(params.device != "cpu"))
        )
        y_mc = (
            batch[params.target_col_multyclass]
            .flatten()
            .long()
            .to(device, non_blocking=(params.device != "cpu"))
        )

        for feat in params.cont_features + params.cont_features_const:
            batch[feat] = batch[feat].to(device, non_blocking=(params.device != "cpu"))
        # if params.__dict__.get("fasttesx_embeddings", False):
        #    batch["fasttesx_embeddings"] = batch["fasttesx_embeddings"].to(
        #        device, non_blocking =(params.device!="cpu")
        #    )

        for feat in list(params.emb_params.keys()) + list(
            params.emb_params_const.keys()
        ):
            batch[feat] = (
                batch[feat].to(device, non_blocking=(params.device != "cpu")).long()
            )
        if batch.get("mask") is not None:
            batch["mask"] = batch["mask"].to(
                device, non_blocking=(params.device != "cpu")
            )

        with torch.cuda.amp.autocast(enabled=params.fp16):
            logit_bi, logit_mc = model(batch)
            loss_bi = crit_bi(logit_bi.squeeze(), y_bi)
            loss_mc = crit_mc(logit_mc, y_mc)
            loss = loss_bi + loss_mc

            loss = loss / params.batch_accum

            scaler.scale(loss).backward()

        if (i + 1) % params.batch_accum == 0:
            if params.clip_value > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), params.clip_value)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=False)
            if scheduler is not None:
                scheduler.step()
        train_loss.append((loss.cpu().detach().item()) * params.batch_accum)
        train_loss_bi.append((loss_bi.cpu().detach().item()) * params.batch_accum)
        train_loss_mc.append((loss_mc.cpu().detach().item()) * params.batch_accum)

    return np.mean(train_loss), np.mean(train_loss_bi), np.mean(train_loss_mc)


def val_fn(model, dataloader, params, device, total=500):
    targets_bi = []
    targets_mc = []
    preds_bi_all = []
    preds_mc_all = []
    logits_mc = []

    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=total):

            y_bi = batch[params.target_col_bin].flatten().float().to(device)

            y_mc = batch[params.target_col_multyclass].flatten().long().to(device)

            for feat in params.cont_features + params.cont_features_const:
                batch[feat] = batch[feat].to(device)
            # if params.__dict__.get("fasttesx_embeddings", False):
            #    batch["fasttesx_embeddings"] = batch["fasttesx_embeddings"].to(
            #        device
            #    )
            for feat in list(params.emb_params.keys()) + list(
                params.emb_params_const.keys()
            ):
                batch[feat] = batch[feat].to(device).long()
            if batch.get("mask") is not None:
                batch["mask"] = batch["mask"].to(device)
            logit_bi, logit_mc = model(batch)
            logit_bi = logit_bi.squeeze()
            preds_bi = sigmoid(logit_bi).tolist()
            preds_bi_all.extend(preds_bi)

            preds_mc = softmax(logit_mc).argmax(dim=1).tolist()
            preds_mc_all.extend(preds_mc)
            logits_mc.extend(logit_mc.tolist())

            targets_bi.extend(y_bi.tolist())
            targets_mc.extend(y_mc.tolist())

    gini, f1, score = get_metrics(targets_bi, preds_bi_all, targets_mc, preds_mc_all)

    return score, gini, f1


def prepr_id(x):
    res = int(str(x)[:10].replace("-", ""))
    return res


def rev_prepr_id(k):
    k = str(k)
    res = k[:4] + "-" + k[4:6] + "-01"
    return res


def inf_fn(model, dataloader, params, device, total=500, with_target=False):
    targets_bi = []
    targets_mc = []
    preds_bi_all = []
    preds_mc_all = []
    logits_mc = []
    # debug
    stats_id = []
    stats_tr_cnt = []
    model.eval()
    softmax = torch.nn.Softmax(dim=1)
    sigmoid = torch.nn.Sigmoid()

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader), total=total):
            # debug
            stats_id.extend(batch["id"].flatten().tolist())
            stats_tr_cnt.extend(batch["user_tr_count"].flatten().tolist())
            if with_target:
                targets_bi.extend(batch[params.target_col_bin].flatten().tolist())
                targets_mc.extend(
                    batch[params.target_col_multyclass].flatten().tolist()
                )

            for feat in params.cont_features + params.cont_features_const:
                batch[feat] = batch[feat].to(device)
            # if params.__dict__.get("fasttesx_embeddings", False):
            #    batch["fasttesx_embeddings"] = batch["fasttesx_embeddings"].to(
            #        device)
            #    )
            for feat in list(params.emb_params.keys()) + list(
                params.emb_params_const.keys()
            ):
                batch[feat] = batch[feat].to(device).long()
            if batch.get("mask") is not None:
                batch["mask"] = batch["mask"].to(device)
            logit_bi, logit_mc = model(batch)
            logit_bi = logit_bi.squeeze()
            preds_bi = sigmoid(logit_bi).tolist()
            preds_bi_all.extend(preds_bi)

            preds_mc = softmax(logit_mc).argmax(dim=1).tolist()
            preds_mc_all.extend(preds_mc)
            logits_mc.extend(logit_mc.tolist())

    res = {
        "customer_global_id": stats_id,
        "user_tr_count": stats_tr_cnt,
        "preds_bi": preds_bi_all,
        "preds_mc_all": preds_mc_all,
        "logits_mc": logits_mc,
    }
    if with_target:
        res[params.target_col_bin] = targets_bi
        res[params.target_col_multyclass] = targets_mc
    res = pd.DataFrame(res)
    res["preds_mc_all"] = res["preds_mc_all"] + 1
    return res


class EarlyStop:
    def __init__(self, path_to_save, params):
        self.path_to_save = path_to_save
        self.more_best = params.early_stop["more_best"]
        self.patience = params.early_stop["patience"]
        self.params = params
        self.save_checkpoints = params.__dict__.get("save_checkpoints", True)
        self.curr_count = 0
        self.curr_epoch = 0
        if self.more_best:
            self.best_metric = np.NINF
        else:
            self.best_metric = np.PINF

    def __call__(self, model, metric, epoch, verbose=True):
        self.curr_epoch = epoch
        if self.more_best:
            if np.round(metric, 4) >= self.best_metric:
                self.best_metric = metric
                if verbose:
                    print(
                        f"----------- EPOCH {epoch} - NEW BEST METRIC: {np.round(metric, 4)} ----------------"
                    )
                self._save_model(model, metric)
                self.curr_count = 0
            else:
                self._save_model(model, metric)
                self.curr_count = self.curr_count + 1
        else:
            if metric <= self.best_metric:
                self.best_metric = metric
                if verbose:
                    print(
                        f"----------- EPOCH {epoch} - NEW BEST METRIC: {np.round(metric, 4)} ----------------"
                    )
                self._save_model(model, metric)
                self.curr_count = 0
            else:
                self._save_model(model, metric)
                self.curr_count = self.curr_count + 1
        if self.curr_count > self.patience:
            return "STOP"

    def _save_model(self, model, metric):
        self.params.early_stop["best_metric"] = self.best_metric
        self.params.early_stop["epoch"] = self.curr_epoch

        metric = str(np.round(metric, 4)).replace(".", "_").replace("-", "_")
        if self.save_checkpoints:
            path_to_save_model = (
                self.path_to_save + f"model_ep_{self.curr_epoch}_best_{metric}.bin"
            )
            torch.save(model.state_dict(), path_to_save_model)
            self.params.early_stop["name_model_state"] = path_to_save_model

        params_to_save = self.params.__dict__
        with open(self.path_to_save + "params.json", "w") as fp:
            json.dump(params_to_save, fp)


def get_model(
    path_to_weights: str, path_to_params: str, load_net, device_name: str = "cuda:0"
) -> tuple:
    """
    Load and return model and params for inference

    Parameters:
    __________
    path_to_weights: str - path to weight of model
    path_to_params: str - path to model config
    load_net - class of model
    device_name:str - device name for load model

    Output:
    _________
    params - config of model pipeline
    model - torch model pretrained

    """
    # load params
    with open(path_to_params, "r") as fh:
        params_dict = json.load(fh)
    params = Args(**params_dict)

    # set device
    params.device = device_name

    # load model and put it to device
    model = load_net(params)
    model.load_state_dict(torch.load(path_to_weights))
    device = torch.device(params.device)
    model.to(device)

    return params, model


def average_snapshots(
    list_of_snapshots_paths, load_net, path_to_params, device_name="cuda:1"
):
    """Use model weights from different folds as one model for inference"""
    snapshots_weights = {}

    for snapshot_path in list_of_snapshots_paths:
        m_params, model = get_model(
            snapshot_path, path_to_params, load_net, device_name
        )
        snapshots_weights[snapshot_path] = dict(model.named_parameters())

    params = model.named_parameters()
    dict_params = dict(params)

    N = len(snapshots_weights)

    for name in dict_params.keys():
        custom_params = None
        for _, snapshot_params in snapshots_weights.items():
            if custom_params is None:
                custom_params = snapshot_params[name].data
            else:
                custom_params += snapshot_params[name].data
        dict_params[name].data.copy_(custom_params / N)

    model_dict = model.state_dict()
    model_dict.update(dict_params)

    model.load_state_dict(model_dict)
    model.eval()
    device = torch.device(m_params.device)
    model.to(device)

    return m_params, model


class Args(object):
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "<{cl_name}> \n @{id:x} {attrs}".format(
            cl_name=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=" ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )
