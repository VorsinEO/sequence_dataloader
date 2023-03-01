from torch.utils.data import DataLoader
from collections import defaultdict
from typing import Dict, Optional, List, Union, Tuple
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from random import shuffle
import webdataset
from functools import partial
from .feature_processors import (
    FeatureProcessorCombiner,
    FeatureProcessorConst,
    FeatureProcessorCallable,
)
from .dataset import get_dataset


def prepr_id(x):
    res = int(str(x)[:10].replace("-", ""))
    return res


def rev_prepr_id(k):
    k = str(k)
    res = k[:4] + "-" + k[4:6] + "-01"
    return res


def get_mask(list_lengths: list) -> torch.tensor:
    mask = [[0] * i + [1] * (max(list_lengths) - i) for i in list_lengths]
    return torch.tensor(mask).bool()


class inf_loop:
    def __init__(self, dataset: DataLoader, length: int = 1_000) -> None:
        self.dataset = dataset
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __iter__(self) -> Dict[str, torch.Tensor]:
        yield from self.dataset()


def prepare_loader(
    dataset: webdataset.dataset.Processor,
    num_workers: int = 0,
    batch_size: Optional[int] = None,
    pin_memory: bool = True,
) -> DataLoader:
    loader = DataLoader(
        dataset=dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=pin_memory,
    )

    return loader


class SequenceCollator:
    def __init__(
        self,
        feature_names: List[str],
        continious_features: Optional[List[str]] = None,
        default_values: Optional[dict] = None,
        count_len_by: Optional[str] = None,
        resample_n: Optional[int] = None,
        batch_size: Optional[int] = None,
        fasttext_emb: Optional[np.array] = None,
        resample_col: str = "target",
    ) -> None:
        self.feature_names = feature_names
        self.continious_features = continious_features or []
        self.default_values = default_values or {}
        self.count_len_by = count_len_by
        self.resample_n = resample_n
        self.resample_col = resample_col
        self.batch_size = batch_size
        self.fasttext_emb = fasttext_emb

        self.default_values = defaultdict(lambda: 0, self.default_values)

    def __call__(
        self, samples: Dict[str, Union[list, np.ndarray]]
    ) -> Dict[str, torch.Tensor]:
        features: Dict[str, torch.Tensor] = defaultdict(list)

        if self.resample_n is not None and self.resample_col is not None:
            undersample_indices = self._get_indices_undersample(samples)
            samples = [samples[i] for i in undersample_indices]

        for sample in samples:
            if self.count_len_by is not None:
                features["lengths"].append(len(sample[self.count_len_by]))
            for key in self.feature_names:
                features[key].append(torch.as_tensor(sample[key]))

        for key in self.feature_names:
            features[key] = pad_sequence(
                features[key], padding_value=self.default_values[key]
            ).T
            if key in self.continious_features:
                features[key] = features[key].type(torch.FloatTensor)

        if self.fasttext_emb is not None:
            features = self._add_item_embedding(features)

        if self.count_len_by is not None:
            features["mask"] = get_mask(features["lengths"])

        return features

    def _add_item_embedding(self, features):
        embeddings = torch.from_numpy(self.fasttext_emb[features["item_id"]])
        features["fasttesx_embeddings"] = embeddings
        return features

    def _get_indices_undersample(
        self, samples: Dict[str, Union[list, np.ndarray]]
    ) -> List[int]:
        limit_true = self.resample_n // 2
        targets = np.array([sample[self.resample_col] for sample in samples])

        true_indices = np.where(targets == 1)[0][:limit_true]
        false_indices = np.where(targets == 0)[0]
        count_for_choice = self.resample_n - len(true_indices)
        if count_for_choice > len(false_indices):
            count_for_choice = len(false_indices)
        false_indices = np.random.choice(false_indices, count_for_choice, replace=False)
        undersample_indices = np.hstack([true_indices, false_indices]).tolist()
        shuffle(undersample_indices)

        return undersample_indices


def get_dataloader(
    path_to_data,
    params,
    batch_size=None,
    num_workers=None,
    fasttext_emb=None,
    shuffle=False,
    with_target=True,
    resample_n=None,
):
    if batch_size is None:
        batch_size = params.batch_size
    if num_workers is None:
        num_workers = params.num_workers

    if with_target:
        const_feats = ["user_id", "user_tr_count", params.target_col]
    else:
        const_feats = [
            "user_id",
            "user_tr_count",
        ]
    standart_processors = (
        FeatureProcessorConst(const_feats),
        FeatureProcessorCallable(["score_dt_border"], prepr_id),
    )

    test_feature_processor = FeatureProcessorCombiner(
        [*standart_processors], max_seq_len=params.seq_len
    )

    feature_names = ["item_id"] + list(
        set(
            test_feature_processor.additional_column_names
            + params.cont_features
            + list(params.emb_params.keys())
        )
    )
    fasttext_cols = [f"fasttext_{i}" for i in range(256)]
    feature_names = [col for col in feature_names if col not in fasttext_cols]
    cont_feats = [col for col in params.cont_features if col not in fasttext_cols]

    resample_col = params.__dict__.get("target_col", "target")

    collator_down = SequenceCollator(
        feature_names=feature_names,
        continious_features=cont_feats,
        count_len_by="item_id",
        resample_n=resample_n,
        resample_col=resample_col,
        fasttext_emb=fasttext_emb,
    )

    test_down = get_dataset(
        path_to_data,
        test_feature_processor,
        collator_down,
        batch_size,
        shuffle,
        hdfs=False,
    )

    test_down_loader = inf_loop(
        partial(prepare_loader, test_down, num_workers=num_workers)
    )

    return test_down_loader
