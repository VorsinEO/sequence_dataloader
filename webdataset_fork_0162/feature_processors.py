from abc import abstractmethod, ABC
from typing import Generator, Callable, List, Dict, Tuple
from functools import partial
from itertools import product
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime
import math


class FeatureProcessorBase(ABC):
    additional_column_names = list()

    @abstractmethod
    def __call__(self, sample: Dict):
        pass


class FeatureProcessorCallable(FeatureProcessorBase):
    def __init__(self, feature_names: List[str], preproc_func: Callable):
        self.feature_names = feature_names
        self.preproc_func = preproc_func

    def __call__(self, sample):
        for feature_name in self.feature_names:

            sample[feature_name] = np.asarray(
                [self.preproc_func(x) for x in sample[feature_name]]
            )

        return sample


class FeatureProcessorConst(FeatureProcessorBase):
    def __init__(self, feature_names: List[str]):
        self.additional_column_names = [*feature_names]
        self.feature_names = feature_names

    def __call__(self, sample):
        for key in self.feature_names:

            if type(sample[key]) in (list, np.ndarray):
                sample[key] = np.asarray(sample[key][:1])
            else:
                sample[key] = np.asarray([sample[key]])
        return sample


def check_for_zero_len(sample):
    for val in sample.values():
        if hasattr(val, "__len__"):
            return len(val) == 0


class FeatureProcessorCombiner:
    def __init__(
        self,
        feature_processors: List[FeatureProcessorBase],
        filter_zero_length: bool = True,
        max_seq_len: int = 24,
    ) -> None:
        self.feature_processors = feature_processors
        self.additional_column_names = list(
            set(
                sum(
                    [proc.additional_column_names for proc in self.feature_processors],
                    [],
                )
            )
        )
        self.filter_zero_length = filter_zero_length
        self.max_seq_len = max_seq_len

    def __call__(self, samples: Generator):
        # print(samples)
        for sample in samples:
            # print(sample["user_id"])
            if self.filter_zero_length and (
                sample["user_tr_count"] == 0 or check_for_zero_len(sample)
            ):
                continue
            for key in sample:
                if type(sample[key]) in (list, np.ndarray):
                    # берем последние записи - сортируй по возрастанию даты
                    sample[key] = sample[key][-self.max_seq_len :]
            for processor in self.feature_processors:
                sample = processor(sample)
                if sample is None:
                    break
            if sample is not None:
                yield sample
