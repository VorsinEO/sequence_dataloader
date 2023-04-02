import os

# import pyarrow
import pyarrow.parquet as pq
import numpy as np
from collections import namedtuple


def read_parquet(source, columns=None):
    dataframe = pq.read_table(
        source=source,
        columns=columns,
        # use_legacy_dataset=True
    ).to_pandas()
    return dataframe


def read_unknown(source, *args, **kwargs):
    raise ValueError(f"{source}: unknown file type")


table_readers = dict(__default__=read_unknown, parquet=read_parquet)


def read_table(shards, file_type, columns=None):
    table_reader = table_readers.get(file_type, table_readers["__default__"])
    for shard in shards:
        assert isinstance(shard, dict)
        dataframe = table_reader(shard["data"], columns=columns)
        if dataframe.shape[0] < 1:
            continue
        yield dataframe


def chain_dataframe(dataframes):
    for df in dataframes:
        for i in range(df.shape[0]):
            yield df.iloc[i].to_dict()
