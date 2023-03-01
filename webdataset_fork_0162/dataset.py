import webdataset
import subprocess as sp
import pathlib
import numpy as np

from webdataset_fork_0162 import download_shards, read_table, chain_dataframe


def ls_path_hdfs(path):
    pipe = sp.Popen(
        f"hadoop fs -ls -C {path}", shell=True, stdout=sp.PIPE, stderr=sp.PIPE
    )
    out, err = pipe.communicate()
    url_list = list(filter(len, out.decode("utf-8").split("\n")))
    return url_list


def ls_path(path):
    data_path = pathlib.Path(path)
    url_list = [item for item in data_path.iterdir() if not item.is_dir()]
    url_list = [item.as_uri() for item in url_list if str(item).endswith("parquet")]
    return url_list


def get_dataset(
    path, feature_processor, collator, batch_size, shuffle=False, hdfs=False
):
    if hdfs:
        path_to_shards = ls_path_hdfs(path)
    else:
        path_to_shards = ls_path(path)

    dataset = (
        webdataset.ShardList(urls=path_to_shards, shuffle=shuffle)
        .then(download_shards)
        .then(read_table, "parquet")
        .then(chain_dataframe)
        .then(feature_processor)
        .batched(batchsize=batch_size, collation_fn=collator)
    )

    return dataset
