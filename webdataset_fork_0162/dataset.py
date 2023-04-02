import webdataset
import subprocess as sp
import pathlib
import numpy as np
import random

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
    path, feature_processor, collator, batch_size, shuffle=False, hdfs=False, shuffle_buf_size=100
):
    """
    shuffle - random parquet file read
    shuffle_buf_size if>1, - random batch inside shard(parquet). It could be <= len(shard)/batch_size
    """
    if hdfs:
        path_to_shards = ls_path_hdfs(path)
    else:
        path_to_shards = ls_path(path)

    dataset = (
        webdataset.ShardList(urls=path_to_shards, shuffle=shuffle)
        .then(download_shards)
        .then(read_table, "parquet")
        .then(chain_dataframe)
        .then(shuffle_it, shuffle_buf_size)
        .then(feature_processor)
        .batched(batchsize=batch_size, collation_fn=collator)
    )

    return dataset


def shuffle_it(data, bufsize=1000, initial=100, rng=random, handler=None):
    """Shuffle the data in the stream.
    This uses a buffer of size `bufsize`. Shuffling at
    startup is less random; this is traded off against
    yielding samples quickly.
    data: iterator
    bufsize: buffer size for shuffling
    returns: iterator
    rng: either random module or random.Random instance
    """
    initial = min(initial, bufsize)
    buf = []
    startup = True
    for sample in data:
        if len(buf) < bufsize:
            try:
                buf.append(next(data))  # skipcq: PYL-R1708
                #print(buf)
            except StopIteration:
                pass
        k = rng.randint(0, len(buf) - 1)
        sample, buf[k] = buf[k], sample
        if startup and len(buf) < initial:
            buf.append(sample)
            continue
        startup = False
        yield sample
    for sample in buf:
        yield sample