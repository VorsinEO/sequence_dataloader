# import os
import io
import urllib.parse

# import contextlib
import pyarrow
from contextlib import closing

# from webdataset import reraise_exception
# from webdataset.shardcache import guess_shard
from webdataset.gopen import gopen_pipe


def file_download_wds(uri):
    """Read parquet with gopen_pipe(webdataset) - wrapper over subproccess popen/PIPE
    load parquet in memory
    """
    pq_file = f"pipe:cat {uri.path}"
    with closing(gopen_pipe(pq_file)) as inp_stream:
        seekable_stream = io.BytesIO(inp_stream.read())
    return seekable_stream


def file_download(uri, *args, **kwargs):
    """Read parquet with open and BytesIO - wrapper over subproccess popen/PIPE
    load parquet in memory
    """
    with open(uri.path, mode="rb") as inp_stream:
        seekable_stream = io.BytesIO(inp_stream.read())
    return seekable_stream


# TODO add logger
def download_shards(shards, **kwargs):
    """Modification of webdataset.url_opener for pyarrow.parquet requirements"""
    for shard in shards:
        assert isinstance(shard, dict)
        assert "url" in shard
        data = download_shard(shard["url"], **kwargs)
        shard.update(data=data)
        yield shard


def download_shard(url, *args, **kwargs):
    uri = urllib.parse.urlparse(url, scheme="file", allow_fragments=False)
    if uri.scheme not in download_schemes:
        raise ValueError(f"{uri.scheme}: unknown source")
    data = download_schemes[uri.scheme](uri, *args, **kwargs)
    return data


# TODO test with HDFS
def hdfs_download(uri, **kwargs):
    if uri.hostname is not None:
        kwargs.update(host=uri.hostname)
    if uri.port is not None:
        kwargs.update(port=uri.port)
    with closing(pyarrow.hdfs.connect(**kwargs)) as hdfs:
        with hdfs.open(uri.path) as stream:
            return stream.read()


# TODO test with HDFS
def hdfs_download_wds(uri):
    """Read parquet with gopen_pipe(webdataset) - wrapper over subproccess popen/PIPE
    load parquet in memory
    """
    pq_file = f"pipe:hdfs dfs -cat {uri.path}"
    with closing(gopen_pipe(pq_file)) as inp_stream:
        seekable_stream = io.BytesIO(inp_stream.read())
    return seekable_stream


download_schemes = dict(file=file_download_wds, hdfs=hdfs_download_wds)
