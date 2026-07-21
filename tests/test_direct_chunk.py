"""Decoding HDF5 chunks outside h5py's lock must match h5py exactly.

:class:`~nisar_tools.gslc.DirectChunkReader` exists only to go faster: h5py
serialises every call on one global lock, so the gzip inflate that dominates a
GSLC read cannot use more than a core through it. Reading the raw chunk and
inflating it ourselves lifts that, but it means reimplementing the filter
pipeline — so the contract these tests pin is that the result is *byte*
identical to h5py, and that anything we cannot decode falls back rather than
guessing.
"""

import h5py
import numpy as np
import pytest

from nisar_tools import GSLC
from nisar_tools.gslc import DirectChunkReader, _is_directly_decodable

CHUNK = 64


def _dataset(tmp_path, name="d.h5", *, shape=(200, 180), write=True, **kwargs):
    """A standalone chunked dataset, gzip+shuffle unless overridden."""
    opts = {"compression": "gzip", "compression_opts": 1, "shuffle": True,
            "fillvalue": np.complex64(complex(np.nan, np.nan))}
    opts.update(kwargs)
    if opts.get("compression") != "gzip":
        opts.pop("compression_opts", None)   # h5py rejects it for other filters
    path = tmp_path / name
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(
        np.complex64
    )
    with h5py.File(path, "w") as f:
        d = f.create_dataset("img", shape=shape, dtype=np.complex64,
                             chunks=(CHUNK, CHUNK), **opts)
        if write:
            d[...] = data
    return path


# -- byte-exactness ---------------------------------------------------------


@pytest.mark.parametrize("window", [
    (slice(0, CHUNK), slice(0, CHUNK)),                 # exactly one chunk
    (slice(10, 150), slice(20, 170)),                   # straddles four chunks
    (slice(0, 200), slice(0, 180)),                     # the whole array
    (slice(190, 200), slice(170, 180)),                 # ragged far corner
    (slice(0, 1), slice(0, 180)),                       # single row
    (slice(0, 200), slice(5, 6)),                       # single column
    (slice(CHUNK - 1, CHUNK + 1), slice(CHUNK - 1, CHUNK + 1)),  # across a seam
])
def test_reader_is_byte_identical_to_h5py(tmp_path, window):
    path = _dataset(tmp_path)
    with h5py.File(path, "r") as f:
        dset = f["img"]
        got = DirectChunkReader(dset)[window]
        ref = np.asarray(dset[window])
        assert got.dtype == ref.dtype
        assert got.shape == ref.shape
        # .view(uint8) so NaN payloads must match too, not just compare equal.
        np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8))


def test_unallocated_chunks_return_the_fill_value(tmp_path):
    """~45% of a real granule is outside the swath and never written.

    ``read_direct_chunk`` raises for those, so they have to be detected first.
    """
    path = _dataset(tmp_path, shape=(192, 192), write=False)
    with h5py.File(path, "a") as f:
        f["img"][0:CHUNK, 0:CHUNK] = np.complex64(1 + 2j)   # one chunk only

    with h5py.File(path, "r") as f:
        dset = f["img"]
        assert dset.id.get_chunk_info_by_coord((CHUNK, CHUNK)).byte_offset is None
        got = DirectChunkReader(dset)[slice(0, 192), slice(0, 192)]
        ref = np.asarray(dset[0:192, 0:192])
        np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8))
        assert got[0, 0] == np.complex64(1 + 2j)
        assert np.isnan(got[CHUNK + 1, CHUNK + 1])


def test_reader_without_shuffle(tmp_path):
    """Shuffle is a separate filter; gzip alone must decode too."""
    path = _dataset(tmp_path, shuffle=False)
    with h5py.File(path, "r") as f:
        dset = f["img"]
        assert _is_directly_decodable(dset)
        got = DirectChunkReader(dset)[slice(0, 200), slice(0, 180)]
        np.testing.assert_array_equal(
            got.view(np.uint8), np.asarray(dset[:]).view(np.uint8)
        )


# -- fallback ---------------------------------------------------------------


@pytest.mark.parametrize("tag,kwargs", [
    ("uncompressed", {"compression": None, "shuffle": False}),
    ("lzf not gzip", {"compression": "lzf", "compression_opts": None}),
    ("fletcher32 checksum", {"fletcher32": True}),
])
def test_unsupported_pipelines_fall_back(tmp_path, tag, kwargs):
    """Anything we cannot invert must use h5py, not guess at the bytes."""
    path = _dataset(tmp_path, name=f"{tag}.h5", **kwargs)
    with h5py.File(path, "r") as f:
        dset = f["img"]
        assert not _is_directly_decodable(dset)
        with pytest.raises(ValueError, match="not directly decodable"):
            DirectChunkReader(dset)


def test_contiguous_dataset_falls_back(tmp_path):
    path = tmp_path / "contig.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("img", data=np.zeros((8, 8), np.complex64))  # no chunks
    with h5py.File(path, "r") as f:
        assert f["img"].chunks is None
        assert not _is_directly_decodable(f["img"])


# -- through the real GSLC path ---------------------------------------------


def test_gslc_uses_the_reader_and_matches_h5py(gslc_factory):
    g = GSLC(gslc_factory(ny=200, nx=180))
    try:
        assert isinstance(g._reader(), DirectChunkReader)
        got = g.data.compute().values
        ref = np.asarray(g._dset[:])
        np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8))
    finally:
        g.close()


def test_gslc_falls_back_for_uncompressed_granules(gslc_factory):
    g = GSLC(gslc_factory(ny=120, nx=120, compressed=False))
    try:
        assert g._reader() is None
        got = g.data.compute().values
        np.testing.assert_array_equal(got, np.asarray(g._dset[:]))
    finally:
        g.close()


def test_reader_result_is_chunk_layout_independent(gslc_factory):
    """Dask block size must not change the values (it changes the read windows)."""
    path = gslc_factory(ny=200, nx=180, seed=7)
    ref = None
    for chunks in ((64, 64), (128, 96), (2048, 2048)):
        g = GSLC(path, chunks=chunks)
        try:
            got = g.data.compute().values
        finally:
            g.close()
        if ref is None:
            ref = got
        np.testing.assert_array_equal(got.view(np.uint8), ref.view(np.uint8))
