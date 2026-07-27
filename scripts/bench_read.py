"""Measure GSLC read/write throughput on a real granule.

Three things, all of which the code makes a choice about that only a measurement
can justify:

1. **direct-chunk vs plain h5py** -- the direct-chunk path exists because h5py
   serialises every call on one global lock, so the gzip inflate cannot use more
   than a core through it.
2. **read-block size** -- how much one task decompresses is independent of how the
   result is stored. Too big and a modest crop is a handful of dask blocks on a
   ten-core box; too small and the downstream multilook's ``4 * looks`` halo
   dominates. ``GSLC.read_chunks`` picks from this trade-off; this sweep is how you
   check its answer.
3. **Zarr write throughput and the compressor** -- ``workspace._default_compressor``
   uses ``Blosc(lz4, clevel=1)`` because SAR noise does not compress and clevel=5
   cost 2.5x the write for a 1.07:1 ratio.

Run it on the machine you actually process on -- every number depends on core
count and on whether the granules live on local disk or a network filesystem.

    python scripts/bench_read.py /path/to/granule.h5 [--size 8192]
                                 [--chunks 512,1024,2048] [--threads 1,2,10]
                                 [--write DIR]
"""

import argparse
import shutil
import time
from pathlib import Path

import dask
import dask.array as da
import numpy as np

from nisar_tools import GSLC


def _ints(text):
    return [int(v) for v in text.split(",") if v.strip()]


def _time(fn):
    start = time.perf_counter()
    out = fn()
    return time.perf_counter() - start, out


def bench_reader(g, reader, sl, gb):
    """direct-chunk vs h5py at the granule's default block size."""
    print("\n== direct-chunk vs h5py ==")
    results = {}
    for label, source, lock in (("h5py", g._dset, True),
                                ("direct-chunk", reader, False)):
        arr = da.from_array(source, chunks=g.chunks, lock=lock)[sl]
        elapsed, out = _time(arr.compute)
        results[label] = (elapsed, out)
        print(f"  {label:14s} {elapsed:6.2f}s  {gb * 1000 / elapsed:7.1f} MB/s")

    t_h5, a_h5 = results["h5py"]
    t_dc, a_dc = results["direct-chunk"]
    identical = np.array_equal(a_h5.view(np.uint8), a_dc.view(np.uint8))
    print(f"  speedup {t_h5 / t_dc:.2f}x   byte-identical: {identical}")
    if not identical:
        raise SystemExit("MISMATCH: direct-chunk output differs from h5py")
    return a_dc


def bench_chunks(g, reader, sl, gb, chunk_sizes, thread_counts):
    """Read-block size x worker threads. The point of the sweep is that the best
    block size is a function of the *region*, not a constant."""
    print("\n== read-block size x threads (direct-chunk, MB/s) ==")
    header = "  chunk   blocks " + "".join(f"{f'{t}t':>10}" for t in thread_counts)
    print(header)
    for size in chunk_sizes:
        chunks = (min(size, g.shape[0]), min(size, g.shape[1]))
        arr = da.from_array(reader, chunks=chunks, lock=False)[sl]
        nblocks = int(np.prod(arr.numblocks))
        cells = []
        for threads in thread_counts:
            with dask.config.set(num_workers=threads):
                elapsed, _ = _time(arr.compute)
            cells.append(f"{gb * 1000 / elapsed:10.1f}")
        print(f"  {size:5d} {nblocks:8d} " + "".join(cells))
    auto = g.read_chunks(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
    print(f"  GSLC.read_chunks would pick {auto} for this region")


def bench_write(array, outdir):
    """Zarr write throughput, and what the compressor choice actually buys."""
    import numcodecs
    import xarray as xr

    from nisar_tools.workspace import _default_compressor

    print("\n== zarr write (one persist of the same window) ==")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gb = array.nbytes / 1e9
    ds = xr.Dataset(
        {"slc": (("y", "x"), da.from_array(array, chunks=(2048, 2048)))}
    )
    codecs = {
        "none": None,
        "lz4 clevel=1 (current)": _default_compressor(),
        "lz4 clevel=5 (zarr default)": numcodecs.Blosc(
            "lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE
        ),
    }
    for label, codec in codecs.items():
        target = outdir / "bench.zarr"
        if target.exists():
            shutil.rmtree(target)
        encoding = {"slc": {"compressor": codec}}
        elapsed, _ = _time(
            lambda: ds.to_zarr(target, mode="w", encoding=encoding, consolidated=True)
        )
        on_disk = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        print(f"  {label:30s} {elapsed:6.2f}s  {gb * 1000 / elapsed:7.1f} MB/s  "
              f"ratio {array.nbytes / max(on_disk, 1):.2f}:1")
        shutil.rmtree(target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("granule")
    ap.add_argument("--size", type=int, default=8192, help="square crop, pixels")
    ap.add_argument("--chunks", type=_ints, default=[512, 1024, 2048],
                    help="read-block sizes to sweep")
    ap.add_argument("--threads", type=_ints, default=None,
                    help="dask worker counts to sweep (default 1,cpu/2,cpu)")
    ap.add_argument("--write", metavar="DIR",
                    help="also time a Zarr write into DIR")
    args = ap.parse_args()

    import os

    cpu = os.cpu_count() or 1
    thread_counts = args.threads or sorted({1, max(1, cpu // 2), cpu})

    g = GSLC(args.granule)
    try:
        reader = g._reader()
        print(f"granule {g.shape} {g._dset.dtype}, hdf5 chunks {g._dset.chunks}, "
              f"compression {g._dset.compression}, shuffle {g._dset.shuffle}")
        print(f"{cpu} cores available")
        if reader is None:
            print("filter pipeline is not directly decodable -- falling back to "
                  "h5py; there is nothing to compare here.")
            return

        # A window in the middle of the granule, which is inside the swath.
        n = min(args.size, g.shape[0], g.shape[1])
        y0 = (g.shape[0] - n) // 2
        x0 = (g.shape[1] - n) // 2
        sl = (slice(y0, y0 + n), slice(x0, x0 + n))
        gb = n * n * g._dset.dtype.itemsize / 1e9
        print(f"reading {n}x{n} = {gb:.2f} GB, default dask chunks {g.chunks}")

        window = bench_reader(g, reader, sl, gb)
        bench_chunks(g, reader, sl, gb, args.chunks, thread_counts)
        if args.write:
            bench_write(window, args.write)
    finally:
        g.close()


if __name__ == "__main__":
    main()
