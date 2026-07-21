"""Measure GSLC read throughput on a real granule, direct-chunk vs plain h5py.

The direct-chunk path exists because h5py serialises every call on one global
lock, so the gzip inflate cannot use more than a core through it. Run this on
the machine you actually process on -- the win depends on core count and on
whether the granules live on local disk or a network filesystem.

    python scripts/bench_read.py /path/to/granule.h5 [--size 8192]
"""

import argparse
import time

import dask.array as da
import numpy as np

from nisar_tools import GSLC
from nisar_tools.gslc import DirectChunkReader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("granule")
    ap.add_argument("--size", type=int, default=8192, help="square crop, pixels")
    args = ap.parse_args()

    g = GSLC(args.granule)
    try:
        reader = g._reader()
        print(f"granule {g.shape} {g._dset.dtype}, hdf5 chunks {g._dset.chunks}, "
              f"compression {g._dset.compression}, shuffle {g._dset.shuffle}")
        if reader is None:
            print("filter pipeline is not directly decodable -- falling back to h5py; "
                  "there is nothing to compare here.")
            return

        # A window in the middle of the granule, which is inside the swath.
        n = min(args.size, g.shape[0], g.shape[1])
        y0 = (g.shape[0] - n) // 2
        x0 = (g.shape[1] - n) // 2
        sl = (slice(y0, y0 + n), slice(x0, x0 + n))
        gb = n * n * g._dset.dtype.itemsize / 1e9
        print(f"reading {n}x{n} = {gb:.2f} GB, dask chunks {g.chunks}\n")

        results = {}
        for label, source, lock in (("h5py", g._dset, True),
                                    ("direct-chunk", reader, False)):
            arr = da.from_array(source, chunks=g.chunks, lock=lock)[sl]
            start = time.perf_counter()
            out = arr.compute()
            elapsed = time.perf_counter() - start
            results[label] = (elapsed, out)
            print(f"  {label:14s} {elapsed:6.2f}s  {gb * 1000 / elapsed:7.1f} MB/s")

        t_h5, a_h5 = results["h5py"]
        t_dc, a_dc = results["direct-chunk"]
        identical = np.array_equal(a_h5.view(np.uint8), a_dc.view(np.uint8))
        print(f"\n  speedup {t_h5 / t_dc:.2f}x   byte-identical: {identical}")
        if not identical:
            raise SystemExit("MISMATCH: direct-chunk output differs from h5py")
    finally:
        g.close()


if __name__ == "__main__":
    main()
