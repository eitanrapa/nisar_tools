#!/usr/bin/env python
"""Compare the HDF5 *format* of two NISAR GSLC files, ignoring their data.

Usage::

    python scripts/gslc_format_diff.py REFERENCE.h5 CANDIDATE.h5 [--values]

Walks both files and reports, per object: presence, dtype, rank, chunking,
filter pipeline, fill value, and every attribute's name and type. Shapes are
compared only where they do not depend on the scene (scalars, and the rank of
everything else), because two granules of different frames legitimately differ
in size. ``--values`` additionally compares scalar values and attribute values
that are scene-independent (descriptions, units, CF names).

Built to check :mod:`nisar_tools.gslc_product` against an ASF-distributed GSLC;
exit status is non-zero if anything structural differs.
"""

import argparse
import sys

import h5py
import numpy as np

#: Attributes whose values legitimately vary between granules.
SCENE_DEPENDENT_ATTRS = {
    "min_real_value", "max_real_value", "mean_real_value", "sample_stddev_real",
    "min_imag_value", "max_imag_value", "mean_imag_value", "sample_stddev_imag",
    "min_value", "max_value", "mean_value", "sample_stddev", "units",
    "epsg_code", "utm_zone_number", "longitude_of_central_meridian",
    "spatial_ref", "frameCoveragePercentage", "epsg", "DIMENSION_LIST",
    "REFERENCE_LIST",
}


def _attr_signature(value):
    if isinstance(value, bytes):
        return ("bytes",)
    if isinstance(value, str):
        return ("str",)
    arr = np.asarray(value)
    if arr.dtype.kind == "O":
        return ("object", arr.shape)
    return (type(value).__name__, arr.dtype.str, arr.shape)


def signature(path):
    out = {}

    def visit(name, obj):
        key = "/" + name
        if isinstance(obj, h5py.Dataset):
            plist = obj.id.get_create_plist()
            filters = tuple(plist.get_filter(i)[3].decode() + str(plist.get_filter(i)[2])
                            for i in range(plist.get_nfilters()))
            fill = obj.fillvalue
            fill_sig = None if fill is None else repr(np.asarray(fill).item()) \
                if np.asarray(fill).dtype.kind != "V" else "compound"
            out[key] = {
                "kind": "dataset",
                "dtype": obj.dtype.str if obj.dtype.kind != "S" else "S",
                "ndim": obj.ndim,
                "shape": obj.shape if obj.ndim == 0 else None,
                "chunked": obj.chunks is not None,
                "chunk_cap": None if obj.chunks is None else tuple(min(c, 512) for c in obj.chunks),
                "filters": filters,
                "fill": fill_sig,
                "attrs": {k: _attr_signature(v) for k, v in obj.attrs.items()},
                "attr_values": {k: v for k, v in obj.attrs.items()
                                if k not in SCENE_DEPENDENT_ATTRS},
                "value": obj[()] if obj.ndim == 0 else None,
            }
        elif isinstance(obj, h5py.Group):
            out[key] = {"kind": "group", "attrs": {k: _attr_signature(v) for k, v in obj.attrs.items()}}
        else:
            out[key] = {"kind": type(obj).__name__}

    with h5py.File(path, "r") as f:
        out["/"] = {"kind": "root", "attrs": {k: _attr_signature(v) for k, v in f.attrs.items()},
                    "attr_values": dict(f.attrs)}
        for name, obj in f.items():
            if isinstance(obj, h5py.Datatype):
                out["/" + name] = {"kind": "Datatype", "dtype": obj.dtype.str}
        f.visititems(visit)
        fcpl = f.id.get_create_plist()
        out["<file>"] = {"kind": "file", "fs_strategy": fcpl.get_file_space_strategy(),
                         "page_size": fcpl.get_file_space_page_size()}
    return out


def _same(a, b):
    try:
        aa, bb = np.asarray(a), np.asarray(b)
        if aa.dtype.kind in "fc" and bb.dtype.kind in "fc":
            return aa.shape == bb.shape and bool(np.array_equal(aa, bb, equal_nan=True))
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(aa, bb))
        return bool(a == b)
    except Exception:  # noqa: BLE001
        return False


def compare(ref_path, cand_path, values=False, ignore=()):
    ref, cand = signature(ref_path), signature(cand_path)
    problems = []

    def skip(key):
        return any(s in key for s in ignore)

    for key in sorted(set(ref) | set(cand)):
        if skip(key):
            continue
        if key not in cand:
            problems.append(f"MISSING   {key}")
            continue
        if key not in ref:
            problems.append(f"EXTRA     {key}")
            continue
        r, c = ref[key], cand[key]
        for field in ("kind", "dtype", "ndim", "shape", "chunked", "filters", "fill",
                      "fs_strategy", "page_size"):
            if field in r and r.get(field) != c.get(field):
                problems.append(f"{field.upper():9s} {key}: ref={r.get(field)!r} ours={c.get(field)!r}")
        ra, ca = r.get("attrs", {}), c.get("attrs", {})
        for name in sorted(set(ra) | set(ca)):
            if name not in ca:
                problems.append(f"ATTR-MISS {key} @{name} ({ra[name]})")
            elif name not in ra:
                problems.append(f"ATTR-XTRA {key} @{name} ({ca[name]})")
            elif ra[name] != ca[name]:
                problems.append(f"ATTR-TYPE {key} @{name}: ref={ra[name]} ours={ca[name]}")
        if values:
            rv, cv = r.get("attr_values", {}), c.get("attr_values", {})
            for name in sorted(set(rv) & set(cv)):
                if not _same(rv[name], cv[name]):
                    problems.append(f"ATTR-VAL  {key} @{name}: ref={rv[name]!r:.80} ours={cv[name]!r:.80}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--values", action="store_true",
                        help="also compare scene-independent attribute values")
    parser.add_argument("--ignore", action="append", default=[],
                        help="substring of paths to skip (repeatable)")
    args = parser.parse_args(argv)
    problems = compare(args.reference, args.candidate, values=args.values,
                       ignore=args.ignore)
    for line in problems:
        print(line)
    print(f"{len(problems)} difference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
