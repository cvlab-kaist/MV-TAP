"""Materialize a view-sampled Panoptic-multiview dataset loadable by data/panoptic.py.

The raw dataset at ``--src`` has, per sequence::

    <seq>/
        tapvid3d_annotations.npz      # trajectories_pixelspace (V,T,N,2),
                                      # per_view_visibilities (V,T,N),
                                      # intrinsics (V,3,3), extrinsics (V,4,4) [world->cam],
                                      # trajectories (T,N,3), query_points_3d (N,4)
        ims/<view>/*.jpg              # one sub-dir of frames per camera view
        (seg/, dynamic3dgs_depth/, *_meta.json, ...)

Unlike DexYCB, *all* per-view arrays (tracks AND camera params) live in the
single seq-level npz, and the view dimension is axis 0.  This script selects
``K`` of the ``V`` views per sequence with one of the strategies in
``data/view_sampling.py`` and writes a new ``--dst`` root with only the kept
views, renumbered contiguously to ``ims/0 .. ims/{K-1}``.  Every view-axis
array in the npz is sliced in the SAME order so it stays aligned with the
renumbered image dirs (exactly what data/panoptic.py assumes: ims dirs sorted
by int(name) <-> axis 0 of the npz arrays).

Image dirs are symlinked by default to avoid duplicating the (large) frames;
pass --copy to hard-copy instead.

Run from the MV-TAP repo root; the raw data is assumed to live under
``datasets/panoptic-multiview``::

    python scripts/sample_panoptic_views.py --method fps --num-views 4

By default the output is written to ``datasets/panoptic_sampled/<method>_<K>views``.
Then point the loader at it::

    Panoptic(data_root="datasets/panoptic_sampled/fps_4views")
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

# allow running as a standalone script (python scripts/sample_panoptic_views.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.view_sampling import camera_centers, sample_views  # noqa: E402

ANNOT_FILE = "tapvid3d_annotations.npz"
IMS_DIR = "ims"
# arrays inside the npz whose axis 0 is the view dimension and must be sliced
VIEW_AXIS_KEYS = ("trajectories_pixelspace", "per_view_visibilities", "intrinsics", "extrinsics")


def list_sequences(src):
    return [
        name for name in sorted(os.listdir(src))
        if os.path.isdir(os.path.join(src, name))
        and not name.startswith(".")
        and not name.startswith("_")
    ]


def list_view_dirs(seq_dir):
    """Return ims sub-dirs sorted by integer name (matches data/panoptic.py)."""
    ims = os.path.join(seq_dir, IMS_DIR)
    names = [n for n in os.listdir(ims) if os.path.isdir(os.path.join(ims, n))]
    return sorted(names, key=lambda x: int(x))


def link_or_copy(src_path, dst_path, copy):
    if copy:
        shutil.copytree(src_path, dst_path)
    else:
        os.symlink(os.path.abspath(src_path), dst_path)


def write_sliced_annotations(src_npz, dst_npz, sel_indices, num_src_views):
    data = np.load(src_npz, allow_pickle=True)
    out = {}
    for key in data.files:
        arr = data[key]
        if key in VIEW_AXIS_KEYS:
            if arr.shape[0] != num_src_views:
                raise ValueError(
                    f"{key} has axis-0 {arr.shape[0]} != #views {num_src_views} in {src_npz}"
                )
            out[key] = arr[list(sel_indices)]
        else:
            out[key] = arr
    np.savez(dst_npz, **out)


def process_sequence(src_seq, dst_seq, method, k, copy):
    view_dirs = list_view_dirs(src_seq)          # i-th entry <-> npz view index i
    num_src_views = len(view_dirs)

    annot = np.load(os.path.join(src_seq, ANNOT_FILE), allow_pickle=True)
    centers = camera_centers(annot["extrinsics"])
    if len(centers) != num_src_views:
        raise ValueError(
            f"{src_seq}: extrinsics has {len(centers)} views but ims/ has {num_src_views}"
        )

    sel = sample_views(method, centers, k)        # sorted ascending view indices
    sel_view_dirs = [view_dirs[i] for i in sel]

    os.makedirs(os.path.join(dst_seq, IMS_DIR), exist_ok=True)

    # 1) sliced annotation npz (all view-axis arrays sliced in selection order)
    write_sliced_annotations(
        os.path.join(src_seq, ANNOT_FILE),
        os.path.join(dst_seq, ANNOT_FILE),
        sel,
        num_src_views,
    )

    # 2) renumbered ims/<new_idx> dirs linked/copied from the selected views
    for new_idx, src_view in enumerate(sel_view_dirs):
        link_or_copy(os.path.join(src_seq, IMS_DIR, src_view),
                     os.path.join(dst_seq, IMS_DIR, str(new_idx)), copy)

    return {
        "selected_view_indices": sel,
        "selected_view_dirs": sel_view_dirs,
        "camera_centers": {view_dirs[i]: centers[i].round(5).tolist()
                           for i in range(num_src_views)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["fps", "nearest", "random"])
    ap.add_argument("--num-views", type=int, default=4, help="K: views to keep per sequence")
    ap.add_argument("--src", default="datasets/panoptic-multiview",
                    help="raw panoptic-multiview root (default: datasets/panoptic-multiview)")
    ap.add_argument("--dst", default=None,
                    help="output root (default: datasets/panoptic_sampled/<method>_<K>views)")
    ap.add_argument("--copy", action="store_true",
                    help="copy image dirs instead of symlinking (default: symlink)")
    ap.add_argument("--overwrite", action="store_true",
                    help="remove --dst if it already exists")
    args = ap.parse_args()

    if args.dst is None:
        args.dst = os.path.join("datasets", "panoptic_sampled",
                                f"{args.method}_{args.num_views}views")

    if os.path.exists(args.dst):
        if args.overwrite:
            shutil.rmtree(args.dst)
        else:
            sys.exit(f"[error] --dst already exists: {args.dst} (use --overwrite)")

    seqs = list_sequences(args.src)
    if not seqs:
        sys.exit(f"[error] no sequences found under {args.src}")

    os.makedirs(args.dst, exist_ok=True)
    manifest = {
        "src": os.path.abspath(args.src),
        "method": args.method,
        "num_views": args.num_views,
        "linked": not args.copy,
        "sequences": {},
    }

    for i, seq in enumerate(seqs, 1):
        info = process_sequence(
            os.path.join(args.src, seq),
            os.path.join(args.dst, seq),
            args.method, args.num_views, args.copy,
        )
        manifest["sequences"][seq] = info
        print(f"[{i}/{len(seqs)}] {seq}: kept views "
              f"{info['selected_view_dirs']} (orig idx {info['selected_view_indices']})")

    with open(os.path.join(args.dst, "sampling_info.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(seqs)} sequences -> {args.dst}")
    print(f"  method={args.method} num_views={args.num_views} "
          f"({'symlink' if not args.copy else 'copy'})")
    print(f"  manifest: {os.path.join(args.dst, 'sampling_info.json')}")


if __name__ == "__main__":
    main()
