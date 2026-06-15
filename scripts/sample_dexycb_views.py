"""Materialize a view-sampled DexYCB-multiview dataset loadable by data/dexycb.py.

The raw dataset at ``--src`` has, per sequence::

    <seq>/
        tracks_3d.npz                 # tracks_2d (V,T,N,2), tracks_2d_visibilities (V,T,N),
                                      # tracks_2d_z (V,T,N), tracks_3d (T,N,3), ...
        view_00/ ... view_07/
            rgb/*.png
            intrinsics_extrinsics.npz # intrinsics (4,4), extrinsics (4,4) [world->cam]
            (depth/, mask/, ...)

This script selects ``K`` of the ``V`` views per sequence with one of the
sampling strategies in ``data/view_sampling.py`` and writes a new ``--dst``
root with the *same layout* but only the kept views, renumbered contiguously
to ``view_00 .. view_{K-1}``.  The view-axis arrays in ``tracks_3d.npz`` are
sliced in the SAME order so they stay aligned with the renumbered views (this
is exactly what data/dexycb.py assumes: view dirs sorted by index <-> axis 0
of tracks_2d).

RGB frames and the per-view camera npz are symlinked by default to avoid
duplicating the (large) image data; pass --copy to hard-copy instead.

Run from the MV-TAP repo root; the raw data is assumed to live under
``datasets/dex-ycb-multiview``::

    python scripts/sample_dexycb_views.py --method fps --num-views 4

By default the output is written to ``datasets/dexycb_sampled/<method>_<K>views``.
Then point the loader at it::

    DexYCB(data_root="datasets/dexycb_sampled/fps_4views")
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

# allow running as a standalone script (python scripts/sample_dexycb_views.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.view_sampling import camera_centers, sample_views  # noqa: E402

TRACKS_FILE = "tracks_3d.npz"
CAM_FILE = "intrinsics_extrinsics.npz"
RGB_DIR = "rgb"
# arrays inside tracks_3d.npz whose axis 0 is the view dimension and must be sliced
VIEW_AXIS_KEYS = ("tracks_2d", "tracks_2d_z", "tracks_2d_visibilities")


def list_sequences(src):
    return [
        name for name in sorted(os.listdir(src))
        if os.path.isdir(os.path.join(src, name))
        and not name.startswith(".")
        and not name.startswith("_")
    ]


def list_views(seq_dir):
    views = [
        name for name in os.listdir(seq_dir)
        if name.startswith("view_") and os.path.isdir(os.path.join(seq_dir, name))
    ]
    return sorted(views, key=lambda x: int(x.split("_")[-1]))


def load_extrinsics(seq_dir, view_names):
    extr = []
    for v in view_names:
        cam = np.load(os.path.join(seq_dir, v, CAM_FILE))
        extr.append(cam["extrinsics"])
    return extr


def link_or_copy(src_path, dst_path, copy):
    if copy:
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    else:
        os.symlink(os.path.abspath(src_path), dst_path)


def write_sliced_tracks(src_npz, dst_npz, sel_indices, num_src_views):
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
    view_names = list_views(src_seq)
    num_src_views = len(view_names)

    extr = load_extrinsics(src_seq, view_names)
    centers = camera_centers(extr)

    sel = sample_views(method, centers, k)  # sorted ascending indices
    sel_view_names = [view_names[i] for i in sel]

    os.makedirs(dst_seq, exist_ok=True)

    # 1) sliced tracks_3d.npz (view-axis arrays sliced in selection order)
    write_sliced_tracks(
        os.path.join(src_seq, TRACKS_FILE),
        os.path.join(dst_seq, TRACKS_FILE),
        sel,
        num_src_views,
    )

    # 2) renumbered view dirs with rgb + camera npz linked/copied
    for new_idx, src_view in enumerate(sel_view_names):
        dst_view = os.path.join(dst_seq, f"view_{new_idx:02d}")
        os.makedirs(dst_view, exist_ok=True)
        link_or_copy(os.path.join(src_seq, src_view, RGB_DIR),
                     os.path.join(dst_view, RGB_DIR), copy)
        link_or_copy(os.path.join(src_seq, src_view, CAM_FILE),
                     os.path.join(dst_view, CAM_FILE), copy)

    return {
        "selected_view_indices": sel,
        "selected_view_names": sel_view_names,
        "camera_centers": {view_names[i]: centers[i].round(5).tolist()
                           for i in range(num_src_views)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["fps", "nearest", "random"])
    ap.add_argument("--num-views", type=int, default=4, help="K: views to keep per sequence")
    ap.add_argument("--src", default="datasets/dex-ycb-multiview",
                    help="raw dex-ycb-multiview root (default: datasets/dex-ycb-multiview)")
    ap.add_argument("--dst", default=None,
                    help="output root (default: datasets/dexycb_sampled/<method>_<K>views)")
    ap.add_argument("--copy", action="store_true",
                    help="copy rgb/camera files instead of symlinking (default: symlink)")
    ap.add_argument("--overwrite", action="store_true",
                    help="remove --dst if it already exists")
    args = ap.parse_args()

    if args.dst is None:
        args.dst = os.path.join("datasets", "dexycb_sampled",
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
              f"{info['selected_view_names']} (orig idx {info['selected_view_indices']})")

    with open(os.path.join(args.dst, "sampling_info.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(seqs)} sequences -> {args.dst}")
    print(f"  method={args.method} num_views={args.num_views} "
          f"({'symlink' if not args.copy else 'copy'})")
    print(f"  manifest: {os.path.join(args.dst, 'sampling_info.json')}")


if __name__ == "__main__":
    main()
