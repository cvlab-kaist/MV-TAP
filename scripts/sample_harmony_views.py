"""Materialize a view-sampled Harmony4D dataset loadable by data/harmony4d.py.

Unlike DexYCB / Panoptic, the raw extracted Harmony4D data is NOT already in the
layout the loader expects -- the per-view tracks and camera parameters are split
across several files.  This script both (a) ASSEMBLES the per-sequence
``annotations.npy`` the loader wants and (b) sub-samples ``K`` of the ``V``
camera views.

Raw layout at ``--src`` (a ``data_root`` with category/scene nesting)::

    <category>/<scene>/
        rectified_tracks_from_smpl.npy   # dict: trajectory (V,T,N,2), visibility (V,T,N),
                                         #       flow_filter_masks (V,N), trajectory_3d (T,N,3)
        extrinsics.npy                   # (V,4,4) world->cam
        cam01/ .. camVV/
            *.jpg                        # T frames
            pinhole_K.npy                # (3,3) intrinsics for that camera

Loader (data/harmony4d.py) expects, per ``<category>/<scene>``::

    <category>/<scene>/
        annotations.npy                  # dict: trajectory, visibility, flow_filter_masks,
                                         #       intrinsics (V,3,3), extrinsics (V,4,4)
        cam*/ *.jpg

This script writes a new ``--dst`` root with the same category/scene nesting but
only the kept views, renumbered contiguously to ``cam01 .. cam{K}``.  Every
view-axis array is sliced in the SAME order so it stays aligned with the
renumbered cam dirs (data/harmony4d.py pairs cam dirs sorted by int(name) with
axis 0 of the annotation arrays).

Camera image dirs are symlinked by default to avoid duplicating frames; pass
--copy to hard-copy instead.

Run from the MV-TAP repo root; the raw data is assumed to live under
``datasets/harmony-multiview``::

    python scripts/sample_harmony_views.py --method fps --num-views 4

By default the output is written to ``datasets/harmony_sampled/<method>_<K>views``.
Then point the loader at it::

    Harmony4D(data_root="datasets/harmony_sampled/fps_4views")
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

# allow running as a standalone script (python scripts/sample_harmony_views.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.view_sampling import camera_centers, sample_views  # noqa: E402

TRACKS_FILE = "rectified_tracks_from_smpl.npy"
EXTR_FILE = "extrinsics.npy"
INTR_FILE = "pinhole_K.npy"
ANNOT_FILE = "annotations.npy"
# keys copied through from the raw tracks dict that carry a view axis (axis 0)
VIEW_AXIS_TRACK_KEYS = ("trajectory", "visibility", "flow_filter_masks")


def list_sequences(src):
    """Return (category, scene) pairs, mirroring data/harmony4d.py's walk."""
    seqs = []
    for cat in sorted(os.listdir(src)):
        cat_dir = os.path.join(src, cat)
        if cat.startswith(".") or cat.startswith("_") or not os.path.isdir(cat_dir):
            continue
        for scene in sorted(os.listdir(cat_dir)):
            if os.path.isdir(os.path.join(cat_dir, scene)):
                seqs.append((cat, scene))
    return seqs


def list_cam_dirs(seq_dir):
    """Return cam sub-dirs sorted by integer id (matches data/harmony4d.py)."""
    names = [n for n in os.listdir(seq_dir)
             if n.startswith("cam") and os.path.isdir(os.path.join(seq_dir, n))]
    return sorted(names, key=lambda x: int(x.replace("cam", "")))


def link_or_copy(src_path, dst_path, copy):
    if copy:
        shutil.copytree(src_path, dst_path)
    else:
        os.symlink(os.path.abspath(src_path), dst_path)


def sorted_jpgs(cam_dir):
    jpgs = glob.glob(os.path.join(cam_dir, "*.jpg"))
    return sorted(jpgs, key=lambda x: int(os.path.basename(x).replace("rgba_", "").split(".")[0]))


def link_cam(src_cam, dst_cam, keep_n, n_jpg, copy):
    """Link a camera's frames into dst_cam, keeping only the first `keep_n` jpgs.

    Fast path (whole-dir link) when no trimming is needed; otherwise link the
    first `keep_n` frames individually.
    """
    if keep_n == n_jpg:
        link_or_copy(src_cam, dst_cam, copy)
        return
    os.makedirs(dst_cam, exist_ok=True)
    for jpg in sorted_jpgs(src_cam)[:keep_n]:
        link_or_copy(jpg, os.path.join(dst_cam, os.path.basename(jpg)), copy)


def build_annotations(seq_dir, cam_dirs, sel):
    """Assemble + view-slice the annotation dict for the selected views."""
    tracks = np.load(os.path.join(seq_dir, TRACKS_FILE), allow_pickle=True).item()
    extr = np.load(os.path.join(seq_dir, EXTR_FILE))                       # (V,4,4)
    intr = np.stack([np.load(os.path.join(seq_dir, c, INTR_FILE))
                     for c in cam_dirs], axis=0)                            # (V,3,3)

    V = len(cam_dirs)
    for key in VIEW_AXIS_TRACK_KEYS:
        if tracks[key].shape[0] != V:
            raise ValueError(f"{seq_dir}: {key} axis-0 {tracks[key].shape[0]} != #cams {V}")
    if extr.shape[0] != V:
        raise ValueError(f"{seq_dir}: extrinsics has {extr.shape[0]} views != #cams {V}")

    ann = {
        "trajectory": tracks["trajectory"][sel],
        "visibility": tracks["visibility"][sel],
        "flow_filter_masks": tracks["flow_filter_masks"][sel],
        "intrinsics": intr[sel],
        "extrinsics": extr[sel],
    }
    # view-independent extra, kept for completeness (loader ignores it)
    if "trajectory_3d" in tracks:
        ann["trajectory_3d"] = tracks["trajectory_3d"]
    return ann, extr


def process_sequence(src_seq, dst_seq, method, k, copy):
    cam_dirs = list_cam_dirs(src_seq)                 # i-th entry <-> view index i
    num_src_views = len(cam_dirs)

    # need extrinsics for camera centers; build_annotations reloads them too (cheap)
    extr = np.load(os.path.join(src_seq, EXTR_FILE))
    centers = camera_centers(extr)
    sel = sample_views(method, centers, k)            # sorted ascending view indices
    sel_cam_dirs = [cam_dirs[i] for i in sel]

    os.makedirs(dst_seq, exist_ok=True)

    # 1) assembled + sliced annotations.npy
    ann, _ = build_annotations(src_seq, cam_dirs, sel)

    # align frame count and track length: data/harmony4d.py derives T from the
    # number of jpgs, so #frames must equal the trajectory time axis.  Some raw
    # sequences are off-by-one (e.g. 60 jpgs vs 59 tracked frames); clamp both to
    # the common minimum and trim the extra frames/tracks.
    track_T = ann["trajectory"].shape[1]
    n_jpgs = [len(sorted_jpgs(os.path.join(src_seq, c))) for c in sel_cam_dirs]
    keep_n = min(track_T, *n_jpgs)
    trimmed = keep_n != track_T or any(n != keep_n for n in n_jpgs)
    if keep_n < track_T:
        ann["trajectory"] = ann["trajectory"][:, :keep_n]
        ann["visibility"] = ann["visibility"][:, :keep_n]
        if "trajectory_3d" in ann:
            ann["trajectory_3d"] = ann["trajectory_3d"][:keep_n]

    np.save(os.path.join(dst_seq, ANNOT_FILE), ann, allow_pickle=True)

    # 2) renumbered cam dirs (cam01..cam{K}) linked/copied from the selected views
    for new_idx, (src_cam, n_jpg) in enumerate(zip(sel_cam_dirs, n_jpgs), start=1):
        link_cam(os.path.join(src_seq, src_cam),
                 os.path.join(dst_seq, f"cam{new_idx:02d}"), keep_n, n_jpg, copy)

    return {
        "selected_view_indices": sel,
        "selected_cam_dirs": sel_cam_dirs,
        "num_source_views": num_src_views,
        "num_frames": keep_n,
        "frames_trimmed": bool(trimmed),
        "camera_centers": {cam_dirs[i]: centers[i].round(5).tolist()
                           for i in range(num_src_views)},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True, choices=["fps", "nearest", "random"])
    ap.add_argument("--num-views", type=int, default=4, help="K: views to keep per sequence")
    ap.add_argument("--src", default="datasets/harmony-multiview",
                    help="raw harmony data_root (default: datasets/harmony-multiview)")
    ap.add_argument("--dst", default=None,
                    help="output root (default: datasets/harmony_sampled/<method>_<K>views)")
    ap.add_argument("--copy", action="store_true",
                    help="copy image dirs instead of symlinking (default: symlink)")
    ap.add_argument("--overwrite", action="store_true",
                    help="remove --dst if it already exists")
    args = ap.parse_args()

    if args.dst is None:
        args.dst = os.path.join("datasets", "harmony_sampled",
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

    for i, (cat, scene) in enumerate(seqs, 1):
        info = process_sequence(
            os.path.join(args.src, cat, scene),
            os.path.join(args.dst, cat, scene),
            args.method, args.num_views, args.copy,
        )
        manifest["sequences"][f"{cat}/{scene}"] = info
        note = f"  [trimmed frames->{info['num_frames']}]" if info["frames_trimmed"] else ""
        print(f"[{i}/{len(seqs)}] {cat}/{scene}: kept {info['selected_cam_dirs']} "
              f"(orig idx {info['selected_view_indices']} of {info['num_source_views']}){note}")

    # dotfile so data/harmony4d.py (which lists data_root without an isdir check)
    # skips it as a category
    manifest_path = os.path.join(args.dst, ".sampling_info.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. {len(seqs)} sequences -> {args.dst}")
    print(f"  method={args.method} num_views={args.num_views} "
          f"({'symlink' if not args.copy else 'copy'})")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
