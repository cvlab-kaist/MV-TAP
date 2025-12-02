import glob
import os
from typing import Tuple

import torch
import imageio.v2 as iio
import mediapy as media
import numpy as np
from einops import repeat


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)

class Panoptic(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        resize_to = [256, 256],
        num_points: int = 300,
    ):
        self.data_root = data_root
        self.resize_to = resize_to
        self.num_points = num_points

        seq_paths = [
            os.path.join(data_root, fname)
            for fname in sorted(os.listdir(data_root))
            if not fname.startswith("_cache") and not fname.startswith(".") and os.path.isdir(os.path.join(data_root, fname))
        ]

        self.samples = []
        
        for seq_path in seq_paths:
            view_pattern = 'ims/*'
            key_func = lambda x: int(os.path.basename(x))

            all_view_paths = sorted(glob.glob(os.path.join(seq_path, view_pattern)), key=key_func)

            self.samples.append({
                "seq_path": seq_path,
                "all_view_paths": all_view_paths,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        seq_path = sample["seq_path"]
        seq_name = os.path.basename(seq_path)
        all_view_paths = sample["all_view_paths"]
        
        npz_filename = 'tapvid3d_annotations.npz'
        tracks_key = 'trajectories_pixelspace'
        visibility_key = 'per_view_visibilities'
        image_pattern = '*.jpg'
            
        nrgbs = []

        for i, view_path in enumerate(all_view_paths):
            img_list = glob.glob(os.path.join(view_path, image_pattern))
            img_list_sorted = sorted(img_list, key=lambda x: int(os.path.basename(x).replace('rgba_', '').split('.')[0]))
            rgbs = [iio.imread(p) for p in img_list_sorted]
            rgbs = np.stack(rgbs, axis=0)
            rgbs = rgbs[..., :3]
            nrgbs.append(rgbs)
            
        frames = np.stack(nrgbs, axis=0)

        V, T, H, W, C = frames.shape
        npz_path = os.path.join(seq_path, npz_filename)
        data = np.load(npz_path, allow_pickle=True)
        trajs = data[tracks_key]
        visibles = data[visibility_key]
        
        
        _, _, N, _ = trajs.shape

        intrinsics = data['intrinsics']  
        extrinsics = data['extrinsics']  

        # scaling
        scale_x = self.resize_to[1] / W
        scale_y = self.resize_to[0] / H
        intrinsics[:, 0, :] *= scale_x  # cx
        intrinsics[:, 1, :] *= scale_y  # cy

        intrinsic = torch.from_numpy(intrinsics).float()
        extrinsic = torch.from_numpy(extrinsics).float()

        intrinsics = repeat(intrinsic, 'v h w -> v t h w', t=T)
        extrinsics = repeat(extrinsic, 'v h w -> v t h w', t=T)

        frames = frames.reshape(-1, H, W, C)
        frames = resize_video(frames, (self.resize_to[0], self.resize_to[1]))
        frames = frames.reshape(V, T, self.resize_to[0], self.resize_to[1], C)
        trajs *= np.array([self.resize_to[1] / W, self.resize_to[0] / H])
        rgbs = torch.from_numpy(frames).permute(0, 1, 4, 2, 3).float()

        trajs = torch.from_numpy(trajs).float()
        visibles = torch.from_numpy(visibles).bool()
        nan_mask = torch.isnan(trajs).any(dim=-1)  # shape: (V, N, T)
        nan_mask_expanded = nan_mask.unsqueeze(-1)  # (V, N, T, 1)
        trajs = torch.where(nan_mask_expanded, trajs.new_tensor([-50.0, -50.0]), trajs)


        
        has_visible = visibles.any(dim=1)  
        first_visible_idx = (visibles.float().cumsum(dim=1) == 1).int()  
        t_idx = (first_visible_idx * torch.arange(T)[None, :, None]).max(dim=1).values
        
        v_idx = torch.arange(V)[:, None].expand(V, N)
        n_idx = torch.arange(N)[None, :].expand(V, N)
        coords = trajs[v_idx, t_idx, n_idx]  # (V, N, 2)


        t_idx = torch.where(has_visible, t_idx, torch.tensor(T, device=visibles.device))
        queries = torch.stack([
            t_idx.float(),
            coords[..., 0],
            coords[..., 1]
        ], dim=-1)




        if self.num_points is not None:
            V, T, N, _ = trajs.shape
            n_sample = min(int(self.num_points), int(N))
            sel_pts = torch.randperm(N)[:n_sample]
            trajs = trajs[:, :, sel_pts, :]
            visibles = visibles[:, :, sel_pts]
            queries = queries[:, sel_pts]
        


        return {
            "video": rgbs.float(),
            "trajectory": trajs.float(),
            "visibility": visibles.bool(),
            "seq_name": str(seq_name),
            "query_points": queries,
            "valid": None,
            "intrinsic": intrinsics,
            "extrinsic": extrinsics,
        }

        
