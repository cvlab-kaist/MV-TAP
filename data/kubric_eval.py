import os
import cv2
import glob
import torch
import numpy as np
import mediapy as media

from typing import Tuple

def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)



class KubricEval(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        resize_to=[256, 256],
        num_points=300,
    ):
        self.data_root = data_root
        self.num_points = num_points
        self.resize_to = resize_to


        self.data_dir = [
            data
            for data in sorted(os.listdir(self.data_root))
        ]

        self.seq_names = []
        for d_dir in self.data_dir:
            frames_dir = os.path.join(self.data_root, d_dir, 'frames')
            if os.path.exists(frames_dir):
                for seq_name in sorted(os.listdir(frames_dir)):
                    self.seq_names.append({
                        'seq_name':seq_name,
                        'data_dir':d_dir
                        })


    def __getitem__(self, index):
        # ============================================= #
        # 1. Data Loading
        # ============================================= #

        data = self.seq_names[index]

        seq_name = data['seq_name']
        data_dir = data['data_dir']

        view_names = [
            vname
            for vname in sorted(os.listdir(os.path.join(self.data_root, data_dir, 'frames', seq_name)))
        ]
        
        
        nrgbs = []
        for i, view in enumerate(view_names):
            rgb_path = os.path.join(self.data_root, data_dir, 'frames', seq_name, view)
            img_paths = sorted(glob.glob(os.path.join(rgb_path, 'rgba_*.png')))
            rgbs = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) for p in img_paths]
            rgbs = np.stack(rgbs, axis=0)
            rgbs = rgbs[..., :3]
            nrgbs.append(rgbs)


        npy_path = os.path.join(self.data_root, data_dir, 'stacked_tracks', seq_name)
        npy_file = os.path.join(npy_path, 'tracks.npy')
        annot = np.load(npy_file, allow_pickle=True).item()
        
        traj_2d = annot['coords']
        occ = annot['occluded']
        bkgd_flag = annot['is_bkgd'][0, :, 0]

        video = np.stack(nrgbs, axis=0)

        cam_rot = annot['cam_rot']            
        cam_trans = annot['cam_trans']        
        intrinsics = annot['intrinsics']  
        

        extrinsic = np.zeros((V, T, 4, 4), dtype=np.float32)
        extrinsic[..., :3, :3] = cam_rot
        extrinsic[..., :3, 3] = cam_trans
        extrinsic[..., 3, 3] = 1.0

        # Change to COLMAP convention
        extrinsic[..., 1:3, :] *= -1.0

        V, N, T, _ = traj_2d.shape

        
        # ============================================= #
        # 2. Dynamic Point Sampling
        # ============================================= #
        
        flag = ~bkgd_flag
        dyn_idx = np.where(flag)[0]

        np.random.shuffle(dyn_idx)
        traj_2d = traj_2d[:, dyn_idx, :, :]
        visibility = occ[:, dyn_idx, :]

        resize_to = self.resize_to
        traj_2d = np.transpose(traj_2d, (0, 2, 1, 3))
        visibility = np.transpose(np.logical_not(visibility), (0, 2, 1))

        V, T, N, D = traj_2d.shape
        _, _, H, W, C = video.shape
        traj_2d = traj_2d.reshape(V * T, N, 2)
        visibility = visibility.reshape(V * T, N)

        video = video.reshape(-1, H, W, 3)
        resize_to = self.resize_to
        video = resize_video(video, (resize_to[0], resize_to[1]))

        scale_x = resize_to[1] / W
        scale_y = resize_to[0] / H
        intrinsics[:, :, 0, :] *= scale_x
        intrinsics[:, :, 1, :] *= scale_y

        intrinsics = torch.from_numpy(intrinsics).float()
        extrinsic = torch.from_numpy(extrinsic).float()
            
        traj_2d *= np.array([self.resize_to[1] / W, self.resize_to[0] / H])


        _, H, W, C = video.shape

        visibility[traj_2d[..., 0] > resize_to[1] - 1] = False
        visibility[traj_2d[..., 0] < 0] = False
        visibility[traj_2d[..., 1] > resize_to[0] - 1] = False
        visibility[traj_2d[..., 1] < 0] = False

        trajs = torch.from_numpy(traj_2d).float()
        visibles = torch.from_numpy(visibility).bool()

        trajs = trajs.reshape(V, T, N, D)
        visibles = visibles.reshape(V, T, N)        
        video = video.reshape(V * T, H, W, C)
        video = torch.from_numpy(video).float()
        video = video.reshape(V, T, H, W, C).permute(0, 1, 4, 2, 3)  


        crop_tensor = torch.tensor(resize_to).flip(0)[None, None] / 2.0
        dists = torch.linalg.vector_norm(trajs[..., :2] - crop_tensor, dim=-1)  # (V, T, N)
        inside_mask = dists < 1000.0  
        close_pts_inds = inside_mask.all(dim=(0, 1)) 

        trajs = trajs[:, :, close_pts_inds]
        visibles = visibles[:, :, close_pts_inds]

        V, T, N, D = trajs.shape

        point_inds = torch.randperm(N)[: self.num_points]
        trajs = trajs[:, :, point_inds]
        visibles = visibles[:, :, point_inds]

        V, T, N, D = trajs.shape
        has_visible = visibles.any(dim=1)  
        first_visible_idx = (visibles.float().cumsum(dim=1) == 1).int()
        t_idx = (first_visible_idx * torch.arange(T, device=visibles.device)[None, :, None]).max(dim=1).values  # (V, N)
        v_idx = torch.arange(V, device=visibles.device)[:, None].expand(V, N)
        n_idx = torch.arange(N, device=visibles.device)[None, :].expand(V, N)
        coords = trajs[v_idx, t_idx, n_idx]  
        t_idx = torch.where(has_visible, t_idx, torch.tensor(T, device=visibles.device))
        queries = torch.stack([
            t_idx.float(),
            coords[..., 0],
            coords[..., 1]
        ], dim=-1)  # (V, N, 3)


        valids = torch.ones_like(visibles)
        return {
            "video": video,
            "trajectory": trajs,
            "visibility": visibles,
            "valid": valids,
            "query_points": queries,
            "seq_name": seq_name,
            "intrinsic": intrinsics,
            "extrinsic": extrinsic,
        }
    
    def __len__(self):
        return len(self.seq_names)