import glob
import os
import random

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter, GaussianBlur

class KubricTrain(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=[256, 256],
        seq_len=24,
        traj_per_sample=768,
        use_augs=False,
        keep_principal_point_centered = False,
    ):    
        super().__init__()
        # aug parameters
        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15

        self.photo_aug = ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14
        )
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5

        self.data_root = data_root
        self.crop_size = crop_size
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.use_augs = use_augs
        self.keep_principal_point_centered = keep_principal_point_centered
        
        self.samples = self._build_index()
        print(f"Found {len(self.samples)} valid sequences in {self.data_root}")

    def add_photometric_augs(self, rgbs, trajs, visibles):

        V, T, N, _ = trajs.shape

        S = len(rgbs[0])
        H, W = rgbs[0].shape[1:3]
        assert S == T


        ############ eraser transform (per image after the first) ############
        for v in range(V):
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):

                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        dy = np.random.randint(
                            self.eraser_bounds[0], self.eraser_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        patch = rgbs[v, i, y0:y1, x0:x1].reshape(-1, 3)

                        mean_color = np.mean(patch, axis=0).astype(rgbs.dtype)
                        rgbs[v, i, y0:y1, x0:x1] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[v, i, :, 0] >= x0, trajs[v, i, :, 0] < x1),
                            np.logical_and(trajs[v, i, :, 1] >= y0, trajs[v, i, :, 1] < y1),
                        )
                        visibles[v, i, occ_inds] = 0

        

        for v in range(V):
            rgbs_view = rgbs[v]  

            # === double photometric augmentation (as in original) ===
            # convert once to uint8 to avoid repeated float copies
            # rgbs_uint8 = rgbs_view.clip(0, 255).astype(np.uint8, copy=False)

            # apply jitter twice in one pass
            rgbs_view_alt = np.empty_like(rgbs_view)
            for t in range(T):
                img = Image.fromarray(rgbs_view[t])
                img = self.photo_aug(img)
                img = self.photo_aug(img)  # double augmentation
                rgbs_view_alt[t] = np.asarray(img, dtype=np.uint8)


            ############ replace transform (per image after the first) ############
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        dy = np.random.randint(
                            self.replace_bounds[0], self.replace_bounds[1]
                        )
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_view_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs_view[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[v,i, :, 0] >= x0, trajs[v,i, :, 0] < x1),
                            np.logical_and(trajs[v,i, :, 1] >= y0, trajs[v,i, :, 1] < y1),
                        )
                        visibles[v, i, occ_inds] = 0
            rgbs[v] = np.clip(rgbs_view, 0, 255).astype(np.uint8, copy=False)


        ############ photometric augmentation ############
        # for v in range(V):
        #     for t in range(T):
        #         if np.random.rand() < self.color_aug_prob:
        #             rgb = rgbs[v,t].clip(0,255).astype(np.uint8)
        #             rgb = np.asarray(self.photo_aug(Image.fromarray(rgb)))
        #             rgbs[v,t] = rgb
        #         if np.random.rand() < self.blur_aug_prob:
        #             rgb = rgbs[v,t].clip(0,255).astype(np.uint8)
        #             rgb = np.asarray(self.blur_aug(Image.fromarray(rgb)))
        #             rgbs[v,t] = rgb
        
        for v in range(V):
            do_grayscale = np.random.rand() < 0.1
            do_color_aug = np.random.rand() < self.color_aug_prob
            do_blur_aug = np.random.rand() < self.blur_aug_prob

            for t in range(T):
                img_np = rgbs[v, t]
                img_pil = Image.fromarray(img_np)
                # [Augmentation] Add Random Grayscale
                if do_grayscale:
                     img_pil = img_pil.convert('L').convert('RGB')

                if do_color_aug:
                    img_pil = self.photo_aug(img_pil)

                if do_blur_aug:
                    # [Augmentation] Switch between Gaussian and Motion Blur
                    img_pil = self.blur_aug(img_pil)

                img_np = np.asarray(img_pil)
                if np.random.rand() < 0.15: # 15% chance
                    noise_sigma = np.random.uniform(5, 20)
                    noise = np.random.normal(0, noise_sigma, img_np.shape).astype(np.int16)
                    img_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)

                rgbs[v, t] = img_np        


        return rgbs, trajs, visibles
    
    def add_spatial_augs(self, rgbs, trajs, visibles, crop_size, intrinsic=None, extrinsic=None):
        V, T, H, W, C = rgbs.shape
        S = T
        ch, cw = crop_size
        
        rgbs_new = np.empty((V, T, ch, cw, 3), dtype=np.uint8)


        pad_x0, pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1], 2)
        pad_y0, pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1], 2)
        H0, W0 = H + pad_y0 + pad_y1, W + pad_x0 + pad_x1

        scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
        sx_cont_seq = np.zeros(S) 
        sy_cont_seq = np.zeros(S)
        H_new_seq = np.zeros(S, dtype=int)
        W_new_seq = np.zeros(S, dtype=int)

        scale_x = scale
        scale_y = scale
        dx = 0.0
        dy = 0.0

        do_h_flip = self.do_flip and (np.random.rand() < self.h_flip_prob)
        do_v_flip = self.do_flip and (np.random.rand() < self.v_flip_prob)

        for s in range(S):
            if s == 1:
                dx = np.random.uniform(-self.resize_delta, self.resize_delta)
                dy = np.random.uniform(-self.resize_delta, self.resize_delta)
            elif s > 1:
                dx = dx * 0.8 + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                dy = dy * 0.8 + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2

            sx_drift = np.clip(scale_x + dx, 0.2, 2.0)
            sy_drift = np.clip(scale_y + dy, 0.2, 2.0)
            
            H_new = int(max(H0 * sy_drift, ch + 10))
            W_new = int(max(W0 * sx_drift, cw + 10))
            H_new_seq[s] = H_new
            W_new_seq[s] = W_new

            sx_cont_seq[s] = W_new / float(W0)
            sy_cont_seq[s] = H_new / float(H0)

            scale_x = sx_drift
            scale_y = sy_drift


        ok_inds = visibles[0, 0, :] > 0 
        if np.sum(ok_inds) > 0:
            trajs_v0_t0 = trajs[0, 0, ok_inds]
            trajs_aug = trajs_v0_t0.copy()
            trajs_aug[:, 0] = (trajs_aug[:, 0] + pad_x0) * sx_cont_seq[0]
            trajs_aug[:, 1] = (trajs_aug[:, 1] + pad_y0) * sy_cont_seq[0]
            mid_x = float(np.mean(trajs_aug[:, 0]))
            mid_y = float(np.mean(trajs_aug[:, 1]))
        else:
            mid_x = (W0 * sx_cont_seq[0]) / 2.0 
            mid_y = (H0 * sy_cont_seq[0]) / 2.0
            
        x0_base = int(mid_x - cw // 2)
        y0_base = int(mid_y - ch // 2)

        off_x_seq = np.zeros(S, dtype=int)
        off_y_seq = np.zeros(S, dtype=int)
        off_x = 0
        off_y = 0
        for s in range(S):
            if s == 1:
                off_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                off_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
            elif s > 1:
                off_x = int(off_x * 0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2)
                off_y = int(off_y * 0.8 + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2)
            off_x_seq[s] = off_x
            off_y_seq[s] = off_y


        for v in range(V):

            rgbs_v = np.pad(rgbs[v], ((0,0),(pad_y0,pad_y1),(pad_x0,pad_x1),(0,0)))

            trajs[v, :, :, 0] += pad_x0
            trajs[v, :, :, 1] += pad_y0
            if intrinsic is not None:
                intrinsic[v, :, 0, 2] += pad_x0
                intrinsic[v, :, 1, 2] += pad_y0


            # imgs_resized = [None] * S 
            for s in range(S):
                W_new, H_new = W_new_seq[s], H_new_seq[s]
                # img_s = cv2.resize(rgbs_v[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
                # imgs_resized[s] = img_s  
                interp = cv2.INTER_LINEAR if (W_new > W0) else cv2.INTER_AREA
                img_s = cv2.resize(rgbs_v[s], (W_new, H_new), interpolation=interp)                

                trajs[v, s, :, 0] *= sx_cont_seq[s]
                trajs[v, s, :, 1] *= sy_cont_seq[s]
                if intrinsic is not None:
                    intrinsic[v, s, 0, :] *= sx_cont_seq[s]
                    intrinsic[v, s, 1, :] *= sy_cont_seq[s]

            # for s in range(S):
                x0 = x0_base + off_x_seq[s]
                y0 = y0_base + off_y_seq[s]
                Hs, Ws = H_new_seq[s], W_new_seq[s]
                
                y0_clipped = int(np.clip(y0, 0, max(0, Hs - ch)))
                x0_clipped = int(np.clip(x0, 0, max(0, Ws - cw)))
                y1 = min(y0_clipped + ch, Hs)
                x1 = min(x0_clipped + cw, Ws)

                crop_rgb = img_s[y0_clipped:y1, x0_clipped:x1]

                h, w = crop_rgb.shape[:2]
                rgbs_new[v, s, :h, :w] = crop_rgb

                if (h < ch) or (w < cw):
                    if h < ch: rgbs_new[v, s, h:ch, :w] = 0
                    if w < cw: rgbs_new[v, s, :h, w:cw] = 0
                    if (h < ch) and (w < cw): rgbs_new[v, s, h:ch, w:cw] = 0
                
                trajs[v, s, :, 0] -= x0_clipped
                trajs[v, s, :, 1] -= y0_clipped
                if intrinsic is not None:
                    intrinsic[v, s, 0, 2] -= x0_clipped
                    intrinsic[v, s, 1, 2] -= y0_clipped

            if do_h_flip:
                rgbs_new[v] = rgbs_new[v, :, :, ::-1]
                trajs[v, :, :, 0] = cw - trajs[v, :, :, 0]
                intrinsic[v, :, 0, 2] = cw - intrinsic[v, :, 0, 2]
                extrinsic[v, :, 1, :] *= -1.0  # flip x axis
            if do_v_flip:    
                rgbs_new[v] = rgbs_new[v, :, ::-1, :]
                trajs[v, :, :, 1] = ch - trajs[v, :, :, 1]
                intrinsic[v, :, 1, 2] = ch - intrinsic[v, :, 1, 2]
                extrinsic[v, :, 0, :] *= -1.0  # flip y axis


        return rgbs_new, trajs, intrinsic, extrinsic


    def _build_index(self):
        samples = []
        if not os.path.exists(self.data_root):
            return samples

        data_dirs = sorted(os.listdir(self.data_root))
        
        for d_dir in data_dirs:
            full_d_dir = os.path.join(self.data_root, d_dir)
            frames_dir = os.path.join(full_d_dir, 'frames')
            
            if not os.path.exists(frames_dir):
                continue
                
            seq_names = sorted(os.listdir(frames_dir))
            for seq_name in seq_names:
                seq_path = os.path.join(frames_dir, seq_name)
                views = sorted(os.listdir(seq_path))
                
                view_paths = {}
                valid_seq = True
                
                for view in views:
                    view_dir = os.path.join(seq_path, view)
                    imgs = sorted([
                        os.path.join(view_dir, f) 
                        for f in os.listdir(view_dir) 
                        if f.startswith('rgba_') and f.endswith('.png')
                    ])
                    if not imgs:
                        valid_seq = False
                        break
                    view_paths[view] = imgs
                
                if valid_seq:
                    samples.append({
                        'data_dir': d_dir,
                        'seq_name': seq_name,
                        'view_paths': view_paths,
                        'root_path': full_d_dir
                    })
        return samples
    
    def __getitem__(self, index):
        # ============================================= #
        # 1. Data Loading
        # ============================================= #

        sample_info = self.samples[index]
        seq_name = sample_info['seq_name']
        data_dir = sample_info['data_dir']
        view_paths = sample_info['view_paths']
        root_path = sample_info['root_path']

        view_names = [
            vname
            for vname in sorted(os.listdir(os.path.join(self.data_root, data_dir, 'frames', seq_name)))
        ]
        
        
        nrgbs = []
        for i, view in enumerate(view_names):
            paths = view_paths[view]
            # Batch read could be parallelized but seq read is okay on SSD
            rgbs = [cv2.cvtColor(cv2.imread(p, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) for p in paths]
            # List -> Numpy Stack
            nrgbs.append(np.stack(rgbs, axis=0))

        npy_path = os.path.join(root_path, 'stacked_tracks', seq_name)
        npy_file = os.path.join(npy_path, 'tracks.npy')
        npz_file = os.path.join(npy_path, 'tracks.npz')

        if os.path.exists(npy_file):
            annot = np.load(npy_file, allow_pickle=True).item()
        elif os.path.exists(npz_file):
            annot = dict(np.load(npz_file, allow_pickle=True))
        else:
            raise FileNotFoundError(f"No .npy or .npz file found at {npy_path}")
        
        traj_2d = annot['coords']
        occ = annot['occluded']
        bkgd_flag = annot['is_bkgd'][0, :, 0]

        video = np.stack(nrgbs, axis=0)
        V, N, T, _ = traj_2d.shape

        cam_rot = annot['cam_rot']              # V 3 3
        cam_trans = annot['cam_trans']          # V 3
        intrinsics = annot['intrinsics']        # V T 3 3

        extrinsic = np.zeros((V, T, 4, 4), dtype=np.float32)
        extrinsic[..., :3, :3] = cam_rot
        extrinsic[..., :3, 3] = cam_trans
        extrinsic[..., 3, 3] = 1.0

        extrinsic[..., 1:3, :] *= -1.0

        n_view = np.random.randint(1, 5)   
        
        view_indices = np.arange(V)
        random.shuffle(view_indices)
        sel_indices = view_indices[:n_view]
        

        video = video[sel_indices, ...]
        traj_2d = traj_2d[sel_indices, ...]
        occ = occ[sel_indices, ...]
        intrinsics = intrinsics[sel_indices, ...]
        extrinsic = extrinsic[sel_indices, ...]
        V, N, T, _ = traj_2d.shape

        
        # ============================================= #
        # 2. Dynamic Point Sampling
        # ============================================= #
        
        flag = ~bkgd_flag
        dyn_idx = np.where(flag)[0]

        np.random.shuffle(dyn_idx)
        traj_2d = traj_2d[:, dyn_idx, :, :]
        visibility = occ[:, dyn_idx, :]
        
        
        crop_size = self.crop_size
        traj_2d = np.transpose(traj_2d, (0, 2, 1, 3))
        visibility = np.transpose(np.logical_not(visibility), (0, 2, 1))
        V, T, N, D = traj_2d.shape
        _, _, H, W, C = video.shape


        center = np.array([W / 2.0, H / 2.0])[None, None, None, :]      # (1,1,1,2)
        dists = np.linalg.norm(traj_2d - center, axis=-1)               # (V, T, N)
        inside_mask = dists < 1000.0             
        close_pts_inds = np.all(inside_mask, axis=(0, 1))   

        traj_2d = traj_2d[:, :, close_pts_inds]
        visibility = visibility[:, :, close_pts_inds]


        V, T, N, _ = traj_2d.shape
        if N == 0:
            print(f"[Warning] Empty track detected in seq {seq_name} from {data_dir}")
            new_index = np.random.randint(0, len(self.seq_names))
            return self.__getitem__(new_index)
        if N < self.traj_per_sample:
            print(f"[Warning] Less track detected in seq {seq_name} ({N} < {self.traj_per_sample})")
            point_inds = np.random.choice(N, self.traj_per_sample, replace=True)
            raise NotImplementedError
        else:
            point_inds = np.random.choice(N, self.traj_per_sample, replace=False)

            traj_2d = traj_2d[:, :, point_inds]
            visibility = visibility[:, :, point_inds]

        # ============================================= #
        # 3. Augmentations
        # ============================================= #

        crop_size = self.crop_size

        if self.use_augs:
            if self.keep_principal_point_centered:
                # video, traj_2d, visibility = self.add_photometric_augs(video, traj_2d, visibility)
                # video, traj_2d, intrinsics = self.add_spatial_augs_pp_center(video, traj_2d, crop_size, intrinsics)
                raise NotImplementedError("Not implemented yet.")
            else:
                video, traj_2d, visibility = self.add_photometric_augs(video, traj_2d, visibility)
                video, traj_2d, intrinsics, extrinsic = self.add_spatial_augs(video, traj_2d, visibility, crop_size, intrinsics, extrinsic)

        _, _, H, W, C = video.shape

        if self.use_augs:
            visibility[traj_2d[..., 0] > crop_size[1] - 1] = False
            visibility[traj_2d[..., 0] < 0] = False
            visibility[traj_2d[..., 1] > crop_size[0] - 1] = False
            visibility[traj_2d[..., 1] < 0] = False


        trajs = torch.from_numpy(traj_2d).float()
        visibles = torch.from_numpy(visibility).bool()

        video = torch.from_numpy(video).float()
        video = video.reshape(V, T, H, W, C).permute(0, 1, 4, 2, 3)  
        intrinsics = torch.from_numpy(intrinsics).float()
        extrinsic = torch.from_numpy(extrinsic).float()
            

        valids = None
        queries = None

        V, T, N, D = trajs.shape


        # ============================================= #
        # 3. Query Construction
        # ============================================= #

        has_visible = visibles.any(dim=1)  
        first_visible_idx = (visibles.float().cumsum(dim=1) == 1).int()
        t_idx = (first_visible_idx * torch.arange(T, device=visibles.device)[None, :, None]).max(dim=1).values  # (V, N)

        v_idx = torch.arange(V, device=visibles.device)[:, None].expand(V, N)
        n_idx = torch.arange(N, device=visibles.device)[None, :].expand(V, N)
        coords = trajs[v_idx, t_idx, n_idx]  # (V, N, 2)
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
        return len(self.samples)    