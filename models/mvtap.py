import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Refiner
from models.blocks import Mlp, BasicEncoder
from models.cam_embed import get_plucker_coords
from models.embeddings import get_1d_sincos_pos_embed_from_grid
from models.utils import sample_features5d, bilinear_sampler, get_points_on_a_grid

def posenc(x, min_deg, max_deg):
    """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).
    Args:
      x: torch.Tensor, variables to be encoded. Note that x should be in [-pi, pi].
      min_deg: int, the minimum (inclusive) degree of the encoding.
      max_deg: int, the maximum (exclusive) degree of the encoding.
      legacy_posenc_order: bool, keep the same ordering as the original tf code.
    Returns:
      encoded: torch.Tensor, encoded variables.
    """
    if min_deg == max_deg:
        return x
    scales = torch.tensor(
        [2**i for i in range(min_deg, max_deg)], dtype=x.dtype, device=x.device
    )

    xb = (x[..., None, :] * scales[:, None]).reshape(list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)

class MVTAP(nn.Module):
    def __init__(
        self,
        window_len: int = 16,
        stride: int = 4,
        corr_radius: int = 3,
        corr_levels: int = 4,
        num_virtual_tracks: int = 64,
        hidden_dim: int =  256,                  
        latent_dim: int =  128,       
        model_resolution_H: int =  384,
        model_resolution_W: int =  512,
        use_checkpoint: bool = True,
        view_att: bool = False,
        use_cam_embed: bool = False,
        bilinear_mode: str = None,
    ):
        super().__init__()

        self.window_len = window_len
        self.stride = stride
        self.corr_radius = corr_radius
        self.corr_levels = corr_levels 
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_virtual_tracks = num_virtual_tracks
        self.model_resolution_H = model_resolution_H
        self.model_resolution_W = model_resolution_W
        self.former_input_dim = corr_levels*hidden_dim + 86
        self.use_checkpoint = use_checkpoint
        self.view_att = view_att
        self.use_cam_embed = use_cam_embed
        self.bilinear_mode = bilinear_mode

        self.fnet = BasicEncoder(input_dim=3, output_dim=latent_dim, stride=self.stride)
        self.corr_mlp = Mlp(in_features=49 * 49, hidden_features=384, out_features=256)

        time_grid = torch.linspace(0, window_len - 1, window_len).reshape(
            1, window_len, 1
        )

        self.register_buffer(
            "time_emb", get_1d_sincos_pos_embed_from_grid(self.former_input_dim, time_grid[0])
        )        
        
        self.updateformer = Refiner(
            space_depth=3,
            time_depth=3,
            view_depth=3,
            input_dim=self.former_input_dim,
            hidden_size=384,
            output_dim=4,
            mlp_ratio=4.0,
            num_virtual_tracks=self.num_virtual_tracks,
            linear_layer_for_vis_conf=True,
            use_checkpoint=self.use_checkpoint,
            view_att=self.view_att,
        )            
        
        if self.use_cam_embed:
            # plucker embedding dim -> refiner input dim
            self.cam_proj = nn.Linear(6, self.former_input_dim)
            self.cam_norm = nn.LayerNorm(self.former_input_dim)    
     

    def get_support_points(self, coords, r, reshape_back=True):
        B, _, N, _ = coords.shape
        device = coords.device
        centroid_lvl = coords.reshape(B, N, 1, 1, 3)

        dx = torch.linspace(-r, r, 2 * r + 1, device=device)
        dy = torch.linspace(-r, r, 2 * r + 1, device=device)

        xgrid, ygrid = torch.meshgrid(dy, dx, indexing="ij")
        zgrid = torch.zeros_like(xgrid, device=device)
        delta = torch.stack([zgrid, xgrid, ygrid], axis=-1)
        delta_lvl = delta.reshape(1, 1, 2 * r + 1, 2 * r + 1, 3)
        coords_lvl = centroid_lvl + delta_lvl

        if reshape_back:
            return coords_lvl.reshape(B, N, (2 * r + 1) ** 2, 3).permute(0, 2, 1, 3)
        else:
            return coords_lvl
    
    def get_track_feat(self, fmaps, queried_frames, queried_coords, support_radius=0):

        sample_frames = queried_frames[:, None, :, None]
        sample_coords = torch.cat(
            [
                sample_frames,
                queried_coords[:, None],
            ],
            dim=-1,
        )
        support_points = self.get_support_points(sample_coords, support_radius)
        support_track_feats = sample_features5d(fmaps, support_points, mode=self.bilinear_mode)
        return (
            support_track_feats[:, None, support_track_feats.shape[1] // 2],
            support_track_feats,
        )        

    def get_correlation_feat(self, fmaps, queried_coords):
        B, T, D, H_, W_ = fmaps.shape
        N = queried_coords.shape[1]
        r = self.corr_radius
        sample_coords = torch.cat(
            [torch.zeros_like(queried_coords[..., :1]), queried_coords], dim=-1
        )[:, None]
        support_points = self.get_support_points(sample_coords, r, reshape_back=False)
        
        if self.bilinear_mode == "border":
            correlation_feat = bilinear_sampler(
                fmaps.reshape(B * T, D, 1, H_, W_), support_points
            )
        elif self.bilinear_mode == "zeros":
            correlation_feat = bilinear_sampler(
                fmaps.reshape(B * T, D, 1, H_, W_), support_points, padding_mode="zeros"
            )
        else:
            raise NotImplementedError
        
        return correlation_feat.reshape(B, T, D, N, (2 * r + 1), (2 * r + 1)).permute(
            0, 1, 3, 4, 5, 2
        )        

    def interpolate_time_embed(self, x, t):
        previous_dtype = x.dtype
        T = self.time_emb.shape[1]

        if t == T:
            return self.time_emb

        time_emb = self.time_emb.float()
        time_emb = F.interpolate(
            time_emb.permute(0, 2, 1), size=t, mode="linear"
        ).permute(0, 2, 1)
        return time_emb.to(previous_dtype)


    def forward_window(
        self,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        vis=None,
        conf=None,
        iters=4,
        intrinsic=None,
        extrinsic=None,
    ):
        
        r = 2 * self.corr_radius + 1
        B, S, N, _ = coords.shape
        

        coord_preds, vis_preds, conf_preds = [], [], []
        for it in range(iters):
            view_att_mask = None
                
            coords = coords.detach()  # B T N 2
            coords_init = coords.reshape(B * S, N, 2)
            corr_embs = []

            for i in range(self.corr_levels):
                corr_feat = self.get_correlation_feat(
                    fmaps_pyramid[i], coords_init / 2**i
                )
                track_feat_support = (
                    track_feat_support_pyramid[i]
                    .reshape(B, 1, r, r, N, self.latent_dim)
                    .squeeze(1)
                    .permute(0, 3, 1, 2, 4)
                )
                corr_volume = torch.einsum(
                    "btnhwc,bnijc->btnhwij", corr_feat, track_feat_support
                )
                corr_emb = self.corr_mlp(corr_volume.reshape(B * S * N, r * r * r * r))

                corr_embs.append(corr_emb)

            corr_embs = torch.cat(corr_embs, dim=-1)
            corr_embs = corr_embs.reshape(B, S, N, corr_embs.shape[-1])

            transformer_input = [vis, conf, corr_embs]

            rel_coords_forward = coords[:, :-1] - coords[:, 1:]
            rel_coords_backward = coords[:, 1:] - coords[:, :-1]

            rel_coords_forward = torch.nn.functional.pad(
                rel_coords_forward, (0, 0, 0, 0, 0, 1)
            )
            rel_coords_backward = torch.nn.functional.pad(
                rel_coords_backward, (0, 0, 0, 0, 1, 0)
            )

            scale = (
                torch.tensor(
                    [self.model_resolution_W, self.model_resolution_H],
                    device=coords.device,
                )
                / self.stride
            )

            rel_coords_forward = rel_coords_forward / scale
            rel_coords_backward = rel_coords_backward / scale

            rel_pos_emb_input = posenc(
                torch.cat([rel_coords_forward, rel_coords_backward], dim=-1),
                min_deg=0,
                max_deg=10,
            )
            transformer_input.append(rel_pos_emb_input)

            x = (
                torch.cat(transformer_input, dim=-1)
                .permute(0, 2, 1, 3)
                .reshape(B * N, S, -1)
            )

            

            if self.use_cam_embed: # cam_embed ; B 6 S H W
                extrinsic_ = extrinsic.clone()
                extrinsic_ = extrinsic_.squeeze(0)  # 1 V T 4 4 -> V T 4 4

                intrinsic_ = intrinsic.clone()
                intrinsic_ = intrinsic_.squeeze(0)  # 1 V T 3 3 -> V T 3 3

                extrinsic_ = extrinsic_.permute(1, 0, 2, 3).reshape(-1, 4, 4)  # (T*V) 4 4
                intrinsic_ = intrinsic_.permute(1, 0, 2, 3).reshape(-1, 3, 3)  # (T*V) 3 3

                cam_embed = get_plucker_coords(
                    coords.permute(1, 0, 2, 3) *float(self.stride),  # S B N 2
                    intrinsic_,
                    extrinsic_,
                    normalize=True,
                )   # B N S 6

                _, _, _, C = cam_embed.shape

                x = x + self.interpolate_time_embed(x, S)
                x = x.reshape(B, N, S, -1)  # (B N) T D -> B N T D

                cam_embed = self.cam_proj(cam_embed.reshape(-1, C))
                cam_embed = F.normalize(cam_embed, dim=-1)
                cam_embed = self.cam_norm(cam_embed)
                cam_embed = cam_embed.reshape(B, N, S, -1)
                x = x + cam_embed
            else:
                x = x + self.interpolate_time_embed(x, S)
                x = x.reshape(B, N, S, -1)  # (B N) T D -> B N T D


            delta = self.updateformer(x, view_att_mask=view_att_mask)

            delta_coords = delta[..., :2].permute(0, 2, 1, 3)
            delta_vis = delta[..., 2:3].permute(0, 2, 1, 3)
            delta_conf = delta[..., 3:].permute(0, 2, 1, 3)

            vis = vis + delta_vis
            conf = conf + delta_conf

            coords = coords + delta_coords
            coord_preds.append(coords[..., :2] * float(self.stride))

            vis_preds.append(vis[..., 0])
            conf_preds.append(conf[..., 0])
        return coord_preds, vis_preds, conf_preds
    
    def forward(
        self,
        video,
        queries,
        intrinsic,
        extrinsic,
        iters=4,
        is_train=False,
        fmaps_chunk_size=8,
        supp_points=None,
    ):
        """Predict tracks

        Args:
            video (FloatTensor[B, T, 3]): input videos.
            queries (FloatTensor[B, N, 3]): point queries.
            iters (int, optional): number of updates. Defaults to 4.
            is_train (bool, optional): enables training mode. Defaults to False.
            intrinsic (FloatTensor[B, V, T, 3, 3], optional): camera intrinsics. Defaults to None.
            extrinsic (FloatTensor[B, V, T, 4, 4], optional): camera extrinsics. Defaults to None.
            fmaps_chunk_size (int, optional): chunk size for feature map extraction. Defaults to 8.
            supp_points (int, optional): number of supplementary points. Defaults to None.

        Returns:
            - coords_predicted (FloatTensor[B, T, N, 2]):
            - vis_predicted (FloatTensor[B, T, N]):
            - train_data: `None` if `is_train` is false, otherwise:
                - all_vis_predictions (List[FloatTensor[B, S, N, 1]]):
                - all_coords_predictions (List[FloatTensor[B, S, N, 2]]):
                - mask (BoolTensor[B, T, N]):

        B = batch size
        S_trimmed = actual number of frames in the window
        N = number of tracks
        C = color channels (3 for RGB)
        E = positional embedding size
        LRR = local receptive field radius
        D = dimension of the transformer input tokens

        video = B T C H W
        queries = B N 3
        coords_init = B T N 2
        vis_init = B T N 1
        """
        B, V, N, __ = queries.shape                  
        B, V, T, C, H, W = video.shape
        device = queries.device
        assert H % self.stride == 0 and W % self.stride == 0
        S = self.window_len
        assert S >= 2  
        step = S // 2  

        video = 2 * (video / 255.0) - 1.0
        pad = ((S - T % S) % S)  

        if supp_points is not None and supp_points > 0:
            
            xy = get_points_on_a_grid(supp_points, video.shape[4 :])
            xy = torch.cat([torch.zeros_like(xy[:, :, :1]), xy], dim=2).to(
                device
            )
            xy = xy.unsqueeze(1).expand(-1, V, -1, -1)  # B V N 3
            queries = torch.cat([queries, xy], dim=2)  
            eval_N = N
            _, _, N, __ = queries.shape       

        video = video.reshape(B, V, T, C * H * W)
        if pad > 0:
            padding_tensor = video[:, :, -1:, :].expand(B, V, pad, C * H * W)
            video = torch.cat([video, padding_tensor], dim=2)

        video = video.reshape(B, V, -1, C, H, W)
        T_pad = video.shape[2]

        # We store our predictions here
        coords_predicted = torch.zeros((B*V, T, N, 2), device=device)
        vis_predicted = torch.zeros((B*V, T, N), device=device)
        conf_predicted = torch.zeros((B*V, T, N), device=device)

        # We store our predictions here
        all_coords_predictions, all_vis_predictions, all_confidence_predictions = (
            [],
            [],
            [],
        )

        H4, W4 = H // self.stride, W // self.stride


        if T_pad > fmaps_chunk_size:
            fmaps_list = []
            for ch_t in range(0, T_pad, fmaps_chunk_size):
                ch_end = min(ch_t + fmaps_chunk_size, T_pad)
                vid_ch = video[:, :, ch_t:ch_end]
                ch_size = vid_ch.shape[2]

                if ch_size == 0:
                    continue

                fmaps_ch = self.fnet(vid_ch.reshape(-1, C, H, W))      
                fmaps_ch = fmaps_ch.reshape(B*V, ch_size, -1, H4, W4)
                
                fmaps_list.append(fmaps_ch)
            fmaps = torch.cat(fmaps_list, dim=1)
            fmaps = fmaps.reshape(-1, self.latent_dim, H4, W4)

        else:    
            fmaps = self.fnet(video.reshape(-1, C, H, W))      
            
        fmaps = fmaps.permute(0, 2, 3, 1)
        fmaps = F.normalize(fmaps, dim=-1, eps=1e-12, p=2)
        fmaps = fmaps.permute(0, 3, 1, 2).reshape(
            B*V, -1, self.latent_dim, H4, W4
        )
        fmaps_pyramid = []

        fmaps_pyramid.append(fmaps)
        for i in range(self.corr_levels - 1):
            fmaps_ = fmaps.reshape(
                B*V*T_pad, self.latent_dim, fmaps.shape[-2], fmaps.shape[-1]
            )
            fmaps_ = F.avg_pool2d(fmaps_, 2, stride=2)
            fmaps = fmaps_.reshape(
                -1, T_pad, self.latent_dim, fmaps_.shape[-2], fmaps_.shape[-1]
            )
            fmaps_pyramid.append(fmaps)

        queries[..., 1:3] /= self.stride
        queried_coords = queries[..., 1:3].reshape(-1, N, 2)
        queried_frames = queries[..., 0].long().reshape(-1, N)


        track_feat_pyramid = []     
        track_feat_support_pyramid = []
        for i in range(self.corr_levels):
            track_feat, track_feat_support = self.get_track_feat(
                fmaps_pyramid[i],
                queried_frames,
                queried_coords / 2**i,
                support_radius=self.corr_radius,
            )

            track_feat_support_pyramid.append(track_feat_support.unsqueeze(1))
            track_feat_pyramid.append(track_feat)


        vis_init = torch.zeros((B*V, S, N, 1), device=device).float()
        conf_init = torch.zeros((B*V, S, N, 1), device=device).float()

        coords_init = queried_coords.unsqueeze(1).expand(-1, S, -1, -1).float()  # B N 2 -> B 1 N 2 -> B S N 2

        num_windows = (T - S + step - 1) // step + 1        

        
        indices = range(0, step * num_windows, step)


        for ind in indices:
            if ind > 0:
                overlap = S - step
                copy_over = (queried_frames < ind + overlap)[
                    :, None, :, None
                ]  # B 1 N 1
                coords_prev = coords_predicted[:, ind : ind + overlap] / self.stride
                padding_tensor = coords_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                coords_prev = torch.cat([coords_prev, padding_tensor], dim=1)

                vis_prev = vis_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = vis_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                vis_prev = torch.cat([vis_prev, padding_tensor], dim=1)

                conf_prev = conf_predicted[:, ind : ind + overlap, :, None].clone()
                padding_tensor = conf_prev[:, -1:, :, :].expand(-1, step, -1, -1)
                conf_prev = torch.cat([conf_prev, padding_tensor], dim=1)

                coords_init = torch.where(
                    copy_over.expand_as(coords_init), coords_prev, coords_init
                )
                vis_init = torch.where(
                    copy_over.expand_as(vis_init), vis_prev, vis_init
                )
                conf_init = torch.where(
                    copy_over.expand_as(conf_init), conf_prev, conf_init
                )

            attention_mask = (queried_frames < ind + S).reshape(B*V, 1, N)  # B S N

            intrinsic_win = None
            extrinsic_win = None
            if intrinsic is not None and extrinsic is not None:
                intrinsic_win = intrinsic[:, :, ind : ind + S]
                extrinsic_win = extrinsic[:, :, ind : ind + S]

                if intrinsic_win.shape[2] < S:
                    pad_size = S - intrinsic_win.shape[2]
                    intrinsic_padding = intrinsic_win[:, :, -1:].expand(
                        B, V, pad_size, 3, 3
                    )
                    intrinsic_win = torch.cat([intrinsic_win, intrinsic_padding], dim=2)

                    extrinsic_padding = extrinsic_win[:, :, -1:].expand(
                        B, V, pad_size, 4, 4
                    )
                    extrinsic_win = torch.cat([extrinsic_win, extrinsic_padding], dim=2)
            
            
            coords, viss, confs = self.forward_window(
                fmaps_pyramid=(
                    [fmap[:, ind : ind + S] for fmap in fmaps_pyramid]
                ),
                coords=coords_init,
                track_feat_support_pyramid=[
                    attention_mask[:, None, :, :, None] * tfeat
                    for tfeat in track_feat_support_pyramid
                ],
                vis=vis_init,
                conf=conf_init,
                iters=iters,
                intrinsic=intrinsic_win,
                extrinsic=extrinsic_win,
            )

            S_trimmed = (
                min(T - ind, S)
            )  


            coords_predicted[:, ind : ind + S] = coords[-1][:, :S_trimmed]
            vis_predicted[:, ind : ind + S] = viss[-1][:, :S_trimmed]
            conf_predicted[:, ind : ind + S] = confs[-1][:, :S_trimmed]
        
            if is_train:
                all_coords_predictions.append(
                    [coord[:, :S_trimmed] for coord in coords]
                )
                all_vis_predictions.append(
                    [torch.sigmoid(vis[:, :S_trimmed]) for vis in viss]
                )
                all_confidence_predictions.append(
                    [torch.sigmoid(conf[:, :S_trimmed]) for conf in confs]
                )

        vis_predicted = torch.sigmoid(vis_predicted)
        conf_predicted = torch.sigmoid(conf_predicted)

        
        coords_predicted = coords_predicted.reshape(B, V, -1, N, 2)
        vis_predicted = vis_predicted.reshape(B, V, -1, N)
        conf_predicted = conf_predicted.reshape(B, V, -1, N)

        if supp_points is not None and supp_points > 0:
            coords_predicted = coords_predicted[:, :, :, :eval_N]
            vis_predicted = vis_predicted[:, :, :, :eval_N]
            conf_predicted = conf_predicted[:, :, :, :eval_N]


        if is_train:
            valid_mask = (
                queried_frames[:, None]
                <= torch.arange(0, T, device=device)[None, None, :, None]
            )
            train_data = (
                all_coords_predictions,
                all_vis_predictions,
                all_confidence_predictions,
                valid_mask,
            )
        else:
            train_data = None

        return coords_predicted, vis_predicted, conf_predicted, train_data
        # B V T N 2, B V T N, B V T N