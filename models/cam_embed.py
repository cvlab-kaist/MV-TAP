# construct flucker embbedidng
# code from uni3c which treats video


import einops
import numpy as np
import torch
import torch.nn.functional as F


@torch.amp.autocast("cuda", enabled=False)
def batch_sample_rays(intrinsic, extrinsic, image_h=None, image_w=None):
    ''' get rays
    Args:
        intrinsic: [BF, 3, 3],
        extrinsic: [BF, 4, 4],
        h, w: int
        # normalize: let the first camera R=I
    Returns:
        rays_o, rays_d: [BF, N, 3]
    '''

    # FIXME: PPU does not support inverse in GPU
    device = intrinsic.device
    B = intrinsic.shape[0]

    c2w = torch.inverse(extrinsic)[:, :3, :4].to(device)  # [BF,3,4]
    x = torch.arange(image_w, device=device).float() - 0.5
    y = torch.arange(image_h, device=device).float() + 0.5
    points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
    points = einops.repeat(points, 'w h c -> b (h w) c', b=B)
    points = torch.cat([points, torch.ones_like(points)[:, :, 0:1]], dim=-1)
    directions = points @ intrinsic.inverse().to(device).transpose(-1, -2) * 1  # depth is 1

    rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)  # [BF,N,3]
    rays_o = c2w[..., :3, 3]  # [BF, 3]

    rays_o = rays_o[:, None, :].expand_as(rays_d)  # [BF, N, 3]

    return rays_o, rays_d


@torch.amp.autocast("cuda", enabled=False)
def embed_rays(rays_o, rays_d, nframe):
    if len(rays_o.shape) == 4:  # [b,f,n,3]
        rays_o = einops.rearrange(rays_o, "b f n c -> (b f) n c")
        rays_d = einops.rearrange(rays_d, "b f n c -> (b f) n c")
    cross_od = torch.cross(rays_o, rays_d, dim=-1)
    cam_emb = torch.cat([rays_d, cross_od], dim=-1)
    cam_emb = einops.rearrange(cam_emb, "(b f) n c -> b f n c", f=nframe)
    return cam_emb


@torch.amp.autocast("cuda", enabled=False)
def camera_center_normalization(w2c, nframe, camera_scale=2.0):
    # copy from SEVA, w2c: [BF, 4, 4]
    # ensure the first view is eye matrix
    c2w_view0 = w2c[::nframe].inverse()  # [B,4,4]
    c2w_view0 = c2w_view0.repeat_interleave(nframe, dim=0)  # [BF,4,4]
    w2c = c2w_view0 @ w2c

    # camera centering
    c2w = torch.linalg.inv(w2c)
    camera_dist_2med = torch.norm(c2w[:, :3, 3] - c2w[:, :3, 3].median(0, keepdim=True).values, dim=-1)
    valid_mask = camera_dist_2med <= torch.clamp(torch.quantile(camera_dist_2med, 0.97) * 10, max=1e6)
    c2w[:, :3, 3] -= c2w[valid_mask, :3, 3].mean(0, keepdim=True)
    w2c = torch.linalg.inv(c2w)

    # camera normalization
    camera_dists = c2w[:, :3, 3].clone()
    translation_scaling_factor = (
        camera_scale
        if torch.isclose(
            torch.norm(camera_dists[0]),
            torch.zeros(1, dtype=camera_dists.dtype, device=camera_dists.device),
            atol=1e-5,
        ).any()
        else (camera_scale / torch.norm(camera_dists[0]))
    )
    w2c[:, :3, 3] *= translation_scaling_factor
    c2w[:, :3, 3] *= translation_scaling_factor

    return w2c


def get_camera_embedding(intrinsic, extrinsic, f, h, w, normalize=True):
    if normalize:
        extrinsic = camera_center_normalization(extrinsic, nframe=f)

    rays_o, rays_d = batch_sample_rays(intrinsic, extrinsic, image_h=h, image_w=w)
    camera_embedding = embed_rays(rays_o, rays_d, nframe=f)
    camera_embedding = einops.rearrange(camera_embedding, "b f (h w) c -> b c f h w", h=h, w=w)

    return camera_embedding

def get_plucker_coords(coords, intrinsics, extrinsics, normalize=True):
    '''
    coord : S B N 2 - normalize camera for batch axis(= view axis)
    intrinsics : S*B 3 3
    extrinsics : S*B 4 4
    '''
    if normalize:
        extrinsics = camera_center_normalization(extrinsics, nframe=coords.shape[1])
    
    S, B, N, _ = coords.shape
    device = coords.device

    c2w = torch.inverse(extrinsics)[:, :3, :4].to(device)  

    u = coords[..., 0]  
    v = coords[..., 1]
    fx = intrinsics[:, 0, 0].reshape(S, B, 1)
    fy = intrinsics[:, 1, 1].reshape(S, B, 1)
    cx = intrinsics[:, 0, 2].reshape(S, B, 1)
    cy = intrinsics[:, 1, 2].reshape(S, B, 1)

    dirs_cam = torch.stack([
        (u - cx) / fx,
        (v - cy) / fy,
        torch.ones_like(u)
    ], dim=-1)  
    dirs_cam = F.normalize(dirs_cam, dim=-1)

    
    R = c2w[:, :3, :3].view(S, B, 3, 3)
    t = c2w[:, :3, 3].view(S, B, 1, 3)
    dirs_world = torch.matmul(R.unsqueeze(2), dirs_cam.unsqueeze(-1)).squeeze(-1)
    orig_world = t.expand_as(dirs_world)

    
    cross = torch.cross(orig_world, dirs_world, dim=-1)
    plucker = torch.cat([dirs_world, cross], dim=-1)  

    return plucker.permute(1, 2, 0, 3)  