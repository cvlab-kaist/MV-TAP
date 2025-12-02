from torch.utils.data import DataLoader

from data.dexycb import DexYCB
from data.panoptic import Panoptic
from data.harmony4d import Harmony4D
from data.kubric_eval import KubricEval
from data.kubric_train import KubricTrain
from data.utils import collate_fn

def load_train_dataset(cfg):
    train_dataset = KubricTrain(
        data_root=cfg.train_path,
        crop_size=[cfg.resize_H, cfg.resize_W],
        traj_per_sample=cfg.traj_per_sample,
        use_augs=cfg.use_augs,
        keep_principal_point_centered=cfg.keep_principal_point_centered,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,     
        shuffle=True,              
        num_workers=4,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
        collate_fn=collate_fn,  
    )    
    
    return train_loader



def load_eval_dataset(cfg):
    eval_names = cfg.eval.names
    
    eval_loaders = []
    for name in eval_names:
        if name in 'dexycb':
            dataset = DexYCB(
                data_root=cfg.data_root,
                resize_to=[cfg.resize_H, cfg.resize_W],
                num_points=cfg.eval.num_points,
            )
        elif name in 'panoptic':
            dataset = Panoptic(
                data_root=cfg.data_root,
                resize_to=[cfg.resize_H, cfg.resize_W],
                num_points=cfg.eval.num_points,
            )
        elif name in 'kubric':
            dataset = KubricEval(
                data_root=cfg.data_root,
                resize_to=[cfg.resize_H, cfg.resize_W], 
                num_points=cfg.eval.num_points,
            )
        elif name in 'harmony':
            dataset = Harmony4D(
                data_root=cfg.data_root,
                resize_to=[cfg.resize_H, cfg.resize_W],
                num_points=cfg.eval.num_points,
            )
        else:
            raise ValueError(f'Unknown eval dataset name: {name}')

        eval_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
        
        eval_loaders.append(eval_loader)
    
    return eval_loaders