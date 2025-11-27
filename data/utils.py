import torch

def collate_fn(batch_list):
    
    batch_list = [b for b in batch_list if b is not None]
    if len(batch_list) == 0:
        return None  

    out = {}
    keys = batch_list[0].keys()
    for k in keys:
        vals = [b[k] for b in batch_list]

        
        if any(v is None for v in vals):
            out[k] = None
            continue

        v0 = vals[0]
        if torch.is_tensor(v0):
            try:
                out[k] = torch.stack(vals, dim=0)
            except Exception:
                
                out[k] = vals
        else:
            
            out[k] = vals
    return out