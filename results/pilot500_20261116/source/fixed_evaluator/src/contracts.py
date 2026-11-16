"""Six explicit groups; this candidate protocol does not merge historical TSViT runs."""
import hashlib
from pathlib import Path
import numpy as np
import torch

GROUPS={
    'A':('none',False),'D':('none',True),
    'B_ind':('independent',False),'B_ind_D':('independent',True),
    'B_ovl':('overlap',False),'B_ovl_D':('overlap',True),
}
VERSION='ruan-unet-six-v1.1-candidate'
IGNORE_INDEX=-1
SEEDS=[20261113,20261114,20261115]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def encode(raw):
    if torch.any((raw<0)|(raw>8)):raise ValueError('Raw labels outside 0..8')
    return torch.where(raw==0,IGNORE_INDEX,raw.long()-1)

def model_input(s2):
    if s2.ndim!=5 or tuple(s2.shape[1:])!=(4,128,128,10):
        raise ValueError('Expected reflectance B,4,128,128,10')
    if not torch.isfinite(s2).all() or torch.max(torch.abs(s2))>20:
        raise ValueError('Invalid reflectance or raw DN passed without /10000')
    return s2.permute(0,1,4,2,3).contiguous().reshape(-1,40,128,128)

def trunk_hash(model):
    digest=hashlib.sha256()
    for name,value in model.state_dict().items():
        if name.startswith('refinement.'):continue
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
