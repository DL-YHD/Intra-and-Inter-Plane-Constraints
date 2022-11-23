from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4) # 若x< 1e-4 为1e-4, 若1e-4 < x < 1-1e-4为 x, 若x < 1-1e-4 为1-1e-4
  return y

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2) # dim = 18
    
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim) # shape = (8,32,18)
    
    feat = feat.gather(1, ind) # 根据索引找到对应下标的值。
    # feat = [8, 32, 18]
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _gather_feat_np(feat, ind, flag, mask=None):
    
    if flag !='Plane_ind':
        dim_1  = feat.size(2) # dim_1 = 6
        dim_2  = feat.size(3) # dim_2 = 8
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim_1) # shape = (8,32,6)
        ind  = ind.unsqueeze(3).expand(ind.size(0), ind.size(1), ind.size(2), dim_2) # shape = (8, 32, 6, 8)
    else:
        # v1
        dim_3  = feat.size(4) # dim_3 = 2
        ind  = ind.unsqueeze(4).expand(ind.size(0), ind.size(1), ind.size(2), ind.size(3), dim_3) # shape = (8, 32, 6, 4, 2)
        
        # v2 
        # dim_3 = feat.size(3) # dim_3 = 2
        # ind  = ind.unsqueeze(3).expand(ind.size(0), ind.size(1), ind.size(2), dim_3) # shape = (8, 32, 6, 2)
    feat = feat.gather(1, ind) # 根据索引找到对应下标的值。 shape = [8, 32, 6, 8]
    
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim_1,dim_2)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous() # 调换通道顺序,将(B,C,W,H) -> (B,W,H,C)
    feat = feat.view(feat.size(0), -1, feat.size(3)) # (B,96,320,18) ->(B,30720,18)
    feat = _gather_feat(feat, ind) #(B,32,18)
    return feat

def _transpose_and_gather_feat_planes_np(feat, ind, flag):

    if flag != 'Plane_ind':
        feat = feat.permute(0, 3, 4, 1, 2).contiguous() # 调换通道顺序,将(B,C,W,H,D) -> (B,W,H,D,C) [8, 96, 320, 6, 8]
        feat = feat.view(feat.size(0), -1, feat.size(3), feat.size(4)) # (B,96,320,6,8) ->(B,30720,6,8)
        # feat = feat.view(feat.size(0), -1 , feat.size(3)*feat.size(4)) # (B,96,320,6,8) ->(B,30720,6*8)
    else:
        # v1
        feat = feat.permute(0, 4, 5, 1, 2, 3).contiguous() # 调换通道顺序,将(B,C,W,H,D) -> (B,W,H,D,C) (B, 6, 4, 2, 96, 360) ->(B, 96, 360, 6, 4, 2)
        feat = feat.view(feat.size(0), -1, feat.size(3), feat.size(4),feat.size(5)) # (B, 96, 360, 6, 4, 2) ->(B, 34560, 6, 4, 2)
        
        # v2
        # feat = feat.permute(0, 3, 4, 1, 2).contiguous() # 调换通道顺序,将(B,C,W,H,D) -> (B,W,H,D,C) (B, 6, 2, 96, 360) ->(B, 96, 360, 6, 2)
        # feat = feat.view(feat.size(0), -1, feat.size(3), feat.size(4)) # (B, 96, 360, 6, 2) ->(B, 34560, 6, 2)
    feat = _gather_feat_np(feat, ind, flag) #  shape = [8, 32, 6, 8]
    return feat

def flip_tensor(x):
    return torch.flip(x, [3])
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)

def flip_lr(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)

def flip_lr_off(x, flip_idx):
  tmp = x.detach().cpu().numpy()[..., ::-1].copy()
  shape = tmp.shape
  tmp = tmp.reshape(tmp.shape[0], 17, 2, 
                    tmp.shape[2], tmp.shape[3])
  tmp[:, :, 0, :, :] *= -1
  for e in flip_idx:
    tmp[:, e[0], ...], tmp[:, e[1], ...] = \
      tmp[:, e[1], ...].copy(), tmp[:, e[0], ...].copy()
  return torch.from_numpy(tmp.reshape(shape)).to(x.device)
