from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
from .utils import _gather_feat, _transpose_and_gather_feat, _transpose_and_gather_feat_planes_np

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _left_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _right_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    shape = heat.shape 
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i +1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape) 

def _top_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(1, heat.shape[0]):
        inds = (heat[i] >= heat[i - 1])
        ret[i] += ret[i - 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _bottom_aggregate(heat):
    '''
        heat: batchsize x channels x h x w
    '''
    heat = heat.transpose(3, 2) 
    shape = heat.shape
    heat = heat.reshape(-1, heat.shape[3])
    heat = heat.transpose(1, 0).contiguous()
    ret = heat.clone()
    for i in range(heat.shape[0] - 2, -1, -1):
        inds = (heat[i] >= heat[i + 1])
        ret[i] += ret[i + 1] * inds.float()
    return (ret - heat).transpose(1, 0).reshape(shape).transpose(3, 2)

def _h_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _left_aggregate(heat) + \
           aggr_weight * _right_aggregate(heat) + heat

def _v_aggregate(heat, aggr_weight=0.1):
    return aggr_weight * _top_aggregate(heat) + \
           aggr_weight * _bottom_aggregate(heat) + heat

'''
# Slow for large number of categories
def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs
'''
def _topk_channel(scores, K=40):
      batch, cat, height, width = scores.size()
      
      topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

      topk_inds = topk_inds % (height * width)
      topk_ys   = (topk_inds / width).int().float()
      topk_xs   = (topk_inds % width).int().float()

      return topk_scores, topk_inds, topk_ys, topk_xs

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()
      
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
      
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def agnex_ct_decode(
    t_heat, l_heat, b_heat, r_heat, ct_heat, 
    t_regr=None, l_regr=None, b_regr=None, r_regr=None, 
    K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000
):
    batch, cat, height, width = t_heat.size()

    '''
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    '''
    if aggr_weight > 0: 
      t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
      l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
      b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
      r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
      
    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)
      
      
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, _, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, _, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, _, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, _, r_ys, r_xs = _topk(r_heat, K=K)
      
    ct_heat_agn, ct_clses = torch.max(ct_heat, dim=1, keepdim=True)
      
    # import pdb; pdb.set_trace()

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()

    ct_inds     = box_ct_ys * width + box_ct_xs
    ct_inds     = ct_inds.view(batch, -1)
    ct_heat_agn = ct_heat_agn.view(batch, -1, 1)
    ct_clses    = ct_clses.view(batch, -1, 1)
    ct_scores   = _gather_feat(ct_heat_agn, ct_inds)
    clses       = _gather_feat(ct_clses, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None \
      and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5
      
    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()


    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                            b_xs, b_ys, r_xs, r_ys, clses], dim=2)

    return detections

def exct_decode(
    t_heat, l_heat, b_heat, r_heat, ct_heat, 
    t_regr=None, l_regr=None, b_regr=None, r_regr=None, 
    K=40, scores_thresh=0.1, center_thresh=0.1, aggr_weight=0.0, num_dets=1000
):
    batch, cat, height, width = t_heat.size()
    '''
    t_heat  = torch.sigmoid(t_heat)
    l_heat  = torch.sigmoid(l_heat)
    b_heat  = torch.sigmoid(b_heat)
    r_heat  = torch.sigmoid(r_heat)
    ct_heat = torch.sigmoid(ct_heat)
    '''

    if aggr_weight > 0:   
      t_heat = _h_aggregate(t_heat, aggr_weight=aggr_weight)
      l_heat = _v_aggregate(l_heat, aggr_weight=aggr_weight)
      b_heat = _h_aggregate(b_heat, aggr_weight=aggr_weight)
      r_heat = _v_aggregate(r_heat, aggr_weight=aggr_weight)
      
    # perform nms on heatmaps
    t_heat = _nms(t_heat)
    l_heat = _nms(l_heat)
    b_heat = _nms(b_heat)
    r_heat = _nms(r_heat)
      
    t_heat[t_heat > 1] = 1
    l_heat[l_heat > 1] = 1
    b_heat[b_heat > 1] = 1
    r_heat[r_heat > 1] = 1

    t_scores, t_inds, t_clses, t_ys, t_xs = _topk(t_heat, K=K)
    l_scores, l_inds, l_clses, l_ys, l_xs = _topk(l_heat, K=K)
    b_scores, b_inds, b_clses, b_ys, b_xs = _topk(b_heat, K=K)
    r_scores, r_inds, r_clses, r_ys, r_xs = _topk(r_heat, K=K)

    t_ys = t_ys.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    t_xs = t_xs.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_ys = l_ys.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    l_xs = l_xs.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_ys = b_ys.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    b_xs = b_xs.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_ys = r_ys.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    r_xs = r_xs.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)

    t_clses = t_clses.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_clses = l_clses.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_clses = b_clses.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_clses = r_clses.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    box_ct_xs = ((l_xs + r_xs + 0.5) / 2).long()
    box_ct_ys = ((t_ys + b_ys + 0.5) / 2).long()
    ct_inds = t_clses.long() * (height * width) + box_ct_ys * width + box_ct_xs
    ct_inds = ct_inds.view(batch, -1)
    ct_heat = ct_heat.view(batch, -1, 1)
    ct_scores = _gather_feat(ct_heat, ct_inds)

    t_scores = t_scores.view(batch, K, 1, 1, 1).expand(batch, K, K, K, K)
    l_scores = l_scores.view(batch, 1, K, 1, 1).expand(batch, K, K, K, K)
    b_scores = b_scores.view(batch, 1, 1, K, 1).expand(batch, K, K, K, K)
    r_scores = r_scores.view(batch, 1, 1, 1, K).expand(batch, K, K, K, K)
    ct_scores = ct_scores.view(batch, K, K, K, K)
    scores    = (t_scores + l_scores + b_scores + r_scores + 2 * ct_scores) / 6

    # reject boxes based on classes
    cls_inds = (t_clses != l_clses) + (t_clses != b_clses) + \
               (t_clses != r_clses)
    cls_inds = (cls_inds > 0)

    top_inds  = (t_ys > l_ys) + (t_ys > b_ys) + (t_ys > r_ys)
    top_inds = (top_inds > 0)
    left_inds  = (l_xs > t_xs) + (l_xs > b_xs) + (l_xs > r_xs)
    left_inds = (left_inds > 0)
    bottom_inds  = (b_ys < t_ys) + (b_ys < l_ys) + (b_ys < r_ys)
    bottom_inds = (bottom_inds > 0)
    right_inds  = (r_xs < t_xs) + (r_xs < l_xs) + (r_xs < b_xs)
    right_inds = (right_inds > 0)

    sc_inds = (t_scores < scores_thresh) + (l_scores < scores_thresh) + \
              (b_scores < scores_thresh) + (r_scores < scores_thresh) + \
              (ct_scores < center_thresh)
    sc_inds = (sc_inds > 0)

    scores = scores - sc_inds.float()
    scores = scores - cls_inds.float()
    scores = scores - top_inds.float()
    scores = scores - left_inds.float()
    scores = scores - bottom_inds.float()
    scores = scores - right_inds.float()

    scores = scores.view(batch, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    if t_regr is not None and l_regr is not None \
      and b_regr is not None and r_regr is not None:
        t_regr = _transpose_and_gather_feat(t_regr, t_inds)
        t_regr = t_regr.view(batch, K, 1, 1, 1, 2)
        l_regr = _transpose_and_gather_feat(l_regr, l_inds)
        l_regr = l_regr.view(batch, 1, K, 1, 1, 2)
        b_regr = _transpose_and_gather_feat(b_regr, b_inds)
        b_regr = b_regr.view(batch, 1, 1, K, 1, 2)
        r_regr = _transpose_and_gather_feat(r_regr, r_inds)
        r_regr = r_regr.view(batch, 1, 1, 1, K, 2)

        t_xs = t_xs + t_regr[..., 0]
        t_ys = t_ys + t_regr[..., 1]
        l_xs = l_xs + l_regr[..., 0]
        l_ys = l_ys + l_regr[..., 1]
        b_xs = b_xs + b_regr[..., 0]
        b_ys = b_ys + b_regr[..., 1]
        r_xs = r_xs + r_regr[..., 0]
        r_ys = r_ys + r_regr[..., 1]
    else:
        t_xs = t_xs + 0.5
        t_ys = t_ys + 0.5
        l_xs = l_xs + 0.5
        l_ys = l_ys + 0.5
        b_xs = b_xs + 0.5
        b_ys = b_ys + 0.5
        r_xs = r_xs + 0.5
        r_ys = r_ys + 0.5
      
    bboxes = torch.stack((l_xs, t_ys, r_xs, b_ys), dim=5)
    bboxes = bboxes.view(batch, -1, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses  = t_clses.contiguous().view(batch, -1, 1)
    clses  = _gather_feat(clses, inds).float()

    t_xs = t_xs.contiguous().view(batch, -1, 1)
    t_xs = _gather_feat(t_xs, inds).float()
    t_ys = t_ys.contiguous().view(batch, -1, 1)
    t_ys = _gather_feat(t_ys, inds).float()
    l_xs = l_xs.contiguous().view(batch, -1, 1)
    l_xs = _gather_feat(l_xs, inds).float()
    l_ys = l_ys.contiguous().view(batch, -1, 1)
    l_ys = _gather_feat(l_ys, inds).float()
    b_xs = b_xs.contiguous().view(batch, -1, 1)
    b_xs = _gather_feat(b_xs, inds).float()
    b_ys = b_ys.contiguous().view(batch, -1, 1)
    b_ys = _gather_feat(b_ys, inds).float()
    r_xs = r_xs.contiguous().view(batch, -1, 1)
    r_xs = _gather_feat(r_xs, inds).float()
    r_ys = r_ys.contiguous().view(batch, -1, 1)
    r_ys = _gather_feat(r_ys, inds).float()


    detections = torch.cat([bboxes, scores, t_xs, t_ys, l_xs, l_ys, 
                            b_xs, b_ys, r_xs, r_ys, clses], dim=2)


    return detections

def ddd_decode(heat, rot, depth, dim, wh=None, reg=None, K=40):
    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
      
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    depth = _transpose_and_gather_feat(depth, inds)
    depth = depth.view(batch, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    xs = xs.view(batch, K, 1)
    ys = ys.view(batch, K, 1)
      
    if wh is not None:
        wh = _transpose_and_gather_feat(wh, inds)
        wh = wh.view(batch, K, 2)
        detections = torch.cat(
            [xs, ys, scores, rot, depth, dim, wh, clses], dim=2)
    else:
        detections = torch.cat(
            [xs, ys, scores, rot, depth, dim, clses], dim=2)
      
    return detections

def ctdet_decode(heat, wh, reg=None, cat_spec_wh=False, K=100):
    batch, cat, height, width = heat.size()

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)
      
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
      reg = _transpose_and_gather_feat(reg, inds)
      reg = reg.view(batch, K, 2)
      xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
      ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
      xs = xs.view(batch, K, 1) + 0.5
      ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    if cat_spec_wh:
      wh = wh.view(batch, K, cat, 2)
      clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
      wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
      wh = wh.view(batch, K, 2)
    clses  = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2, 
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
      
    return detections

def multi_pose_decode(
    heat, wh, kps, reg=None, hm_hp=None, hp_offset=None, K=100):
  batch, cat, height, width = heat.size()
  num_joints = kps.shape[1] // 2
  # heat = torch.sigmoid(heat)
  # perform nms on heatmaps
  heat = _nms(heat)
  scores, inds, clses, ys, xs = _topk(heat, K=K)

  kps = _transpose_and_gather_feat(kps, inds)
  kps = kps.view(batch, K, num_joints * 2)
  kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
  kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
  if reg is not None:
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
    ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
  else:
    xs = xs.view(batch, K, 1) + 0.5
    ys = ys.view(batch, K, 1) + 0.5
  wh = _transpose_and_gather_feat(wh, inds)
  wh = wh.view(batch, K, 2)
  clses  = clses.view(batch, K, 1).float()
  scores = scores.view(batch, K, 1)

  bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
                      ys - wh[..., 1:2] / 2,
                      xs + wh[..., 0:1] / 2, 
                      ys + wh[..., 1:2] / 2], dim=2)
  if hm_hp is not None:
      hm_hp = _nms(hm_hp)
      thresh = 0.1
      kps = kps.view(batch, K, num_joints, 2).permute(
          0, 2, 1, 3).contiguous() # b x J x K x 2
      reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
      hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K) # b x J x K
      if hp_offset is not None:
          hp_offset = _transpose_and_gather_feat(
              hp_offset, hm_inds.view(batch, -1))
          hp_offset = hp_offset.view(batch, num_joints, K, 2)
          hm_xs = hm_xs + hp_offset[:, :, :, 0]
          hm_ys = hm_ys + hp_offset[:, :, :, 1]
      else:
          hm_xs = hm_xs + 0.5
          hm_ys = hm_ys + 0.5
        
      mask = (hm_score > thresh).float()
      hm_score = (1 - mask) * -1 + mask * hm_score
      hm_ys = (1 - mask) * (-10000) + mask * hm_ys
      hm_xs = (1 - mask) * (-10000) + mask * hm_xs
      hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
          2).expand(batch, num_joints, K, K, 2)
      dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
      min_dist, min_ind = dist.min(dim=3) # b x J x K
      hm_score = hm_score.gather(2, min_ind).unsqueeze(-1) # b x J x K x 1
      min_dist = min_dist.unsqueeze(-1)
      min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
          batch, num_joints, K, 1, 2)
      hm_kps = hm_kps.gather(3, min_ind)
      hm_kps = hm_kps.view(batch, num_joints, K, 2)
      l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
      mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
             (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
             (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
      mask = (mask > 0).float().expand(batch, num_joints, K, 2)
      #kps = (1 - mask) * hm_kps + mask * kps

      kps = kps.permute(0, 2, 1, 3).contiguous().view(
          batch, K, num_joints * 2)
  detections = torch.cat([bboxes, scores, kps, clses], dim=2)
    
  return detections

import matplotlib.pyplot as plt
import numpy as np

def gen_position(kps,dim,rot,meta,const):
    b=kps.size(0)
    c=kps.size(1)
    opinv=meta['trans_output_inv']
    calib=meta['calib']

    opinv = opinv.unsqueeze(1)
    opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float() #(100,2,3)
    
    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
    hom = torch.ones(b, c, 1, 9).cuda()
    kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9) # [100, 3, 9]  
    kps = torch.bmm(opinv, kps).view(b, c, 2, 9) # [1, 100, 2, 9]
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18 [1, 100, 2, 9]->[1, 100, 18]
    
    si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1]
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
    alpna_pre = alpna_pre.unsqueeze(2)
    # alpna_pre=rot_gt

    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()
    kpoint = kps[:, :, :16]
    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2)
    cxy = cxy.repeat(1, 1, 8)  # b,c,16
    kp_norm = (kpoint - cxy) / f

    l = dim[:, :, 2:3]
    h = dim[:, :, 0:1]
    w = dim[:, :, 1:2]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)

    B = torch.zeros_like(kpoint)
    C = torch.zeros_like(kpoint)

    kp = kp_norm.unsqueeze(3)  # b,c,16,1
    const = const.expand(b, c, -1, -1)
    A = torch.cat([const, kp], dim=3)

    B[:, :, 0:1] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 1:2] = h * 0.5
    B[:, :, 2:3] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 3:4] = h * 0.5
    B[:, :, 4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 5:6] = h * 0.5
    B[:, :, 6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 7:8] = h * 0.5
    B[:, :, 8:9] = l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 9:10] = -h * 0.5
    B[:, :, 10:11] = l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 11:12] = -h * 0.5
    B[:, :, 12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
    B[:, :, 13:14] = -h * 0.5
    B[:, :, 14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
    B[:, :, 15:16] = -h * 0.5

    C[:, :, 0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 4:5] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 5:6] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 6:7] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 7:8] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 8:9] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 12:13] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 13:14] = l * 0.5 * sinori - w * 0.5 * cosori
    C[:, :, 14:15] = l * 0.5 * sinori + w * 0.5 * cosori
    C[:, :, 15:16] = l * 0.5 * sinori + w * 0.5 * cosori

    B = B - kp_norm * C

    # A=A*kps_mask1

    AT = A.permute(0, 1, 3, 2)
    AT = AT.view(b * c, 3, 16)
    A = A.view(b * c, 16, 3)
    B = B.view(b * c, 16, 1).float()
    # mask = mask.unsqueeze(2)

    pinv = torch.bmm(AT, A)
    pinv = torch.inverse(pinv)  # b*c 3 3



    pinv = torch.bmm(pinv, AT)
    pinv = torch.bmm(pinv, B)
    pinv = pinv.view(b, c, 3, 1).squeeze(3)

    #pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2
    return pinv,rot_y,kps

def gen_position_n_points(kps, n_kps, dim,rot, meta,planes_const, planes):
    b=n_kps.size(0)
    c=n_kps.size(1)
    opinv=meta['trans_output_inv']
    calib=meta['calib']

    opinv = opinv.unsqueeze(1)
    opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float()
    
    kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2)
    hom = torch.ones(b, c, 1, 9).cuda()
    kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9) # [100, 3, 9]  
    kps = torch.bmm(opinv, kps).view(b, c, 2, 9) # [1, 100, 2, 9]
    kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # 16.32,18 [1, 100, 2, 9]->[1, 100, 18]
    
    n_kps = n_kps.view(b, c, planes, -1, 2).permute(0, 1, 4, 3, 2) # [1, 100, 6, 8] -> [1, 100, 2, 4, 6]
    n_points_per_planes = n_kps.size(3)
    n_hom = torch.ones(b, c, 1, n_kps.size(3), n_kps.size(4)).cuda() # [1, 100, 1, 4, 6]
    n_kps = torch.cat((n_kps, n_hom), dim=2).view(-1, 3, n_points_per_planes*planes) # [100, 3, 24]  
    n_kps = torch.bmm(opinv, n_kps).view(b, c, 2, n_points_per_planes, planes)# [1, 100, 2, 4, 6]  
    n_kps = n_kps.permute(0, 1, 4, 3, 2).contiguous().view(b, c, -1)  # 16.32,18 [1, 100, 6, 4, 2] -> [1, 100, 48]
    
    si = torch.zeros_like(n_kps[:, :, 0:1]) + calib[:, 0:1, 0:1] # [1, 100, 1]
    alpha_idx = rot[:, :, 1] > rot[:, :, 5]
    alpha_idx = alpha_idx.float()
    alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi) # [1, 100]
    alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi) # [1, 100]
    alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx) # [1, 100]
    alpna_pre = alpna_pre.unsqueeze(2) # [1, 100, 1]
    # alpna_pre=rot_gt

    rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si)
    rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
    rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

    calib = calib.unsqueeze(1)
    calib = calib.expand(b, c, -1, -1).contiguous()
    kpoint = n_kps
    f = calib[:, :, 0, 0].unsqueeze(2)
    f = f.expand_as(kpoint)
    cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2)
    cxy = torch.cat((cx, cy), dim=2) # [1, 100, 2]
    cxy = cxy.repeat(1, 1, n_points_per_planes*planes)  # b,c,16 [1, 100, 48]
    kp_norm = (kpoint - cxy) / f
    

    l = dim[:, :, 2:3]
    h = dim[:, :, 0:1]
    w = dim[:, :, 1:2]
    cosori = torch.cos(rot_y)
    sinori = torch.sin(rot_y)
    
    kpoint = kpoint.view(b,c,planes,n_points_per_planes*2)
    Planes_B = torch.zeros_like(kpoint)
    Planes_C = torch.zeros_like(kpoint)

    kp = kp_norm.view(b,c,planes,n_points_per_planes*2).unsqueeze(4)  # b,c,16,1
    planes_const = planes_const.expand(b, c, -1, -1, -1) # [1, 100, 6, 8, 2]
    Planes_A = torch.cat([planes_const, kp], dim=4) # [1,100,6,8,3]
    
    points_index= []
    resolution_dic = {'4':2,'9':1,'25':0.5}
    resolution = resolution_dic[str(n_points_per_planes)]
    
    for i in [1, -1]: # 1:right plane ,  -1: left plane
        for j in np.arange(1,-1-resolution,-resolution): # 1:bottom plane , -1: top plane
            for k in np.arange(1,-1-resolution,-resolution): # 1:front plane ,-1: back plane                            
                points_index.append([0.5*i,0.5*round(j,2),0.5*round(k,2)])
    
    for j in [1, -1]: #  1:bottom plane, -1: top plane   
        for k in np.arange(1,-1-resolution,-resolution): # 1:front plane ,-1: back plane
            for i in np.arange(1,-1-resolution,-resolution): #   1:left plane, -1: right plane 
               
                points_index.append([0.5*round(i,2),0.5*j,0.5*round(k,2)])
    
    for k in [1, -1]: # 1:front plane ,-1:  back plane
        for i in np.arange(1,-1-resolution,-resolution): #    1:left plane , -1: right plane 
            for j in np.arange(1,-1-resolution,-resolution): #    1:bottom plane,  -1: top plane                            
                points_index.append([0.5*round(i,2),0.5*round(j,2),0.5*k])
    
    right_plane_p_ind    = points_index[0*n_points_per_planes:1*n_points_per_planes]
    left_plane_p_ind     = points_index[1*n_points_per_planes:2*n_points_per_planes]
    bottom_plane_p_ind   = points_index[2*n_points_per_planes:3*n_points_per_planes]
    top_plane_p_ind      = points_index[3*n_points_per_planes:4*n_points_per_planes]
    front_plane_p_ind    = points_index[4*n_points_per_planes:5*n_points_per_planes]
    back_plane_p_ind     = points_index[5*n_points_per_planes:6*n_points_per_planes]
    plane_p_ind = [right_plane_p_ind,left_plane_p_ind,bottom_plane_p_ind,top_plane_p_ind,front_plane_p_ind,back_plane_p_ind]
    
    planes_n_kps_3d_coef = torch.tensor(plane_p_ind, dtype=torch.float32).cuda()
    planes_n_kps_3d_coef = planes_n_kps_3d_coef.expand(b, c, -1, -1, -1)
    for i in range(n_points_per_planes):
            Planes_B[:,:,:,i*2:i*2+1] = l.unsqueeze(2).repeat(1,1,planes,1) * planes_n_kps_3d_coef[:,:,:,i,0:1] * cosori.unsqueeze(2).repeat(1,1,planes,1) + w.unsqueeze(2).repeat(1,1,planes,1) * planes_n_kps_3d_coef[:,:,:,i,2:3] * sinori.unsqueeze(2).repeat(1,1,planes,1)
            Planes_B[:,:,:,i*2+1:i*2+2] = h.unsqueeze(2).repeat(1,1,planes,1) * planes_n_kps_3d_coef[:,:,:,i,1:2]
    for i in range(n_points_per_planes):
            Planes_C[:,:,:,i*2:i*2+2] = l.unsqueeze(2).repeat(1,1,planes,1) * planes_n_kps_3d_coef[:,:,:,i,0:1] * (-sinori.unsqueeze(2).repeat(1,1,planes,1)) + w.unsqueeze(2).repeat(1,1,planes,1) * planes_n_kps_3d_coef[:,:,:,i,1:2] * cosori.unsqueeze(2).repeat(1,1,planes,1)
    
    kp_norm = kp_norm.view(b,c,planes,n_points_per_planes*2)
    Planes_B = Planes_B - kp_norm * Planes_C # (1,100,6,8)

    # A=A*kps_mask1

    Planes_AT = Planes_A.permute(0, 1, 4, 3, 2) # [1, 100, 6, 8, 3] -> [1, 100, 3, 8, 6]
    
    Planes_AT = Planes_AT.contiguous().view(b * c, 3, n_points_per_planes*planes*2) # [100,3,48]
    
    Planes_A = Planes_A.view(b * c, n_points_per_planes*planes*2, 3) # [100, 48, 3]
    Planes_B = Planes_B.view(b * c, n_points_per_planes*planes*2, 1).float()
    # mask = mask.unsqueeze(2)

    pinv = torch.bmm(Planes_AT, Planes_A)
    pinv = torch.inverse(pinv + 1e-4)  # b*c 3 3

    pinv = torch.bmm(pinv, Planes_AT)
    pinv = torch.bmm(pinv, Planes_B)
    pinv = pinv.view(b, c, 3, 1).squeeze(3) # ([1, 100, 3])
    #pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2
    return pinv,rot_y,n_kps

def car_pose_decode(
        heat, wh, kps,dim,rot, prob=None,reg=None, hm_hp=None, hp_offset=None, K=100,meta=None,const=None):

    batch, cat, height, width = heat.size()
    num_joints = kps.shape[1] // 2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # hm_show,_=torch.max(hm_hp,1)
    # hm_show=hm_show.squeeze(0)
    # hm_show=hm_show.detach().cpu().numpy().copy()
    # plt.imshow(hm_show, 'gray')
    # plt.show()

    heat = _nms(heat)
    scores, inds, clses, ys, xs = _topk(heat, K=K)

    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch, K, 3)
    # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
    # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
    # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
    rot = _transpose_and_gather_feat(rot, inds)
    rot = rot.view(batch, K, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:,:,0]
    prob = prob.view(batch, K, 1)
    if hm_hp is not None:
        hm_hp = _nms(hm_hp)
        thresh = 0.1
        kps = kps.view(batch, K, num_joints, 2).permute(
            0, 2, 1, 3).contiguous()  # b x J x K x 2
        reg_kps = kps.unsqueeze(3).expand(batch, num_joints, K, K, 2)
        hm_score, hm_inds, hm_ys, hm_xs = _topk_channel(hm_hp, K=K)  # b x J x K
        if hp_offset is not None:
            hp_offset = _transpose_and_gather_feat(
                hp_offset, hm_inds.view(batch, -1))
            hp_offset = hp_offset.view(batch, num_joints, K, 2)
            hm_xs = hm_xs + hp_offset[:, :, :, 0]
            hm_ys = hm_ys + hp_offset[:, :, :, 1]
        else:
            hm_xs = hm_xs + 0.5
            hm_ys = hm_ys + 0.5
        mask = (hm_score > thresh).float()
        hm_score = (1 - mask) * -1 + mask * hm_score
        hm_ys = (1 - mask) * (-10000) + mask * hm_ys
        hm_xs = (1 - mask) * (-10000) + mask * hm_xs
        hm_kps = torch.stack([hm_xs, hm_ys], dim=-1).unsqueeze(
            2).expand(batch, num_joints, K, K, 2)
        dist = (((reg_kps - hm_kps) ** 2).sum(dim=4) ** 0.5)
        min_dist, min_ind = dist.min(dim=3)  # b x J x K
        hm_score = hm_score.gather(2, min_ind).unsqueeze(-1)  # b x J x K x 1
        min_dist = min_dist.unsqueeze(-1)
        min_ind = min_ind.view(batch, num_joints, K, 1, 1).expand(
            batch, num_joints, K, 1, 2)
        hm_kps = hm_kps.gather(3, min_ind)
        hm_kps = hm_kps.view(batch, num_joints, K, 2)
        l = bboxes[:, :, 0].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        t = bboxes[:, :, 1].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        r = bboxes[:, :, 2].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        b = bboxes[:, :, 3].view(batch, 1, K, 1).expand(batch, num_joints, K, 1)
        mask = (hm_kps[..., 0:1] < l) + (hm_kps[..., 0:1] > r) + \
               (hm_kps[..., 1:2] < t) + (hm_kps[..., 1:2] > b) + \
               (hm_score < thresh) + (min_dist > (torch.max(b - t, r - l) * 0.3))
        mask = (mask > 0).float().expand(batch, num_joints, K, 2)
        kps = (1 - mask) * hm_kps + mask * kps
        kps = kps.permute(0, 2, 1, 3).contiguous().view(
            batch, K, num_joints * 2)
        hm_score=hm_score.permute(0, 2, 1, 3).squeeze(3).contiguous()
    position,rot_y,kps_inv=gen_position(kps,dim,rot,meta,const)

    detections = torch.cat([bboxes, scores, kps_inv, dim,hm_score,rot_y,position,prob,clses], dim=2)


    return detections
def car_pose_decode_faster(
        heat, kps, dim, rot, prob, K=100,meta=None,const=None):

    batch, cat, height, width = heat.size() # batch=1, cat=3, height=96, width=320
    num_joints = kps.shape[1] // 2 # num_joints = 18/2
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    # hm_show,_=torch.max(hm_hp,1)
    # hm_show=hm_show.squeeze(0)
    # hm_show=hm_show.detach().cpu().numpy().copy()
    # plt.imshow(hm_show, 'gray')
    # plt.show()

    heat = _nms(heat) # (1,3,96,320)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    # scores = (1,100)
    # inds = (1,100)
    # clses = (1,100)
    # ys = (1,100) 
    # xs = (1,100)
    clses = clses.view(batch, K, 1).float() # (1,100)->(1,100,1) 
    kps = _transpose_and_gather_feat(kps, inds) #  (1,18,96,320) -> (1, 100, 18)
    kps = kps.view(batch, K, num_joints * 2) # (1, 100, 18) ->(1, 100, 18)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)  # (1,100,9)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints) # (1,100,9)
    scores = scores.view(batch, K, 1) # (1,100)->(1,100,1)

    dim = _transpose_and_gather_feat(dim, inds) # (1,3,96,320) -> (1,100,3)
    dim = dim.view(batch, K, 3) # (1,100,3)->(1,100,3)
    
    # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
    # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
    # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
    rot = _transpose_and_gather_feat(rot, inds) # (1,8,96,320) -> (1, 100, 8)
    rot = rot.view(batch, K, 8) # (1, 100, 8)-> (1, 100, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:,:,0] #  (1,1,96,320) -> (1,100,1) -> (1,100)
    prob = prob.view(batch, K, 1) # (1,100) -> (1,100,1)
    position,rot_y,kps_inv=gen_position(kps,dim,rot,meta,const) # position=[1, 100, 3], rot_y=[1, 100, 1], kps_inv=[1, 100, 18]
    #bboxes=kps[:,:,0:4]
    bboxes_kp=kps.view(kps.size(0),kps.size(1),9,2) # (1, 100, 18) - (1, 100, 9, 2) Note:可修改成N个点
    box_min,_=torch.min(bboxes_kp,dim=2)
    box_max,_=torch.max(bboxes_kp,dim=2)
    bboxes=torch.cat((box_min,box_max),dim=2) # (1,100,4)
    hm_score=kps[:,:,0:9]# (1, 100, 9) 

    detections = torch.cat([bboxes, scores, kps_inv, dim, hm_score, rot_y, position, prob, clses], dim=2) #(1, 100, 41) =(4,1,18,3,9,1,3,1,1)
    # bboxes important (could be modify)!!!
    # hm_score can also modify
    # possition can also modify
    return detections, 9, 1

def car_pose_decode_faster_n_points(
        heat, kps, n_kps, dim, rot, prob, K=100,planes=6,meta=None,planes_const=None):
    
    
    
    # n_kps = [1, 48, 96, 320]
    n_kps = n_kps.contiguous().reshape(n_kps.size(0), planes, -1, n_kps.size(2), n_kps.size(3)) # [B, 6, 8, 96, 320]
    batch, cat, height, width = heat.size() # batch=1, cat=3, height=96, width=320
    num_joints_per_planes = n_kps.shape[2] // 2 # num_joints = 8/2
    
    heat = _nms(heat) # (1,3,96,320)
    scores, inds, clses, ys, xs = _topk(heat, K=K)
    
    #================================
    num_joints = kps.shape[1] // 2
    kps = _transpose_and_gather_feat(kps, inds)
    kps = kps.view(batch, K, num_joints * 2) # (1, 100, 18) ->(1, 100, 18)
    kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, num_joints)  # (1,100,9)
    kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, num_joints) # (1,100,9)
    #================================ 
    
    clses = clses.view(batch, K, 1).float() # (1,100)->(1,100,1)
    n_kps = _transpose_and_gather_feat_planes_np(n_kps, inds, flag='np_ind') #  (1,6,8,96,320) -> (1, 100, 6, 8)
    n_kps = n_kps.view(batch, K, planes, num_joints_per_planes * 2) # (1, 100, 6, 8) ->(1, 100, 6, 8)
    n_kps[..., ::2] += xs.view(batch, K, 1).expand(batch, K, planes).view(batch, K, planes, 1).expand(batch, K, planes, num_joints_per_planes) # (1,100,6,4)
    n_kps[..., 1::2] += ys.view(batch, K, 1).expand(batch, K, planes).view(batch, K, planes, 1).expand(batch, K, planes, num_joints_per_planes) # (1,100,6,4)
    scores = scores.view(batch, K, 1) # (1,100)->(1,100,1)

    dim = _transpose_and_gather_feat(dim, inds) # (1,3,96,320) -> (1,100,3)
    dim = dim.view(batch, K, 3) # (1,100,3)->(1,100,3)
    
    # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
    # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
    # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
    rot = _transpose_and_gather_feat(rot, inds) # (1,8,96,320) -> (1, 100, 8)
    rot = rot.view(batch, K, 8) # (1, 100, 8)-> (1, 100, 8)
    prob = _transpose_and_gather_feat(prob, inds)[:,:,0] #  (1,1,96,320) -> (1,100,1) -> (1,100)
    prob = prob.view(batch, K, 1) # (1,100) -> (1,100,1)
    
    position,rot_y,kps_inv=gen_position_n_points(kps, n_kps, dim, rot, meta, planes_const, planes) # position=[1, 100, 3], rot_y=[1, 100, 1], kps_inv=[1, 100, 48]
    #bboxes=kps[:,:,0:4]
    n_kps_temp = n_kps.view(n_kps.size(0),n_kps.size(1),planes,num_joints_per_planes,2) # (1, 100, 6, 8) -> (1, 100, 6, 4, 2) Note:可修改成N个点
    
    bboxes_kp = n_kps_temp.view(n_kps.size(0),n_kps.size(1),planes*num_joints_per_planes,2) # (1, 100, 6, 8, 2) -> (1, 100, 24, 2)
    
    box_min,_=torch.min(bboxes_kp,dim=2) # ([1, 100, 2])
    box_max,_=torch.max(bboxes_kp,dim=2) # ([1, 100, 2])
    bboxes=torch.cat((box_min,box_max),dim=2) # (1,100,4)
    
    hm_score= n_kps[:,:,:,0:num_joints_per_planes]# (1, 100, 6, 4) 
    hm_score = hm_score.contiguous().view(hm_score.size(0),hm_score.size(1),-1) # (1,100,24)
    detections = torch.cat([bboxes, scores, kps_inv, dim, hm_score, rot_y, position, prob, clses], dim=2) #(1, 100, 86) = (4,1,48,3,24,1,3,1,1)
    
    # bboxes important (could be modify)!!!
    # hm_score can also modify
    # possition can also modify
    return detections, num_joints_per_planes, planes
