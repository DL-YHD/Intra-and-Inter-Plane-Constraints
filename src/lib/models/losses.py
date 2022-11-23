# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from .utils import _transpose_and_gather_feat, _transpose_and_gather_feat_planes_np
import torch.nn.functional as F
import iou3d_cuda
from utils import kitti_utils_torch as kitti_utils
import time
import numpy as np

def calculate_angle(vector_1, vector_2):
    # vector_1 (B,C,3)
    # vector_2 (B,C,3)
    b = vector_1.size(0)
    c = vector_1.size(1)
    vector_1 = vector_1.view(b * c, -1) # (B,C,3) -> (B*C,3)
    vector_2 = vector_2.view(b * c, -1) # (B,C,3) -> (B*C,3)
    
    # 向量模
    l_vector_1 = torch.sqrt(torch.sum(vector_1*vector_1,dim=1)) # (B*C)
    l_vector_2 = torch.sqrt(torch.sum(vector_2*vector_2,dim=1)) # (B*C)
    
    # 点积
    dot_product = torch.sum(vector_1*vector_2,dim=1) # (B*C)
    
    # 计算夹角的cos值:
    cos = dot_product / ((l_vector_1 * l_vector_2) + 1e-4)
    
    # 计算夹角(弧度制)
    theata_curvature = torch.acos(cos)
    
    # 计算夹角(角度制)
    theata_angle = theata_curvature * 180 / 3.14
    
    return cos

def boxes_iou_bev(boxes_a, boxes_b):
    """
    :param boxes_a: (M, 5)
    :param boxes_b: (N, 5)
    :return:
        ans_iou: (M, N)
    """
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a: (N, 7) [x, y, z, h, w, l, ry]
    :param boxes_b: (M, 7) [x, y, z, h, w, l, ry]
    :return:
        ans_iou: (M, N)
    """
    boxes_a_bev = kitti_utils.boxes3d_to_bev_torch(boxes_a)
    boxes_b_bev = kitti_utils.boxes3d_to_bev_torch(boxes_b)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # height overlap
    boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, 3]).view(-1, 1) # a顶部的高度
    boxes_a_height_max = boxes_a[:, 1].view(-1, 1)                   # a底部的高度
    boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, 3]).view(1, -1) # b顶部的高度
    boxes_b_height_max = boxes_b[:, 1].view(1, -1)                   # b底部的高度

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min) # a框，b框之间最小的高度。
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max) # a框，b框之间最大的高度
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0) # 计算两个3D框重叠的高度

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h # 重叠的面积乘以高度得到3D iou

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    return iou3d


def boxes_iou2d(boxes_a_x, boxes_a_y, boxes_a_h, boxes_a_w, boxes_b_x, boxes_b_y, boxes_b_h, boxes_b_w):
    """
    :param boxes_a pred: (N, 6, 7) [[x, y, z, h, w, l, ry]...[x, y, z, h, w, l, ry]]
    :param boxes_b ground truth: (N, 6, 7) [[x, y, z, h, w, l, ry]...[x, y, z, h, w, l, ry]]
    :return:
        ans_iou_score: (N, 6)
    """

    # predict
    pcx = boxes_a_x
    pcy = boxes_a_y
    ph  = boxes_a_h
    pw  = boxes_a_w
    plane_h_max =  pcx + ph/2 # + h/2
    plane_h_min =  pcx - ph/2 # - h/2
    plane_w_max =  pcy + pw/2 # + w/2
    plane_w_min =  pcy - pw/2 # - w/2
    
    # ground truth
    gcx = boxes_b_x
    gcy = boxes_b_y
    gh  =  boxes_b_h
    gw  =  boxes_b_w
    
    plane_h_max_gt =  gcx + gh/2 # + h/2
    plane_h_min_gt =  gcx - gh/2 # - h/2
    plane_w_max_gt =  gcy + gw/2 # + w/2
    plane_w_min_gt =  gcy - gw/2 # - w/2

    # calculate right plane inter
    plane_inter_x_max = torch.min(plane_h_max, plane_h_max_gt)
    plane_inter_x_min = torch.max(plane_h_min, plane_h_min_gt)
    
    plane_inter_y_max = torch.min(plane_w_max, plane_w_max_gt)
    plane_inter_y_min = torch.max(plane_w_min, plane_w_min_gt)
    
    plane_inter_w = plane_inter_x_max - plane_inter_x_min
    plane_inter_h = plane_inter_y_max - plane_inter_y_min
    
    rh = (torch.abs(plane_inter_h) + plane_inter_h)/2 # 去除小于0的数
    rw = (torch.abs(plane_inter_w) + plane_inter_w)/2 # 去除小于0的数
    plane_inter = torch.mul(rh,rw) 
    
    # calculate the union
    union = torch.clamp(pw*ph + gw*gh -  plane_inter , min=1e-7)
    plane_IoU = plane_inter / union
    
    return plane_IoU
'''
def boxes_iou2d_gpu(boxes_a, boxes_b):
    """
    :param boxes_a pred: (N, 6, 7) [[x, y, z, h, w, l, ry]...[x, y, z, h, w, l, ry]]
    :param boxes_b ground truth: (N, 6, 7) [[x, y, z, h, w, l, ry]...[x, y, z, h, w, l, ry]]
    :return:
        ans_iou_score: (N, 6)
    """

    # predict     
    right_plane_h_max = boxes_a[:,:,0:3][:,0,0] + boxes_a[:,:,3:6][:,0,0]/2 # + h/2
    right_plane_h_min = boxes_a[:,:,0:3][:,0,0] - boxes_a[:,:,3:6][:,0,0]/2 # - h/2
    right_plane_w_max = boxes_a[:,:,0:3][:,0,1] + boxes_a[:,:,3:6][:,0,1]/2 # + w/2
    right_plane_w_min = boxes_a[:,:,0:3][:,0,1] - boxes_a[:,:,3:6][:,0,1]/2 # - w/2
    
    left_plane_h_max = boxes_a[:,:,0:3][:,1,0] + boxes_a[:,:,3:6][:,1,0]/2 # + h/2
    left_plane_h_min = boxes_a[:,:,0:3][:,1,0] - boxes_a[:,:,3:6][:,1,0]/2 # - h/2
    left_plane_w_max = boxes_a[:,:,0:3][:,1,1] + boxes_a[:,:,3:6][:,1,1]/2 # + w/2
    left_plane_w_min = boxes_a[:,:,0:3][:,1,1] - boxes_a[:,:,3:6][:,1,1]/2 # - w/2
    
    bottom_plane_w_max = boxes_a[:,:,0:3][:,2,1] + boxes_a[:,:,3:6][:,2,1]/2 # + w/2
    bottom_plane_w_min = boxes_a[:,:,0:3][:,2,1] - boxes_a[:,:,3:6][:,2,1]/2 # - w/2
    bottom_plane_l_max = boxes_a[:,:,0:3][:,2,2] + boxes_a[:,:,3:6][:,2,2]/2 # + l/2
    bottom_plane_l_min = boxes_a[:,:,0:3][:,2,2] - boxes_a[:,:,3:6][:,2,2]/2 # - l/2
    
    top_plane_w_max = boxes_a[:,:,0:3][:,3,1] + boxes_a[:,:,3:6][:,3,1]/2 # + w/2
    top_plane_w_min = boxes_a[:,:,0:3][:,3,1] - boxes_a[:,:,3:6][:,3,1]/2 # - w/2
    top_plane_l_max = boxes_a[:,:,0:3][:,3,2] + boxes_a[:,:,3:6][:,3,2]/2 # + l/2
    top_plane_l_min = boxes_a[:,:,0:3][:,3,2] - boxes_a[:,:,3:6][:,3,2]/2 # - l/2
    
    front_plane_h_max = boxes_a[:,:,0:3][:,4,0] + boxes_a[:,:,3:6][:,4,0]/2 # + h/2
    front_plane_h_min = boxes_a[:,:,0:3][:,4,0] - boxes_a[:,:,3:6][:,4,0]/2 # - h/2
    front_plane_l_max = boxes_a[:,:,0:3][:,4,2] + boxes_a[:,:,3:6][:,4,2]/2 # + l/2
    front_plane_l_min = boxes_a[:,:,0:3][:,4,2] - boxes_a[:,:,3:6][:,4,2]/2 # - l/2
    
    back_plane_h_max = boxes_a[:,:,0:3][:,5,0] + boxes_a[:,:,3:6][:,5,0]/2 # + h/2
    back_plane_h_min = boxes_a[:,:,0:3][:,5,0] - boxes_a[:,:,3:6][:,5,0]/2 # - h/2
    back_plane_l_max = boxes_a[:,:,0:3][:,5,2] + boxes_a[:,:,3:6][:,5,2]/2 # + l/2
    back_plane_l_min = boxes_a[:,:,0:3][:,5,2] - boxes_a[:,:,3:6][:,5,2]/2 # - l/2
    
    # ground truth
    right_plane_h_max_gt = boxes_b[:,:,0:3][:,0,0] + boxes_b[:,:,3:6][:,0,0]/2 # + h/2
    right_plane_h_min_gt = boxes_b[:,:,0:3][:,0,0] - boxes_b[:,:,3:6][:,0,0]/2 # - h/2
    right_plane_w_max_gt = boxes_b[:,:,0:3][:,0,1] + boxes_b[:,:,3:6][:,0,1]/2 # + w/2
    right_plane_w_min_gt = boxes_b[:,:,0:3][:,0,1] - boxes_b[:,:,3:6][:,0,1]/2 # - w/2
    
    left_plane_h_max_gt = boxes_b[:,:,0:3][:,1,0] + boxes_b[:,:,3:6][:,1,0]/2 # + h/2
    left_plane_h_min_gt= boxes_b[:,:,0:3][:,1,0] - boxes_b[:,:,3:6][:,1,0]/2 # - h/2
    left_plane_w_max_gt = boxes_b[:,:,0:3][:,1,1] + boxes_b[:,:,3:6][:,1,1]/2 # + w/2
    left_plane_w_min_gt = boxes_b[:,:,0:3][:,1,1] - boxes_b[:,:,3:6][:,1,1]/2 # - w/2
    
    bottom_plane_w_max_gt = boxes_b[:,:,0:3][:,2,1] + boxes_b[:,:,3:6][:,2,1]/2 # + w/2
    bottom_plane_w_min_gt = boxes_b[:,:,0:3][:,2,1] - boxes_b[:,:,3:6][:,2,1]/2 # - w/2
    bottom_plane_l_max_gt = boxes_b[:,:,0:3][:,2,2] + boxes_b[:,:,3:6][:,2,2]/2 # + l/2
    bottom_plane_l_min_gt = boxes_b[:,:,0:3][:,2,2] - boxes_b[:,:,3:6][:,2,2]/2 # - l/2
    
    top_plane_w_max_gt = boxes_b[:,:,0:3][:,3,1] + boxes_b[:,:,3:6][:,3,1]/2 # + w/2
    top_plane_w_min_gt = boxes_b[:,:,0:3][:,3,1] - boxes_b[:,:,3:6][:,3,1]/2 # - w/2
    top_plane_l_max_gt = boxes_b[:,:,0:3][:,3,2] + boxes_b[:,:,3:6][:,3,2]/2 # + l/2
    top_plane_l_min_gt = boxes_b[:,:,0:3][:,3,2] - boxes_b[:,:,3:6][:,3,2]/2 # - l/2
    
    front_plane_h_max_gt = boxes_b[:,:,0:3][:,4,0] + boxes_b[:,:,3:6][:,4,0]/2 # + h/2
    front_plane_h_min_gt = boxes_b[:,:,0:3][:,4,0] - boxes_b[:,:,3:6][:,4,0]/2 # - h/2
    front_plane_l_max_gt = boxes_b[:,:,0:3][:,4,2] + boxes_b[:,:,3:6][:,4,2]/2 # + l/2
    front_plane_l_min_gt = boxes_b[:,:,0:3][:,4,2] - boxes_b[:,:,3:6][:,4,2]/2 # - l/2
    
    back_plane_h_max_gt = boxes_b[:,:,0:3][:,5,0] + boxes_b[:,:,3:6][:,5,0]/2 # + h/2
    back_plane_h_min_gt = boxes_b[:,:,0:3][:,5,0] - boxes_b[:,:,3:6][:,5,0]/2 # - h/2
    back_plane_l_max_gt = boxes_b[:,:,0:3][:,5,2] + boxes_b[:,:,3:6][:,5,2]/2 # + l/2
    back_plane_l_min_gt = boxes_b[:,:,0:3][:,5,2] - boxes_b[:,:,3:6][:,5,2]/2 # - l/2
    
    
    # calculate right plane inter
    right_plane_inter_x_max = torch.min(right_plane_h_max, right_plane_h_max_gt)
    right_plane_inter_x_min = torch.max(right_plane_h_min, right_plane_h_min_gt)
    
    right_plane_inter_y_max = torch.min(right_plane_w_max, right_plane_w_max_gt)
    right_plane_inter_y_min = torch.max(right_plane_w_min, right_plane_w_min_gt)
    
    right_plane_inter_w = right_plane_inter_x_max - right_plane_inter_x_min
    right_plane_inter_h = right_plane_inter_y_max - right_plane_inter_y_min
    
    rh = (torch.abs(right_plane_inter_h) + right_plane_inter_h)/2 # 去除小于0的数
    rw = (torch.abs(right_plane_inter_w) + right_plane_inter_w)/2 # 去除小于0的数
    right_plane_inter = torch.mul(rh,rw) 
    
    # calculate the union
    right_plane_union = boxes_a[:,:,3:6][:,0,0]*boxes_a[:,:,3:6][:,0,1] + boxes_b[:,:,3:6][:,0,0]*boxes_b[:,:,3:6][:,0,1] - right_plane_inter  
    right_plane_IoU = right_plane_inter / right_plane_union
    
    # calculate left plane inter
    left_plane_inter_x_max = torch.min(left_plane_h_max, left_plane_h_max_gt)
    left_plane_inter_x_min = torch.max(left_plane_h_min, left_plane_h_min_gt)
    
    left_plane_inter_y_max = torch.min(left_plane_w_max, left_plane_w_max_gt)
    left_plane_inter_y_min = torch.max(left_plane_w_min, left_plane_w_min_gt)
    
    left_plane_inter_w = left_plane_inter_x_max - left_plane_inter_x_min
    left_plane_inter_h = left_plane_inter_y_max - left_plane_inter_y_min
    
    rh = (torch.abs(left_plane_inter_h) + left_plane_inter_h)/2 # 去除小于0的数
    rw = (torch.abs(left_plane_inter_w) + left_plane_inter_w)/2 # 去除小于0的数
    left_plane_inter = torch.mul(rh,rw) 
    
    # calculate the union
    left_plane_union = boxes_a[:,:,3:6][:,1,0]*boxes_a[:,:,3:6][:,1,1] + boxes_b[:,:,3:6][:,1,0]*boxes_b[:,:,3:6][:,1,1] - left_plane_inter  
    left_plane_IoU = left_plane_inter / left_plane_union
    
    
    
    
    return [right_plane_IoU,left_plane_IoU]
'''

def _slow_neg_loss(pred, gt):
    '''focal loss from CornerNet'''
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)

    neg_weights = torch.pow(1 - gt[neg_inds], 4)

    loss = 0
    pos_pred = pred[pos_inds]
    neg_pred = pred[neg_inds]

    pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
    neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if pos_pred.nelement() == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    # torch.Size([B, 3, 96, 320])
 
    pos_inds = gt.eq(1).float() # eq = equals 找出值为1的位置，若有则为1，无则为0
    neg_inds = gt.lt(1).float() # lt = less than 找出小于1的位置，若有则为1，无则为0


    neg_weights = torch.pow(1 - gt, 4) # neg_weights = (1-gt)^4

    loss = 0
    gamma = 2 # best optimization
    pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds # log(pred) * (1-pred)^2 * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds # log(1 - pred) * (pred)^2 * neg_weights * neg_inds

    num_pos = pos_inds.float().sum() # 位置为1的和 
    pos_loss = pos_loss.sum() # 正样本点每个点loss之和 
    neg_loss = neg_loss.sum() # 负样本点每个点loss之和

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

def _planes_neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
        
        pred = [B, 6, 4, 96, 320]
        gt   = [B, 6, 4, 96, 320]
    '''

    #==================================================================#
    # pos_inds 和 neg_inds 确定关键点和非关键点。
    
    loss = 0
    gamma = 2 # 超参数 best optimization
    alpha = 4 # 超参数
    
    pos_inds = gt.eq(1).float() # eq = equals 找出值为1的位置，若有则为1，无则为0
    neg_inds = gt.lt(1).float() # lt = less than 找出小于1的位置，若有则为1，无则为0.(Q: 这里是否有问题？ A:没问题)

    neg_weights = torch.pow(1 - gt, alpha)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, gamma) * pos_inds # log(pred) * (1-pred)^2 * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, gamma) * neg_weights * neg_inds # log(1 - pred) * (pred)^2 * neg_weights * neg_inds
       
    num_pos  = pos_inds.float().sum() # 位置为1的和，图象中物体关键点个数
    pos_loss = pos_loss.sum() # 关键点正样本时的focal loss之和 
    neg_loss = neg_loss.sum() # 关键点副样本时的focal loss之和 
    
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    #==================================================================#
    
    return loss

def _not_faster_neg_loss(pred, gt):
    
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    num_pos = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -= all_loss
    return loss


def _slow_reg_loss(regr, gt_regr, mask):
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr = regr[mask]
    gt_regr = gt_regr[mask]

    #regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _reg_loss(regr, gt_regr, mask):
    ''' L1 regression loss
      Arguments:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    '''
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    regr = regr * mask
    gt_regr = gt_regr * mask

    #regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, reduction='sum')
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class PlanesFocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''

    def __init__(self):
        super(PlanesFocalLoss, self).__init__()
        self.planes_neg_loss = _planes_neg_loss

    def forward(self, out, target):
        return self.planes_neg_loss(out, target)
    

class RegLoss(nn.Module):
    '''Regression loss for an output tensor
      Arguments:
        output (batch x dim x h x w)
        mask (batch x max_objects)
        ind (batch x max_objects)
        target (batch x max_objects x dim)
    '''

    def __init__(self):
        super(RegLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss


class RegL1Loss(nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # output = [1, 2, 96, 320]
        # mask = [1, 288]
        # ind  = [1, 288]
        # pred = [1, 288, 2]
        pred = _transpose_and_gather_feat(output, ind) # 同样根据索引找到对应位置的预测值
        mask = mask.unsqueeze(2).expand_as(pred).float() # 将mask升维转换成和pred一样的形状
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class PlanesRegL1Loss(nn.Module):
    def __init__(self):
        super(PlanesRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        # v1 
        # output = [1, 6, 4, 2, 96, 320]
        # mask = [1, 32, 6, 4]
        # ind  = [1, 32, 6, 4]
        # pred = [1, 32, 6, 4, 2]
        pred = _transpose_and_gather_feat_planes_np(output, ind, flag='Plane_ind') # (B, 32, 6, 4, 2)
        mask = mask.unsqueeze(4).expand_as(pred).float() # 将mask升维转换成和pred一样的形状 [1, 32, 6, 4, 2]
        
        # v2
        # output = [1, 6, 2, 96, 320]
        # mask = [1, 32, 6]
        # ind  = [1, 32, 6]
        # pred = [1, 32, 6, 2]
        # pred = _transpose_and_gather_feat_planes_np(output, ind, flag='Plane_ind') # (B, 32, 6, 2)
        # mask = mask.unsqueeze(3).expand_as(pred).float() # 将mask升维转换成和pred一样的形状 [1, 32, 6, 2]

        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        # loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        
        
        # v1 innovation loss
        # ================================================================================================== #
        loss = torch.abs(pred * mask - target * mask) # v1:[B, 32, 6, 4, 2]
        loss = torch.mean(loss, dim=[3,4]) # [B, 32, 6] 每个平面上的点计算loss之后取平均值，表示这个平面的loss.
        loss = torch.sum(loss,  dim=[2]) # [B, 32], 计算每个平面的平均loss之后，求和得到平面的总loss. 
        # ================================================================================================== #
        
        # v2 loss 
        # loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        
        loss = loss.sum() # 求出所有物体总loss
        loss = loss / (mask.sum() + 1e-4)
        return loss

class NormRegL1Loss(nn.Module):
    def __init__(self):
        super(NormRegL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        pred = pred / (target + 1e-4)
        target = target * 0 + 1
        #loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss

class RegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(RegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target,deps):
        dep=deps.squeeze(2) # (B,32,1) -> (B,32)
        dep[dep<5]=dep[dep<5]*0.01 # 将深度值小于5的物体，缩小0.01倍，再赋值回去
        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1 # 将深度值大于5的物体，-4 取对数 +0.1，再赋值回去  dep shape = [8, 32]
        
        # output = [B, 18, 96, 320]
        # ind    = [B, 32]
        # pred   = [B, 32, 18]
        pred = _transpose_and_gather_feat(output, ind) # 根据索引找到对应位置的特征值，shape=(Batch,32,18)
        mask = mask.float() # mask size = [B, 32, 18]
        
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss=torch.abs(pred * mask-target * mask)
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss=torch.abs(pred * mask-target * mask) # 根据label 计算l1 loss shape = [8, 32, 18]     
        loss = torch.sum(loss,dim=2)
        # losses = torch.mul(loss, dep) # 乘以深度权重loss shape = [8, 32], deep shape = [8,32]
    
        loss=loss.sum() # 求出所有物体的总loss
        loss = loss / (mask.sum() + 1e-4) # 求出每个物体的平均loss

        return loss

class PlanesRegWeightedL1Loss(nn.Module):
    def __init__(self):
        super(PlanesRegWeightedL1Loss, self).__init__()

    def forward(self, output, mask, ind, target, deps):
        
        dep=deps.squeeze(2) # (B,32,1) -> (B,32)
        dep[dep<5]=dep[dep<5]*0.01 # 将深度值小于5的物体，缩小0.2倍，再赋值回去
        dep[dep >= 5] = torch.log10(dep[dep >=5]-4)+0.1 # 将深度值大于5的物体，-4 取对数 +0.1，再赋值回去 dep shape = [8, 32]
        # output = [B, 6, 8, 96, 320]
        # ind    = [B, 32]
        # pred   = [B, 32, 6, 8]
        pred = _transpose_and_gather_feat_planes_np(output, ind, flag='np_ind') # 根据索引找到对应位置的特征值，shape=(Batch,32,6,8)
        mask = mask.float() # mask  =  [B, 32, 6, 8]
        
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = torch.abs(pred * mask-target * mask)
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        
        loss=torch.abs(pred * mask-target * mask) # 根据label 计算l1 loss shape = [B, 32, 6, 8]     
        # loss innovation
        # ================================================================================================== #
        loss = torch.mean(loss, dim=[3]) # [B, 32, 6] 每个平面上的点计算loss之后取平均值，表示这个平面的loss.
        loss = torch.sum(loss,  dim=[2]) # [B, 32], 计算每个平面的平均loss之后，求和得到平面的总loss. 
        # ================================================================================================== #
        
        loss = torch.mul(loss, dep) # 乘以深度权重 loss = [B, 32] dep = [B, 32]
        loss = loss.sum() # 求出所有物体的总loss
        loss = loss / (mask.sum() + 1e-4) # 求出每个物体的平均loss
        
        return loss
    
# class RegWeightedL1Loss(nn.Module):
#     def __init__(self):
#         super(RegWeightedL1Loss, self).__init__()
#
#     def forward(self, output, mask, ind, target):
#         pred = _transpose_and_gather_feat(output, ind)\
#         mask = mask.float()
#         # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
#         loss = F.l1_loss(pred * mask, target * mask, size_average=False)
#         loss = loss / (mask.sum() + 1e-4)
#         return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        #loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='mean')
        return loss


class depLoss(nn.Module):
    def __init__(self):
        super(depLoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _transpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss= torch.log(torch.abs((target * mask)-(pred * mask))).mean()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        loss = F.l1_loss(pred * mask, target * mask, reduction='mean')
        return loss


class BinRotLoss(nn.Module):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = _transpose_and_gather_feat(output, ind) # 根据索引找到对应点的特征
        loss = compute_rot_loss(pred, rotbin, rotres, mask) # 计算方向loss
        return loss


def compute_res_loss(output, target):
    #return F.smooth_l1_loss(output, target, reduction='elementwise_mean')
    return F.smooth_l1_loss(output, target, reduction='mean')


# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output) # 将mask转成output[0:2]前两维形状 (32,1) ->(32,2)
    output = output * mask.float()
    #return F.cross_entropy(output, target, reduction='elementwise_mean')
    return F.cross_entropy(output, target, reduction='mean')


class Position_loss(nn.Module):
    def __init__(self, opt):
        super(Position_loss, self).__init__()
        self.box_cpt_coef = torch.Tensor([[1/2, 1/2, 1/2],
                                          [1/2, 1/2,-1/2],                          
                                          [-1/2,1/2,-1/2],
                                          [-1/2,1/2, 1/2],   
                                    
                                          [1/2,-1/2,1/2],
                                          [1/2,-1/2,-1/2],
                                          [-1/2,-1/2,-1/2],
                                          [-1/2,-1/2,1/2],
                                          [0,0,0]])
        self.right_cpt_coef = self.box_cpt_coef.clone()
        self.right_cpt_coef[:,0]  = self.box_cpt_coef[:,0] - 1/2
        
        self.left_cpt_coef = self.box_cpt_coef.clone()
        self.left_cpt_coef[:,0]   = self.box_cpt_coef[:,0] + 1/2
               
        self.bottom_cpt_coef = self.box_cpt_coef.clone()
        self.bottom_cpt_coef[:,1] = self.box_cpt_coef[:,1] - 1/2
        
        self.top_cpt_coef = self.box_cpt_coef.clone()
        self.top_cpt_coef[:,1]  = self.box_cpt_coef[:,1] + 1/2
               
        self.back_cpt_coef = self.box_cpt_coef.clone()
        self.back_cpt_coef[:,2]   = self.box_cpt_coef[:,2] - 1/2
        
        self.front_cpt_coef = self.box_cpt_coef.clone()
        self.front_cpt_coef[:,2]  = self.box_cpt_coef[:,2] + 1/2

        const = torch.Tensor(
                            [[-1, 0], [0, -1], 
                             [-1, 0], [0, -1], 
                             [-1, 0], [0, -1], 
                             [-1, 0], [0, -1], 
                             [-1, 0], [0, -1], 
                             [-1, 0], [0, -1],
                             [-1, 0], [0, -1], 
                             [-1, 0], [0, -1], 
                             [-1, 0], [0, -1]])

        self.const = const.unsqueeze(0).unsqueeze(0)  # b,c,2 shape = [1, 1, 18, 2]
        
        self.bottom_face_index = torch.Tensor([0,1,2,3]).long()
        self.top_face_index    = torch.Tensor([4,5,6,7]).long()
        
        self.left_face_index   = torch.Tensor([2,3,6,7]).long()
        self.right_face_index  = torch.Tensor([0,1,4,5]).long()
        
        self.back_face_index   = torch.Tensor([0,3,4,7]).long()
        self.front_face_index  = torch.Tensor([1,2,5,6]).long()
        
        self.opt = opt
        self.num_joints = 9
        self.n_num_joints = opt.planes_n_kps
        self.planes = 6
        
        planes_const = torch.Tensor([[-1, 0], [0, -1]])
        planes_const = planes_const.repeat((self.n_num_joints,1)).unsqueeze(0).repeat((self.planes,1,1))
        self.planes_const = planes_const.unsqueeze(0).unsqueeze(0)  # b,c,2
        
    def forward(self, output, batch,phase=None):
        dim = _transpose_and_gather_feat(output['dim'], batch['ind']) # (B,32,3)
        rot = _transpose_and_gather_feat(output['rot'], batch['ind']) # (B,32,8)
        probability = _transpose_and_gather_feat(output['prob'], batch['ind']) # (B,32,1)
        kps = _transpose_and_gather_feat(output['hps'], batch['ind']) # (B,32,18)
        
        planes_n_kps = _transpose_and_gather_feat_planes_np(output['planes_n_kps'], batch['ind'], flag='np_ind') # (B,32,6,8)

        rot=rot.detach()### solving............

        b = dim.size(0) # batch_size
        c = dim.size(1) # 通道 32
        # prior, discard in multi-class training
        # dim[:, :, 0] = torch.exp(dim[:, :, 0]) * 1.63
        # dim[:, :, 1] = torch.exp(dim[:, :, 1]) * 1.53
        # dim[:, :, 2] = torch.exp(dim[:, :, 2]) * 3.88
        zeros = torch.zeros(b,c,1).cuda() # [b*c]
        ones  = torch.ones(b,c,1).cuda() #  [b*c]

        mask = batch['hps_mask'] #(B,32,18)
        mask = mask.float()
        n_mask = batch['planes_n_mask'] #(B,32,6,8)
        n_mask = n_mask.float()
        calib = batch['calib'] # (B,3,4)
        opinv = batch['opinv'] # (B,2,3)
        #=====================================================================#
        planes_n_kps_3d_coef = batch['planes_n_kps_3d_coef'] # (B, 32, 6, 4, 3)
        #=====================================================================#
        
        cys = (batch['ind'] / self.opt.output_w).int().float() # (B,32)
        cxs = (batch['ind'] % self.opt.output_w).int().float() # (B,32)
        #  kps[..., ::2] = (B,32,9)      cxs.view(b, c, 1) = (B,32,1)  -> expand(b, c, self.num_joints) = (B,32,9)
        
        opinv = opinv.unsqueeze(1) # (B,2,3)->(B,1,2,3)
        kps_opinv = opinv.expand(b, c, -1, -1).contiguous().view(-1, 2, 3).float() # (B,32,2,3) ->(B*32,2,3)
        #===============================================#
        planes_opinv = opinv.unsqueeze(2).expand(b, c, self.planes, -1, -1).contiguous().view(-1, 2, 3).float() # (B,32,6,2,3) ->(B*32*6, 2, 3)
        #===============================================#
        
        # ...表示遍历每行，2表示步长，选取多索引为0，2，4所在的列
        kps[..., ::2] = kps[..., ::2] + cxs.view(b, c, 1).expand(b, c, self.num_joints) # x坐标(B,32,9)
        kps[..., 1::2] = kps[..., 1::2] + cys.view(b, c, 1).expand(b, c, self.num_joints) # y坐标(B,32,9)
        # 将最后一列拆分成两列
        kps = kps.view(b, c, -1, 2).permute(0, 1, 3, 2) # (B,32,18) -> (B, 32, 2, 9)
 
        hom = torch.ones(b, c, 1, 9).cuda() # (B,32,1,9)
        kps = torch.cat((kps, hom), dim=2).view(-1, 3, 9) # (B*32,3,9)
        kps = torch.bmm(kps_opinv, kps).view(b, c, 2, 9) # >(B*32,2,3) * (B*32,3,9) = (B,32,2,9)
        kps = kps.permute(0, 1, 3, 2).contiguous().view(b, c, -1)  # (B,32,18)
        
        #======================================================================#
        planes_n_kps[..., ::2] = planes_n_kps[..., ::2] + cxs.view(b, c, 1, 1).expand(b, c, self.planes,self.n_num_joints) # x坐标(B,32,9)
        planes_n_kps[..., 1::2] = planes_n_kps[..., 1::2] + cys.view(b, c, 1, 1).expand(b, c, self.planes, self.n_num_joints) # y坐标(B,32,9)
        planes_n_kps = planes_n_kps.view(b, c, self.planes, -1, 2).permute(0, 1, 2, 4, 3) # (B,32,6,8) -> (B, 32, 6, 2, 4)
        
        planes_hom = torch.ones(b, c, self.planes,  1, self.n_num_joints).cuda() # (B, 32, 1, 2, 4)
        planes_n_kps = torch.cat((planes_n_kps, planes_hom), dim=3).view(-1, 3, self.n_num_joints) # (B*32*6,3,4)
        planes_n_kps = torch.bmm(planes_opinv, planes_n_kps).view(b, c, self.planes, 2, self.n_num_joints) # >(B*32*6,2,3) * (B*32*6,3,4) = (B,32,6,2,4)
        planes_n_kps = planes_n_kps.permute(0, 1, 2, 4, 3).contiguous().view(b, c, self.planes, -1)  # (B,32,6,8)
        #======================================================================#
        
        
        si = torch.zeros_like(kps[:, :, 0:1]) + calib[:, 0:1, 0:1] # # (B,32,1)
        alpha_idx = rot[:, :, 1] > rot[:, :, 5] #  bin1_cls[1] >  bin2_cls[1]
        alpha_idx = alpha_idx.float()
        alpha1 = torch.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi) # bin1_sin / bin1_cos (1,32)
        alpha2 = torch.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)  # bin2_sin / bin2_cos (1,32)
        alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx) # (1,32)
        alpna_pre = alpna_pre.unsqueeze(2) # (1,32,1)


        rot_y = alpna_pre + torch.atan2(kps[:, :, 16:17] - calib[:, 0:1, 2:3], si) # (1,32,1)
        rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi # 将rot_y角度大于pi的减去2pi
        rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi #将rot_y角度小于pi的加上2pi

        calib = calib.unsqueeze(1) # (1, 3, 4) -> (1, 1, 3, 4)
        calib = calib.expand(b, c, -1, -1).contiguous() # (1,1,3,4) ->(1,32,3,4)
        kpoint = kps
        planes_n_kpoints = planes_n_kps
        
        
        fs = calib[:, :, 0, 0].unsqueeze(2) # (1, 32, 1)
        f = fs.expand_as(kpoint) # (1,32,18)
        f_planes = fs.unsqueeze(3).expand_as(planes_n_kpoints) # (1,32,6,8)
        
        cx, cy = calib[:, :, 0, 2].unsqueeze(2), calib[:, :, 1, 2].unsqueeze(2) # (1, 32, 1), (1, 32, 1)
        cxy = torch.cat((cx, cy), dim=2) # (1, 32, 2)
        kcxy = cxy.repeat(1, 1, 9)  # b,c,16 (1,32,18)
        
        #===============================================#
        planes_n_cxy = cxy.unsqueeze(2).repeat(1, 1, self.planes, self.n_num_joints)  # b,c,16 (1,32,6,8)
        #===============================================#
        kp_norm = (kpoint - kcxy) / f # (1, 32, 18)
        #===============================================#
        planes_n_norm = (planes_n_kpoints - planes_n_cxy) / f_planes # (1, 32, 6, 8)
        #===============================================#
        
        l = dim[:, :, 2:3] # (1, 32, 1)
        h = dim[:, :, 0:1] # (1, 32, 1)
        w = dim[:, :, 1:2] # (1, 32, 1)
        
        #===============================================#
        zero = torch.zeros_like(l)
        lh = torch.cat((h,zero,l),dim=2).unsqueeze(2)
        hw = torch.cat((h,w,zero),dim=2).unsqueeze(2)
        wl = torch.cat((zero,w,l),dim=2).unsqueeze(2)
        planes_dim = torch.cat([hw,hw,wl,wl,lh,lh],dim=2)
        #===============================================#
        
        cosori = torch.cos(rot_y) # (1, 32, 1)
        sinori = torch.sin(rot_y) # (1, 32, 1)

        B = torch.zeros_like(kpoint) # (1,32,18)
        C = torch.zeros_like(kpoint) # (1,32,18)
        
        B_bottom = torch.zeros_like(kpoint) # (1,32,18)
        C_bottom = torch.zeros_like(kpoint) # (1,32,18)
        B_top = torch.zeros_like(kpoint) # (1,32,18)
        C_top = torch.zeros_like(kpoint) # (1,32,18)
        B_right = torch.zeros_like(kpoint) # (1,32,18)
        C_right = torch.zeros_like(kpoint) # (1,32,18)
        B_left = torch.zeros_like(kpoint) # (1,32,18)
        C_left = torch.zeros_like(kpoint) # (1,32,18)
        B_back = torch.zeros_like(kpoint) # (1,32,18)
        C_back = torch.zeros_like(kpoint) # (1,32,18)
        B_front = torch.zeros_like(kpoint) # (1,32,18)
        C_front = torch.zeros_like(kpoint) # (1,32,18)

        kp = kp_norm.unsqueeze(3)  # b,c,16,1  (1, 32, 18, 1)
 
        const = self.const.cuda()  # (1, 1, 18, 2)
        const = const.expand(b, c, -1, -1) # (1, 32, 18, 2)

        box_cpt_coef = self.box_cpt_coef.cuda()
        
        bottom_cpt_coef = self.bottom_cpt_coef.cuda()
        top_cpt_coef = self.top_cpt_coef.cuda()
        
        right_cpt_coef = self.right_cpt_coef.cuda()
        left_cpt_coef = self.left_cpt_coef.cuda()
        
        back_cpt_coef = self.back_cpt_coef.cuda()
        front_cpt_coef = self.front_cpt_coef.cuda()

        '''
        A = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
 
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
        
        B[:, :, 16:17] = 0
        B[:, :, 17:18] = 0
        
       
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
        
        C[:, :, 16:17] = 0
        C[:, :, 17:18] = 0
        '''
	
        A_temp = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
        
        A_bottom = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
        A_top = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
        A_right = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
        A_left = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
        A_back = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
        A_front = torch.cat([const, kp], dim=3)  # (1, 32, 18, 3)
	
	# 随机选取N个点求取svd伪逆
        A = torch.zeros_like(A_temp)
        index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8], size=4, replace = False))
        # for i in index:
        #     A[:,:,i*2:i*2+1]   = A_temp[:,:,i*2:i*2+1]
        #     A[:,:,i*2+1:i*2+2] = A_temp[:,:,i*2+1:i*2+2]
        for i in range(9):
            A[:,:,i*2:i*2+1]   = A_temp[:,:,i*2:i*2+1]
            A[:,:,i*2+1:i*2+2] = A_temp[:,:,i*2+1:i*2+2]
        for i in range(9): # 体心
            B[:,:,i*2:i*2+1]   = l * box_cpt_coef[i,0:1] * cosori + w * box_cpt_coef[i,2:3] * sinori
            B[:,:,i*2+1:i*2+2] = h * box_cpt_coef[i,1:2]
        for i in range(9):
            C[:,:,i*2:i*2+2]   = l * box_cpt_coef[i,0:1] * (-sinori) + w * box_cpt_coef[i,2:3] * cosori

        for i in range(9): # 右心
            B_right[:,:,i*2:i*2+1]   = l * right_cpt_coef[i,0:1] * cosori + w * right_cpt_coef[i,2:3] * sinori
            B_right[:,:,i*2+1:i*2+2] = h * right_cpt_coef[i,1:2]
        for i in range(9):
            C_right[:,:,i*2:i*2+2]   = l * right_cpt_coef[i,0:1] * (-sinori) + w * right_cpt_coef[i,2:3] * cosori    
            
        for i in range(9): # 左心
            B_left[:,:,i*2:i*2+1]   = l * left_cpt_coef[i,0:1] * cosori + w * left_cpt_coef[i,2:3] * sinori
            B_left[:,:,i*2+1:i*2+2] = h * left_cpt_coef[i,1:2]
        for i in range(9):
            C_left[:,:,i*2:i*2+2]   = l * left_cpt_coef[i,0:1] * (-sinori) + w * left_cpt_coef[i,2:3] * cosori     
            
        for i in range(9): # 底心
            B_bottom[:,:,i*2:i*2+1]   = l * bottom_cpt_coef[i,0:1] * cosori + w * bottom_cpt_coef[i,2:3] * sinori
            B_bottom[:,:,i*2+1:i*2+2] = h * bottom_cpt_coef[i,1:2]
        for i in range(9):
            C_bottom[:,:,i*2:i*2+2]   = l * bottom_cpt_coef[i,0:1] * (-sinori) + w * bottom_cpt_coef[i,2:3] * cosori
        
        for i in range(9): # 顶心
            B_top[:,:,i*2:i*2+1]   = l * top_cpt_coef[i,0:1] * cosori + w * top_cpt_coef[i,2:3] * sinori
            B_top[:,:,i*2+1:i*2+2] = h * top_cpt_coef[i,1:2]
        for i in range(9):
            C_top[:,:,i*2:i*2+2]   = l * top_cpt_coef[i,0:1] * (-sinori) + w * top_cpt_coef[i,2:3] * cosori
          
        for i in range(9): # 背心
            B_back[:,:,i*2:i*2+1]   = l * back_cpt_coef[i,0:1] * cosori + w * back_cpt_coef[i,2:3] * sinori
            B_back[:,:,i*2+1:i*2+2] = h * back_cpt_coef[i,1:2]
        for i in range(9):
            C_back[:,:,i*2:i*2+2]   = l * back_cpt_coef[i,0:1] * (-sinori) + w * back_cpt_coef[i,2:3] * cosori 
        
        for i in range(9): # 正心
            B_front[:,:,i*2:i*2+1]   = l * front_cpt_coef[i,0:1] * cosori + w * front_cpt_coef[i,2:3] * sinori
            B_front[:,:,i*2+1:i*2+2] = h * front_cpt_coef[i,1:2]
        for i in range(9):
            C_front[:,:,i*2:i*2+2]   = l * front_cpt_coef[i,0:1] * (-sinori) + w * front_cpt_coef[i,2:3] * cosori 

        B = B - kp_norm * C # (1,32,18)
        
        B_right  = B_right  - kp_norm * C_right # (1,32,18)
        B_left   = B_left   - kp_norm * C_left # (1,32,18)
        B_bottom = B_bottom - kp_norm * C_bottom # (1,32,18)
        B_top    = B_top    - kp_norm * C_top # (1,32,18)
        B_back   = B_back   - kp_norm * C_back # (1,32,18)
        B_front  = B_front  - kp_norm * C_front # (1,32,18)

        kps_mask = mask # (1,32,18)
        mask2 = torch.sum(kps_mask, dim=2) # (1,32)
        loss_mask = mask2 > 15 # (1,32)
        loss_mask = loss_mask.float()

        AT = A.permute(0, 1, 3, 2) # (1,32,3,18)
        AT = AT.view(b * c, 3, 18) # (32,3,18)
        A = A.view(b * c, 18, 3) # (32, 18, 3)
        B = B.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv = torch.bmm(AT, A) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv = torch.inverse(pinv)  # b*c 3 3 (32,3,3)
        pinv = torch.bmm(pinv, AT) # (32, 3, 18)
        pinv = torch.bmm(pinv, B)  # (32, 3, 1)
        pinv = pinv.view(b, c, 3, 1).squeeze(3) # (1, 32, 3)

    
        AT_right = A_right.permute(0, 1, 3, 2) # (1,32,3,18)
        AT_right = AT_right.view(b * c, 3, 18) # (32,3,18)
        A_right = A_right.view(b * c, 18, 3) # (32, 18, 3)
        B_right = B_right.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv_right = torch.bmm(AT_right, A_right) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv_right = torch.inverse(pinv_right)  # b*c 3 3 (32,3,3)
        pinv_right = torch.bmm(pinv_right, AT_right) # (32, 3, 18)
        pinv_right = torch.bmm(pinv_right, B_right)  # (32, 3, 1)
        pinv_right = pinv_right.view(b, c, 1, 3) # (1, 32, 1, 3)
        
        AT_left = A_left.permute(0, 1, 3, 2) # (1,32,3,18)
        AT_left = AT_left.view(b * c, 3, 18) # (32,3,18)
        A_left = A_left.view(b * c, 18, 3) # (32, 18, 3)
        B_left = B_left.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv_left = torch.bmm(AT_left, A_left) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv_left = torch.inverse(pinv_left)  # b*c 3 3 (32,3,3)
        pinv_left = torch.bmm(pinv_left, AT_left) # (32, 3, 18)
        pinv_left = torch.bmm(pinv_left, B_left)  # (32, 3, 1)
        pinv_left = pinv_left.view(b, c, 1, 3) # (1, 32, 1, 3)
        
        AT_bottom = A_bottom.permute(0, 1, 3, 2) # (1,32,3,18)
        AT_bottom = AT_bottom.view(b * c, 3, 18) # (32,3,18)
        A_bottom = A_bottom.view(b * c, 18, 3) # (32, 18, 3)
        B_bottom = B_bottom.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv_bottom = torch.bmm(AT_bottom, A_bottom) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv_bottom = torch.inverse(pinv_bottom)  # b*c 3 3 (32,3,3)
        pinv_bottom = torch.bmm(pinv_bottom, AT_bottom) # (32, 3, 18)
        pinv_bottom = torch.bmm(pinv_bottom, B_bottom)  # (32, 3, 1)
        pinv_bottom = pinv_bottom.view(b, c, 1, 3) # (1, 32, 1, 3)        
        
        AT_top = A_top.permute(0, 1, 3, 2) # (1,32,3,18)
        AT_top = AT_top.view(b * c, 3, 18) # (32,3,18)
        A_top = A_top.view(b * c, 18, 3) # (32, 18, 3)
        B_top = B_top.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv_top = torch.bmm(AT_top, A_top) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv_top = torch.inverse(pinv_top)  # b*c 3 3 (32,3,3)
        pinv_top = torch.bmm(pinv_top, AT_top) # (32, 3, 18)
        pinv_top = torch.bmm(pinv_top, B_top)  # (32, 3, 1)
        pinv_top = pinv_top.view(b, c, 1, 3) # (1, 32, 1, 3)
            
        AT_back = A_back.permute(0, 1, 3, 2) # (1,32,3,18)
        AT_back = AT_back.view(b * c, 3, 18) # (32,3,18)
        A_back = A_back.view(b * c, 18, 3) # (32, 18, 3)
        B_back = B_back.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv_back = torch.bmm(AT_back, A_back) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv_back = torch.inverse(pinv_back)  # b*c 3 3 (32,3,3)
        pinv_back = torch.bmm(pinv_back, AT_back) # (32, 3, 18)
        pinv_back = torch.bmm(pinv_back, B_back)  # (32, 3, 1)
        pinv_back = pinv_back.view(b, c, 1, 3) # (1, 32, 1, 3)
        
        AT_front = A_front.permute(0, 1, 3, 2) # (1,32,3,18)
        AT_front = AT_front.view(b * c, 3, 18) # (32,3,18)
        A_front = A_front.view(b * c, 18, 3) # (32, 18, 3)
        B_front = B_front.view(b * c, 18, 1).float() # (32,18,1)
        # mask = mask.unsqueeze(2)
        pinv_front = torch.bmm(AT_front, A_front) # (32,3,18) * (32, 18, 3) = (32, 3, 3)
        pinv_front = torch.inverse(pinv_front)  # b*c 3 3 (32,3,3)
        pinv_front = torch.bmm(pinv_front, AT_front) # (32, 3, 18)
        pinv_front = torch.bmm(pinv_front, B_front)  # (32, 3, 1)
        pinv_front = pinv_front.view(b, c, 1, 3) # (1, 32, 1, 3)

        
        corner_1 = torch.cat([pinv[:, :, 0:1] + dim[:, :, 2:] / 2, pinv[:, :, 1:2] + dim[:, :, 0:1] / 2,  pinv[:, :, 2:] + dim[:, :, 1:2] / 2,], dim = 2) 
        corner_2 = torch.cat([pinv[:, :, 0:1] + dim[:, :, 2:] / 2, pinv[:, :, 1:2] + dim[:, :, 0:1] / 2,  pinv[:, :, 2:] - dim[:, :, 1:2] / 2,], dim = 2) 
        corner_3 = torch.cat([pinv[:, :, 0:1] - dim[:, :, 2:] / 2, pinv[:, :, 1:2] + dim[:, :, 0:1] / 2,  pinv[:, :, 2:] - dim[:, :, 1:2] / 2,], dim = 2) 
        corner_4 = torch.cat([pinv[:, :, 0:1] - dim[:, :, 2:] / 2, pinv[:, :, 1:2] + dim[:, :, 0:1] / 2,  pinv[:, :, 2:] + dim[:, :, 1:2] / 2,], dim = 2) 
        corner_5 = torch.cat([pinv[:, :, 0:1] + dim[:, :, 2:] / 2, pinv[:, :, 1:2] - dim[:, :, 0:1] / 2,  pinv[:, :, 2:] + dim[:, :, 1:2] / 2,], dim = 2) 
        corner_6 = torch.cat([pinv[:, :, 0:1] + dim[:, :, 2:] / 2, pinv[:, :, 1:2] - dim[:, :, 0:1] / 2,  pinv[:, :, 2:] - dim[:, :, 1:2] / 2,], dim = 2) 
        corner_7 = torch.cat([pinv[:, :, 0:1] - dim[:, :, 2:] / 2, pinv[:, :, 1:2] - dim[:, :, 0:1] / 2,  pinv[:, :, 2:] - dim[:, :, 1:2] / 2,], dim = 2) 
        corner_8 = torch.cat([pinv[:, :, 0:1] - dim[:, :, 2:] / 2, pinv[:, :, 1:2] - dim[:, :, 0:1] / 2,  pinv[:, :, 2:] + dim[:, :, 1:2] / 2,], dim = 2) 

        # x axis vector
        vector_1 = (corner_2 - corner_3) 
        vector_2 = (corner_1 - corner_4)
        vector_3 = (corner_5 - corner_8)
        vector_4 = (corner_6 - corner_7)  

        # y axis vector
        vector_5 = (corner_1 - corner_5) 
        vector_6 = (corner_2 - corner_6)
        vector_7 = (corner_3 - corner_7)
        vector_8 = (corner_4 - corner_8) 

        # z axis vector
        vector_9  = (corner_1 - corner_2) 
        vector_10 = (corner_4 - corner_3)
        vector_11 = (corner_5 - corner_6)
        vector_12 = (corner_8 - corner_7)

        # horizon
        hv_loss_mask = loss_mask.unsqueeze(2)
        horizon_cos_theata_1 = torch.abs(calculate_angle(vector_1, vector_2)) + torch.abs(calculate_angle(vector_1, vector_3)) + torch.abs(calculate_angle(vector_1, vector_4)) + torch.abs(calculate_angle(vector_2, vector_3)) + torch.abs(calculate_angle(vector_2, vector_4)) + torch.abs(calculate_angle(vector_3, vector_4))
        horizon_cos_theata_2 = torch.abs(calculate_angle(vector_5, vector_6)) + torch.abs(calculate_angle(vector_5, vector_7)) + torch.abs(calculate_angle(vector_5, vector_8)) + torch.abs(calculate_angle(vector_6, vector_7)) + torch.abs(calculate_angle(vector_6, vector_8)) + torch.abs(calculate_angle(vector_7, vector_8))
        horizon_cos_theata_3 = torch.abs(calculate_angle(vector_9, vector_10)) + torch.abs(calculate_angle(vector_9, vector_11)) + torch.abs(calculate_angle(vector_9, vector_12)) + torch.abs(calculate_angle(vector_10, vector_11)) + torch.abs(calculate_angle(vector_10, vector_12)) + torch.abs(calculate_angle(vector_11, vector_12))

        
        horizon_cos_theata = horizon_cos_theata_1 + horizon_cos_theata_2 + horizon_cos_theata_3 # (B*32*1)
        horizon_cos_theata = horizon_cos_theata.contiguous().view(b,c,-1) # (B,32,1)

        horizon_cos_theata_loss = F.l1_loss(horizon_cos_theata * hv_loss_mask, ones*hv_loss_mask,reduction='sum') # loss -> 0
        horizon_cos_theata_loss = horizon_cos_theata_loss / (hv_loss_mask.sum() + 1)
        
        # vertical
        vertical_cos_theata_1 = calculate_angle(vector_1, vector_5) + calculate_angle(vector_1, vector_6) + calculate_angle(vector_1, vector_7) + calculate_angle(vector_1, vector_8) + \
                                calculate_angle(vector_2, vector_5) + calculate_angle(vector_2, vector_6) + calculate_angle(vector_2, vector_7) + calculate_angle(vector_2, vector_8) + \
                                calculate_angle(vector_3, vector_5) + calculate_angle(vector_3, vector_6) + calculate_angle(vector_3, vector_7) + calculate_angle(vector_3, vector_8) + \
                                calculate_angle(vector_4, vector_5) + calculate_angle(vector_4, vector_6) + calculate_angle(vector_4, vector_7) + calculate_angle(vector_4, vector_8) 

        vertical_cos_theata_2 = calculate_angle(vector_1, vector_9) + calculate_angle(vector_1, vector_10) + calculate_angle(vector_1, vector_11) + calculate_angle(vector_1, vector_12) + \
                                calculate_angle(vector_2, vector_9) + calculate_angle(vector_2, vector_10) + calculate_angle(vector_2, vector_11) + calculate_angle(vector_2, vector_12) + \
                                calculate_angle(vector_3, vector_9) + calculate_angle(vector_3, vector_10) + calculate_angle(vector_3, vector_11) + calculate_angle(vector_3, vector_12) + \
                                calculate_angle(vector_4, vector_9) + calculate_angle(vector_4, vector_10) + calculate_angle(vector_4, vector_11) + calculate_angle(vector_4, vector_12) 

        vertical_cos_theata_3 = calculate_angle(vector_5, vector_9) + calculate_angle(vector_5, vector_10) + calculate_angle(vector_5, vector_11) + calculate_angle(vector_5, vector_12) + \
                                calculate_angle(vector_6, vector_9) + calculate_angle(vector_6, vector_10) + calculate_angle(vector_6, vector_11) + calculate_angle(vector_6, vector_12) + \
                                calculate_angle(vector_7, vector_9) + calculate_angle(vector_7, vector_10) + calculate_angle(vector_7, vector_11) + calculate_angle(vector_7, vector_12) + \
                                calculate_angle(vector_8, vector_9) + calculate_angle(vector_8, vector_10) + calculate_angle(vector_8, vector_11) + calculate_angle(vector_8, vector_12) 

        vertical_cos_theata = vertical_cos_theata_1 + vertical_cos_theata_2 + vertical_cos_theata_3
        vertical_cos_theata = vertical_cos_theata.contiguous().view(b,c,-1)

        vertical_cos_theata_loss = F.l1_loss(vertical_cos_theata * hv_loss_mask, zeros*hv_loss_mask,reduction='sum') # loss -> 0
        vertical_cos_theata_loss = vertical_cos_theata_loss / (hv_loss_mask.sum() + 1)
        


        # change the center to kitti center. Note that the pinv is the 3D center point in the camera coordinate system
        pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2

        pinv_planes = torch.cat((pinv_right, pinv_left, pinv_bottom, pinv_top, pinv_back, pinv_front), dim=2) # (1,32,6,3)
        
        # min_value_dim = 0.2
        dim_mask = dim<0 # (1,32,3)
        dim = torch.clamp(dim, 0 , 10) # (1,32,3)
        dim_mask_score_mask = torch.sum(dim_mask, dim=2) # (1,32)
        dim_mask_score_mask = 1 - (dim_mask_score_mask > 0) #(1,32)
        # dim_mask_score_mask = ~(dim_mask_score_mask > 0)
        dim_mask_score_mask = dim_mask_score_mask.float()
        
        box_pred = torch.cat((pinv, dim, rot_y), dim=2).detach() # (1,32,7)
        
        loss = (pinv - batch['location']) # (1,32,3)
        loss_norm = torch.norm(loss, p=2, dim=2) # (1,32)
        loss = loss_norm * loss_mask #  (1, 32)
        mask_num = (loss != 0).sum() #  mask_num =1
        loss = loss.sum() / (mask_num + 1) # loss = 29.2169
        
        plane_center_loss = pinv_planes -  batch['planes_cpt'] #   (1,32,6,3)
        plane_center_loss_norm = torch.norm(plane_center_loss, p=2, dim=3)  #   (1,32,6)
        plane_center_loss_norm = torch.norm(plane_center_loss_norm, p=1, dim=2)  #   (1,32)
        plane_center_loss = plane_center_loss_norm * loss_mask
        plane_mask_num = (plane_center_loss != 0).sum() #  mask_num =1
        plane_loss = plane_center_loss.sum() / (plane_mask_num + 1) # loss = 29.2169


        plane_loss = plane_loss + 1.0*horizon_cos_theata_loss + 1.0*vertical_cos_theata_loss
        '''
        # planes possiontion loss
        Planes_B = torch.zeros_like(planes_n_kpoints) # [1, 32, 6, 8]
        Planes_C = torch.zeros_like(planes_n_kpoints) # [1, 32, 6, 8]
        planes_kp = planes_n_norm.unsqueeze(4)  # b,c,16,1  (1, 32, 6, 8, 1)
        planes_const = self.planes_const.cuda()  # (1, 1, 48, 2)
        planes_const = planes_const.expand(b, c, -1, -1, -1) # [1, 32, 6, 8, 2]
        Planes_A = torch.cat([planes_const, planes_kp], dim=4)  # (1, 32, 6, 8, 3)
        for i in range(self.n_num_joints):
            Planes_B[:,:,:,i*2:i*2+1] = l.unsqueeze(2).repeat(1,1,self.planes,1) * planes_n_kps_3d_coef[:,:,:,i,0:1] * cosori.unsqueeze(2).repeat(1,1,self.planes,1) + w.unsqueeze(2).repeat(1,1,self.planes,1) * planes_n_kps_3d_coef[:,:,:,i,2:3] * sinori.unsqueeze(2).repeat(1,1,self.planes,1)
            Planes_B[:,:,:,i*2+1:i*2+2] = h.unsqueeze(2).repeat(1,1,self.planes,1) * planes_n_kps_3d_coef[:,:,:,i,1:2]
        for i in range(self.n_num_joints):
            Planes_C[:,:,:,i*2:i*2+2] = l.unsqueeze(2).repeat(1,1,self.planes,1) * planes_n_kps_3d_coef[:,:,:,i,0:1] * (-sinori.unsqueeze(2).repeat(1,1,self.planes,1)) + w.unsqueeze(2).repeat(1,1,self.planes,1) * planes_n_kps_3d_coef[:,:,:,i,1:2] * cosori.unsqueeze(2).repeat(1,1,self.planes,1)
        Planes_B = Planes_B - planes_n_norm * Planes_C
        p_mask = n_mask # [1, 32, 6, 8]
        
        Planes_AT = Planes_A.permute(0,1,2,4,3) # (1, 32, 6, 3, 8)
        # Planes_AT = Planes_A.permute(0,1,4,2,3) # (1, 32, 3, 6, 8)
        Planes_AT = Planes_AT.view(b * c * self.planes, 3, self.n_num_joints * 2) # (B*32*6,3,8)
        # Planes_AT = Planes_AT.view(b * c, 3, self.planes * self.n_num_joints * 2) # (B*32,3,6*8)
        Planes_A = Planes_A.view(b * c * self.planes, self.n_num_joints * 2, 3) # (B*32*6,8,3)
        # Planes_A = Planes_A.view(b * c , self.planes*self.n_num_joints * 2, 3) # (B*32,6*8,3)  
        Planes_B = Planes_B.view(b * c * self.planes, self.n_num_joints * 2, 1).float() # (B*32*6,8,1)
        # Planes_B = Planes_B.view(b * c , self.planes*self.n_num_joints * 2, 1).float() # (B*32,6*8,1) 
        planes_pinv = torch.bmm(Planes_AT, Planes_A) # (B*32*6, 3, 8) * (B*32*6, 8, 3) = (B*32*6, 3, 3) ##### (B*32，3, 6*8) * (B*32,6*8, 3) = (B*32*6, 3, 3)
        planes_pinv = torch.inverse(planes_pinv)  # b*c 3 3 (B*32*6,3,3) ##### (B*32,3,3)
        plane_mask = torch.sum(p_mask, dim=3) # (1,32,6)
        # plane_mask = torch.sum(p_mask, dim=[2,3]) # (1,32)     
        plane_loss_mask = plane_mask > 6 # (1, 32, 6)
        # plane_loss_mask = plane_mask > 40 # (1, 32)
        
        planes_pinv = torch.bmm(planes_pinv, Planes_AT) # (B*32*6, 3, 8) (B*32, 3, 8)
        planes_pinv = torch.bmm(planes_pinv, Planes_B)  # (B*32*6, 3, 1) (B*32, 3, 1)
        planes_pinv = planes_pinv.view(b, c, self.planes, 3, 1).squeeze(4) # (1, 32, 6, 3)
        
        
        right_cpt = planes_pinv[ :, :, 0, :] # right plane
        left_cpt = planes_pinv[ :, :, 1, :] # left plane
        bottom_cpt = planes_pinv[ :, :, 2, :] # bottom plane
        top_cpt = planes_pinv[ :, :, 3, :] # top plane
        back_cpt = planes_pinv[ :, :, 4, :] # back plane
        front_cpt = planes_pinv[ :, :, 5, :] # front plane

        # vertical
        plane_vertical_1 = right_cpt - left_cpt
        plane_vertical_2 = bottom_cpt - top_cpt
        plane_vertical_3 = back_cpt - front_cpt
        #vertical_cos_theata   = calculate_angle(plane_vertical_1, plane_vertical_2) + \
        #                        calculate_angle(plane_vertical_2, plane_vertical_3) + \
        #                        calculate_angle(plane_vertical_3, plane_vertical_1)
        #vertical_cos_theata = vertical_cos_theata.contiguous().view(b,c,-1)

        #vertical_cos_theata_loss = F.l1_loss(vertical_cos_theata * hv_loss_mask, zeros*hv_loss_mask,reduction='sum') # loss -> 0
        #vertical_cos_theata_loss = vertical_cos_theata_loss / (hv_loss_mask.sum() + 1)
        vertical_loss =  torch.abs(torch.sum(torch.mul(plane_vertical_1,plane_vertical_2),dim=2)) + \
                         torch.abs(torch.sum(torch.mul(plane_vertical_1,plane_vertical_3),dim=2)) + \
                         torch.abs(torch.sum(torch.mul(plane_vertical_2,plane_vertical_3),dim=2))  # (B,C)
        
        vertical_loss = vertical_loss * loss_mask # (1, 32, 6) ######## (1, 32)       
        vertical_mask_num = (vertical_loss != 0).sum() #  mask_num =1
        vertical_loss = vertical_loss.sum() / (vertical_mask_num + 1) # loss = 29.2169
        # horizon
        plane_horizon_1 =    top_cpt - right_cpt
        plane_horizon_2 =    top_cpt - left_cpt
        plane_horizon_3 =    top_cpt - back_cpt
        plane_horizon_4 =    top_cpt - front_cpt
        plane_horizon_5 = bottom_cpt - right_cpt
        plane_horizon_6 = bottom_cpt - left_cpt
        plane_horizon_7 = bottom_cpt - back_cpt
        plane_horizon_8 = bottom_cpt - front_cpt


        horizon_cos_theata =   torch.abs(calculate_angle(plane_horizon_1, plane_horizon_6)) + \
                               torch.abs(calculate_angle(plane_horizon_2, plane_horizon_5)) + \
                               torch.abs(calculate_angle(plane_horizon_3, plane_horizon_8)) + \
                               torch.abs(calculate_angle(plane_horizon_4, plane_horizon_7)) 

        horizon_cos_theata = horizon_cos_theata.contiguous().view(b,c,-1) # (B,32,1)

        horizon_cos_theata_loss = F.l1_loss(horizon_cos_theata * hv_loss_mask, 4*ones*hv_loss_mask,reduction='sum') # loss -> 0
        horizon_cos_theata_loss = horizon_cos_theata_loss / (hv_loss_mask.sum() + 1)

        x_1, y_1, z_1 = plane_horizon_1[:,:,0], plane_horizon_1[:,:,1],  plane_horizon_1[:,:,2]
        x_2, y_2, z_2 = plane_horizon_2[:,:,0], plane_horizon_2[:,:,1],  plane_horizon_2[:,:,2]
        x_3, y_3, z_3 = plane_horizon_3[:,:,0], plane_horizon_3[:,:,1],  plane_horizon_3[:,:,2]
        x_4, y_4, z_4 = plane_horizon_4[:,:,0], plane_horizon_4[:,:,1],  plane_horizon_4[:,:,2]
        x_5, y_5, z_5 = plane_horizon_5[:,:,0], plane_horizon_5[:,:,1],  plane_horizon_5[:,:,2]
        x_6, y_6, z_6 = plane_horizon_6[:,:,0], plane_horizon_6[:,:,1],  plane_horizon_6[:,:,2]
        x_7, y_7, z_7 = plane_horizon_7[:,:,0], plane_horizon_7[:,:,1],  plane_horizon_7[:,:,2]
        x_8, y_8, z_8 = plane_horizon_8[:,:,0], plane_horizon_8[:,:,1],  plane_horizon_8[:,:,2]
        horizon_1 = torch.abs(x_1*y_6 - x_6*y_1) + \
                    torch.abs(y_1*z_6 - y_6*z_1) + \
                    torch.abs(x_1*z_6 - x_6*z_1)    # (5,32)
        
        horizon_2 = torch.abs(x_2*y_5 - x_5*y_2) + \
                    torch.abs(y_2*z_5 - y_5*z_2) + \
                    torch.abs(x_2*z_5 - x_5*z_2)    # (5,32)

        horizon_3 = torch.abs(x_3*y_8 - x_8*y_3) + \
                    torch.abs(y_3*z_8 - y_8*z_3) + \
                    torch.abs(x_3*z_8 - x_8*z_3)    # (5,32)

        horizon_4 = torch.abs(x_4*y_7 - x_7*y_4) + \
                    torch.abs(y_4*z_7 - y_7*z_4) + \
                    torch.abs(x_4*z_7 - x_7*z_4)    # (5,32)
        
        horizon_loss = horizon_1 + horizon_2 + horizon_3 + horizon_4
        horizon_loss = horizon_loss * loss_mask # (1, 32, 6) ######## (1, 32)       
        horizon_mask_num = (horizon_loss != 0).sum() #  mask_num =1
        horizon_loss = horizon_loss.sum() / (horizon_mask_num + 1) # loss = 29.2169

        #planes_pinv = planes_pinv.view(b, c, 3, 1).squeeze(3) # (1, 32, 3)
        planes_pinv[ :, :, 0,1] = planes_pinv[ :, :, 0, 1] + dim[:, :, 0] / 2 # (B, 32, 6, 3) right plane
        planes_pinv[ :, :, 1,1] = planes_pinv[ :, :, 1, 1] + dim[:, :, 0] / 2 # (B, 32, 6, 3) left plane
        planes_pinv[ :, :, 2,1] = planes_pinv[ :, :, 2, 1]                    # (B, 32, 6, 3) bottom plane
        planes_pinv[ :, :, 3,1] = planes_pinv[ :, :, 3, 1] + dim[:, :, 0]     # (B, 32, 6, 3) top plane
        planes_pinv[ :, :, 4,1] = planes_pinv[ :, :, 4, 1] + dim[:, :, 0] / 2 # (B, 32, 6, 3) back plane
        planes_pinv[ :, :, 5,1] = planes_pinv[ :, :, 5, 1] + dim[:, :, 0] / 2 # (B, 32, 6, 3) front plane

        
        
        planes_dim_mask = planes_dim <0 # [1, 32, 6, 3] 
        planes_dim = torch.clamp(planes_dim, 0 , 10) # [1, 32, 6, 3]
        planes_dim_mask_score_mask = torch.sum(planes_dim_mask, dim=3) # (1,32, 6)
        planes_dim_mask_score_mask = 1 - (planes_dim_mask_score_mask > 0) #(1,32, 6)
        # dim_mask_score_mask = ~(dim_mask_score_mask > 0)
        planes_dim_mask_score_mask = planes_dim_mask_score_mask.float()

        planes_box_pred = torch.cat((planes_pinv, planes_dim, rot_y.unsqueeze(2).repeat(1,1,self.planes,1)), dim=3).detach() # (1,32,6,7)
       
        #planes_box_pred = torch.cat((planes_pinv, dim, rot_y), dim=2).detach() # (1,32,7)
        
        plane_loss = (planes_pinv - batch['planes_cpt']) # (1,32,6,3) # #######应该用每个面的面心 #######
        # plane_loss = (planes_pinv - batch['location']) # (1,32,3)  #######
        plane_loss_norm = torch.norm(plane_loss, p=2, dim=3) # (1,32,6)
        # plane_loss_norm = torch.norm(plane_loss, p=2, dim=2) # (1,32)
        plane_loss_mask = plane_loss_mask.float()
        ploss = plane_loss_norm * plane_loss_mask # (1, 32, 6) ######## (1, 32)       
        plane_mask_num = (ploss != 0).sum() #  mask_num =1
        ploss = ploss.sum() / (plane_mask_num + 1) # loss = 29.2169
        
        
        #==========================================================
        # p_alpha=0.1
        # total_loss  = loss + p_alpha*ploss
        #==========================================================
        '''

        dim_gt = batch['dim'].clone()  # b,c,3 (1,32,3)
        # dim_gt[:, :, 0] = torch.exp(dim_gt[:, :, 0]) * 1.63
        # dim_gt[:, :, 1] = torch.exp(dim_gt[:, :, 1]) * 1.53
        # dim_gt[:, :, 2] = torch.exp(dim_gt[:, :, 2]) * 3.88
        location_gt = batch['location'] # (1, 32, 3)    
        ori_gt = batch['ori'] # (1, 32, 1)
        dim_gt[dim_mask] = 0  # (1, 32, 3)
    
        gt_box = torch.cat((location_gt, dim_gt, ori_gt), dim=2) # (1, 32, 7)
        box_pred = box_pred.view(b * c, -1)  #  (1,32,7) - > (32, 7)
        gt_box = gt_box.view(b * c, -1)      #  (1,32,7) - > (32, 7)
        box_score = boxes_iou3d_gpu(box_pred, gt_box) # (B*32,B*32)
        box_score = torch.diag(box_score).view(b, c) # torch.Size([B, 32])
        
        prob = probability.squeeze(2) # (1 ,32, 1) -> (1,32)
        
        box_score = box_score * loss_mask * dim_mask_score_mask # (B,32)    
        loss_prob = F.binary_cross_entropy_with_logits(prob, box_score.detach(), reduce=False) # (4,32)
        loss_prob = loss_prob * loss_mask * dim_mask_score_mask # (4,32)
        loss_prob = torch.sum(loss_prob, dim=1) # (4)
        loss_prob = loss_prob.sum() / (mask_num + 1)
        box_score = box_score * loss_mask
        box_score = box_score.sum() / (mask_num + 1)
        
        '''
        # 3D planes iou
        gt_l = batch['dim'][:, :, 2:3] # (1, 32, 1)
        gt_h = batch['dim'][:, :, 0:1] # (1, 32, 1)
        gt_w = batch['dim'][:, :, 1:2] # (1, 32, 1)
  
        zero = torch.zeros_like(gt_l)
        gt_lh = torch.cat((gt_h,zero,gt_l),dim=2).unsqueeze(2)
        gt_hw = torch.cat((gt_h,gt_w,zero),dim=2).unsqueeze(2)
        gt_wl = torch.cat((zero,gt_w,gt_l),dim=2).unsqueeze(2)
        planes_dim_gt = torch.cat([gt_hw,gt_hw,gt_wl,gt_wl,gt_lh,gt_lh],dim=2) # [1, 32, 6, 3]
        planes_location_gt = batch['planes_cpt']  # [1, 32, 6, 3]
        planes_dim_gt[planes_dim_mask] = 0  # (1, 32, 6, 3)
        
        planes_gt_box = torch.cat((planes_location_gt, planes_dim_gt, ori_gt.unsqueeze(2).repeat(1,1,self.planes,1)), dim=3) # (1, 32, 7)
        planes_box_pred = planes_box_pred.view(b * c, self.planes, -1)  #  (1,32,6, 7) - > (32, 6, 7)
        planes_gt_box = planes_gt_box.view(b * c, self.planes, -1)      #  (1,32,7) - > (32, 6, 7)
        
        
        right_plane_box_score  = boxes_iou2d(planes_box_pred[:,:,0:3][:,0,0], planes_box_pred[:,:,0:3][:,0,1], planes_box_pred[:,:,3:6][:,0,0], planes_box_pred[:,:,3:6][:,0,1], \
                                               planes_gt_box[:,:,0:3][:,0,0],   planes_gt_box[:,:,0:3][:,0,1],   planes_gt_box[:,:,3:6][:,0,0],   planes_gt_box[:,:,3:6][:,0,1])
        left_plane_box_score   = boxes_iou2d(planes_box_pred[:,:,0:3][:,1,0], planes_box_pred[:,:,0:3][:,1,1], planes_box_pred[:,:,3:6][:,1,0], planes_box_pred[:,:,3:6][:,1,1], \
                                               planes_gt_box[:,:,0:3][:,1,0],   planes_gt_box[:,:,0:3][:,1,1],   planes_gt_box[:,:,3:6][:,1,0],   planes_gt_box[:,:,3:6][:,1,1])
        bottom_plane_box_score = boxes_iou2d(planes_box_pred[:,:,0:3][:,2,1], planes_box_pred[:,:,0:3][:,2,2], planes_box_pred[:,:,3:6][:,2,1], planes_box_pred[:,:,3:6][:,2,2], \
                                               planes_gt_box[:,:,0:3][:,2,1],   planes_gt_box[:,:,0:3][:,2,2],   planes_gt_box[:,:,3:6][:,2,1],   planes_gt_box[:,:,3:6][:,2,2])
        top_plane_box_score    = boxes_iou2d(planes_box_pred[:,:,0:3][:,3,1], planes_box_pred[:,:,0:3][:,3,2], planes_box_pred[:,:,3:6][:,3,1], planes_box_pred[:,:,3:6][:,3,2], \
                                               planes_gt_box[:,:,0:3][:,3,1],   planes_gt_box[:,:,0:3][:,3,2],   planes_gt_box[:,:,3:6][:,3,1],   planes_gt_box[:,:,3:6][:,3,2])
        front_plane_box_score  = boxes_iou2d(planes_box_pred[:,:,0:3][:,4,0], planes_box_pred[:,:,0:3][:,4,2], planes_box_pred[:,:,3:6][:,4,0], planes_box_pred[:,:,3:6][:,4,2], \
                                               planes_gt_box[:,:,0:3][:,4,0],   planes_gt_box[:,:,0:3][:,4,2],   planes_gt_box[:,:,3:6][:,4,0],   planes_gt_box[:,:,3:6][:,4,2])
        back_plane_box_score   = boxes_iou2d(planes_box_pred[:,:,0:3][:,5,0], planes_box_pred[:,:,0:3][:,5,2], planes_box_pred[:,:,3:6][:,5,0], planes_box_pred[:,:,3:6][:,5,2], \
                                               planes_gt_box[:,:,0:3][:,5,0],   planes_gt_box[:,:,0:3][:,5,2],   planes_gt_box[:,:,3:6][:,5,0],   planes_gt_box[:,:,3:6][:,5,2])
        
        right_plane_box_score = right_plane_box_score.view(b,c,1) # torch.Size([B, 32, 1])
        left_plane_box_score = left_plane_box_score.view(b,c,1) # torch.Size([B, 32, 1])
        bottom_plane_box_score = bottom_plane_box_score.view(b,c,1) # torch.Size([B, 32, 1])
        top_plane_box_score = top_plane_box_score.view(b,c,1) # torch.Size([B, 32, 1])
        front_plane_box_score = front_plane_box_score.view(b,c,1) # torch.Size([B, 32, 1])
        back_plane_box_score = back_plane_box_score.view(b,c,1) # torch.Size([B, 32, 1])
        plane_2d_box_score = torch.cat([right_plane_box_score,left_plane_box_score,bottom_plane_box_score,top_plane_box_score,front_plane_box_score,back_plane_box_score], dim=2) # torch.Size([4, 32, 6])
        
        
        plane_prob = probability.repeat(1,1,self.planes) #  torch.Size([4, 32, 6])       
        plane_2d_box_score = plane_2d_box_score * plane_loss_mask * planes_dim_mask_score_mask # (B,32,6)
        ploss_prob = F.binary_cross_entropy_with_logits(plane_prob, plane_2d_box_score.detach(), reduce=False) # (4,32,6)    
        ploss_prob = ploss_prob * plane_loss_mask * planes_dim_mask_score_mask # (4,32,6)
        ploss_prob = torch.sum(ploss_prob, dim=[1,2]) # 4
        ploss_prob = ploss_prob.sum() / (plane_mask_num + 1) # 4
        plane_2d_box_score = plane_2d_box_score * plane_loss_mask # 4
        plane_2d_box_score = plane_2d_box_score.sum() / (plane_mask_num + 1) # 4
        
        '''

        #==========================================================
        # p_beta=0.1
        # total_loss_prob = loss_prob + p_beta*ploss_prob
        #==========================================================
        
        #==========================================================
        # p_gama=0.1
        # total_box_score = box_score + p_gama*plane_2d_box_score
        #==========================================================
         
        return loss, plane_loss, loss_prob, box_score # ploss, ploss_prob, plane_2d_box_score, vertical_loss, horizon_loss#horizon_cos_theata_loss


class kp_align(nn.Module):
    def __init__(self):
        super(kp_align, self).__init__()

        self.index_x=torch.LongTensor([0,2,4,6,8,10,12,14])
    def forward(self, output, batch):

        kps = _transpose_and_gather_feat(output['hps'], batch['ind'])
        mask = batch['inv_mask']
        index=self.index_x.cuda()
        x_bottom=torch.index_select(kps,dim=2,index=index[0:4])
        bottom_mask = torch.index_select(mask,dim=2,index=index[0:4]).float()
        x_up=torch.index_select(kps,dim=2,index=index[4:8])
        up_mask = torch.index_select(mask, dim=2, index=index[4:8]).float()
        mask=bottom_mask*up_mask
        #loss = F.l1_loss(x_up * mask, x_bottom * mask, size_average=False)
        loss = F.l1_loss(x_up * mask, x_bottom * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)

        return loss
class kp_conv(nn.Module):
    def __init__(self):
        super(kp_conv, self).__init__()
        self.con1=torch.nn.Conv2d(18,18,3,padding=1)
        # self.con2 = torch.nn.Conv1d(32, 32, 3, padding=1)
        # self.con3 = torch.nn.Conv1d(32, 32, 3, padding=1)
        self.index_x=torch.LongTensor([0,2,4,6,8,10,12,14])
    def forward(self, output):
        kps = output['hps']
        kps=self.con1(kps)
        return kps

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8) # (B,32,8)->(32,8)
    target_bin = target_bin.view(-1, 2) #(B,32,2) -> (32, 2)
    target_res = target_res.view(-1, 2) #(B,32,2) -> (32, 2)
    mask = mask.view(-1, 1)             #(B,32) ->(32, 1) 
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask) # 计算bin1 的loss
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask) # 计算bin2 的loss
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
