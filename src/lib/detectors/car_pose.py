from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.decode import car_pose_decode,car_pose_decode_faster,car_pose_decode_faster_n_points
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.post_process import car_pose_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CarPoseDetector(BaseDetector):
    def __init__(self, opt):
        super(CarPoseDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx

    def process(self, images, meta, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()
            if self.opt.n_points_hm_hp and not self.opt.n_p_mse_loss:
                output['n_points_hm_hp'] = output['n_points_hm_hp'].sigmoid_()
                

            reg = output['reg'] if self.opt.reg_offset else None # None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None # None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None # None
            n_points_hm_hp = output['n_points_hm_hp'] if self.opt.n_points_hm_hp else None # None 
            planes_n_hp_offset = output['planes_n_hp_offset'] if self.opt.reg_planes_hp_offset else None # None
            torch.cuda.synchronize()
            forward_time = time.time()
            
            if self.opt.flip_test:
                output['hm'] = (output['hm'][0:1] + flip_tensor(output['hm'][1:2])) / 2
                output['wh'] = (output['wh'][0:1] + flip_tensor(output['wh'][1:2])) / 2
                output['hps'] = (output['hps'][0:1] +
                                 flip_lr_off(output['hps'][1:2], self.flip_idx)) / 2
                hm_hp = (hm_hp[0:1] + flip_lr(hm_hp[1:2], self.flip_idx)) / 2 \
                    if hm_hp is not None else None 
                hp_offset = hp_offset[0:1] if hp_offset is not None else None
                
                output['planes_n_kps'] = (output['planes_n_kps'][0:1] +
                                 flip_lr_off(output['planes_n_kps'][1:2], self.flip_idx)) / 2
                n_points_hm_hp = (n_points_hm_hp[0:1] + flip_lr(n_points_hm_hp[1:2], self.flip_idx)) / 2 \
                    if n_points_hm_hp is not None else None 
                planes_n_hp_offset = planes_n_hp_offset[0:1] if planes_n_hp_offset is not None else None
                
                reg = reg[0:1] if reg is not None else None
            if self.opt.faster==True:
                # output['hm'] = [1,3,96,320]
                # output['hps'] = [1,18,96,320]
                # output['dim'] = [1,3,96,320]
                # output['rot'] = [1,8,96,320]
                # output['prob'] = [1,1,96,320]
                # output['planes_n_kps'] = [1, 48, 96, 320]
                # 
                # {'c': array([621. , 187.5], dtype=float32), 's': 1242.0, 'out_height': 96, 'out_width': 320, 
                #  'trans_output_inv': tensor([[[ 3.8813e+00, -0.0000e+00,  0.0000e+00],
                #                               [-3.5527e-16,  3.8813e+00,  1.2000e+00]]], device='cuda:0', dtype=torch.float64), 
                #  'calib': tensor([[[7.2154e+02, 0.0000e+00, 6.0956e+02, 4.4857e+01],
                #                    [0.0000e+00, 7.2154e+02, 1.7285e+02, 2.1638e-01],
                #                    [0.0000e+00, 0.0000e+00, 1.0000e+00, 2.7459e-03]]], device='cuda:0')}
                
                # [1,100,41]
                dets, kp_num, planes= car_pose_decode_faster(
                     output['hm'], output['hps'], output['dim'], output['rot'], prob=output['prob'], K=self.opt.K, meta=meta, const=self.const)
                
                # [1, 100, 86]
                #dets, kp_num, planes= car_pose_decode_faster_n_points(
                #    output['hm'], output['hps'], output['planes_n_kps'], output['dim'], output['rot'], prob=output['prob'], K=self.opt.K, meta=meta, planes_const=self.planes_const)
                   
            else:
                dets = car_pose_decode(
                    output['hm'], output['wh'], output['hps'],output['dim'],output['rot'],prob=output['prob'],
                    reg=reg, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K, meta=meta,const=self.const)

        if return_time:
            return output, dets, kp_num, planes, forward_time
        else:
            return output, dets, kp_num, planes

    def post_process(self, dets, kp_num, planes, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        kp_coor = planes * kp_num * 2
        dets_length = dets.shape[2]
        
        dets = car_pose_post_process(
            dets.copy(), kp_num, planes, [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'])
        for j in range(1,2):#, self.num_classes + 1):
            # print(np.array(dets[0][j], dtype=np.float32).shape)
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, dets_length) #(100,41)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:5+kp_coor] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, kp_num, planes, results, calib, ground_truth, project_points):
        debugger.add_img(image, img_id='car_pose')
        kp_coor = planes* kp_num * 2
        class_dict = {'Car':0, 'Pedestrian':1, 'Cyclist':2}
        if ground_truth != None:
            for gt in ground_truth:
                pass
                #debugger.add_coco_bbox(gt[4:8], class_dict[gt[0]], 1, img_id='car_pose')
            for gt_point in project_points:
                pass
                # debugger.add_kitti_hp(gt_point.reshape(1,-1)[0], img_id='car_pose', color_flag='gt')
                
        for bbox in results[1]:
            if bbox[4] > self.opt.vis_thresh:
                debugger.add_coco_bbox(bbox[:4], bbox[-1], bbox[4], img_id='car_pose')
                debugger.add_kitti_hp(bbox[5:5+kp_coor], img_id='car_pose')
                debugger.add_bev(bbox, kp_coor, img_id='car_pose',is_faster=self.opt.faster)
                debugger.add_3d_detection(bbox, kp_coor, calib, img_id='car_pose')
                debugger.save_kitti_format(bbox, kp_coor,self.image_path,self.opt,img_id='car_pose',is_faster=self.opt.faster)
        if self.opt.vis:
            debugger.show_all_imgs(pause=self.pause)

