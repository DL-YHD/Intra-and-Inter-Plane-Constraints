from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform, n_points_affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

# display image
def display(image):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    ax.imshow(image)
    
def display_2(image):
    cv2.namedWindow("image")
    cv2.imshow('image', image)
    cv2.waitKey (100000) # 显示 10000 ms 即 10s 后消失
    cv2.destroyAllWindows()
    
# convert BGR image to RGB image
def rgb(image):
    image = cv2.cvtcolor(image,cv2.COLOR_BGR2RGB)
    return image

class CarPoseDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i
    def read_calib(self,calib_path):
        f = open(calib_path, 'r')
        for i, line in enumerate(f):
            if i == 2:
                calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
                calib = calib.reshape(3, 4)
                return calib
    def __getitem__(self, index):
        img_id = self.images[index] # img_id = 0
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name'] # file_name = '000000.png'
        img_path = os.path.join(self.img_dir, file_name) # './RTM3D/kitti_format/data/kitti/image/000000.png'
        ann_ids = self.coco.getAnnIds(imgIds=[img_id]) # ann_ids = [1] 每张图片中检测目标的序号，序号依次递增。
        anns = self.coco.loadAnns(ids=ann_ids)
                                            # [{'segmentation': [[0, 0, 0, 0, 0, 0]],
                                            #   'num_keypoints': 9.0,
                                            #   'area': 1,
                                            #   'iscrowd': 0,
                                            #   'keypoints': [808.686767578125,
                                            #    300.5345458984375,
                                            #    2.0,
                                            #    820.2930908203125,
                                            #    307.5869140625,
                                            #    2.0,
                                            #    716.2700805664062,
                                            #    307.4004821777344,
                                            #    2.0,
                                            #    710.4447021484375,
                                            #    300.3682556152344,
                                            #    2.0,
                                            #    808.686767578125,
                                            #    146.02789306640625,
                                            #    2.0,
                                            #    820.2930908203125,
                                            #    144.0020751953125,
                                            #    2.0,
                                            #    716.2700805664062,
                                            #    144.0556182861328,
                                            #    2.0,
                                            #    710.4447021484375,
                                            #    146.07566833496094,
                                            #    2.0,
                                            #    763.7633056640625,
                                            #    224.4706268310547,
                                            #    2.0],
                                            #   'image_id': 0,
                                            #   'bbox': [712.4, 143.0, 98.33000000000004, 164.92000000000002],
                                            #   'category_id': 2,
                                            #   'id': 1,
                                            #   'dim': [1.89, 0.48, 1.2],
                                            #   'rotation_y': 0.01,
                                            #   'alpha': -0.21211633507492916,
                                            #   'location': [1.9047172778844834, 1.47, 8.41],
                                            #   'calib': [707.04931640625,
                                            #    0.0,
                                            #    604.0814208984375,
                                            #    45.75830841064453,
                                            #    0.0,
                                            #    707.04931640625,
                                            #    180.50660705566406,
                                            #    -0.34541571140289307,
                                            #    0.0,
                                            #    0.0,
                                            #    1.0,
                                            #    0.004981015808880329]}]
        num_objs = min(len(anns), self.max_objs) # 识别目标物体个数 1。 
        label_sel = np.array([1.], dtype=np.float32) # array([1.], dtype=float32)
        name_in = int(file_name[:6]) # name_in = 0
        if name_in > 14961 and name_in < 22480: 
            label_sel[0] = 0.
        # img = cv2.imdecode(np.fromfile(img_path,dtype=np.uint8),cv2.IMREAD_COLOR) 中文读取方式
        img = cv2.imread(img_path) # 读取图片 # shape = (370,1224,3)
        height, width = img.shape[0], img.shape[1] # height=370， width=1224
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32) # c = np.array([612., 185.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0 # s = 1224.0
        rot = 0
        flipped = False
        if self.split == 'train' :
            if not self.opt.not_rand_crop: #随机裁剪flag==False
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1)) # array([0.6, 0.7, 0.8, 0.9, 1. , 1.1, 1.2, 1.3]) 随机抽取一个 s=1346.3999999999999
                w_border = self._get_border(128, img.shape[1]) # w_boarder = 128
                h_border = self._get_border(128, img.shape[0]) # h_border = 128
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border) # c[0] = 344
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border) # c[1] = 220
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            # if np.random.random() < self.opt.aug_rot:
            #     rf = self.opt.rotate
            #     rot = np.clip(np.random.randn() * rf, -rf * 2, rf * 2)
            #
            # if np.random.random() < self.opt.flip:
            #     flipped = True
            #     img = img[:, ::-1, :]
            #     c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, rot, [self.opt.input_w, self.opt.input_h]) # array([[ 8.71459666e-01, -0.00000000e+00, -2.58474916e+02],
                                                             #        [ 2.06753463e-17,  8.71459666e-01,  5.69237518e+01]])
        inp = cv2.warpAffine(img, trans_input,
                             (self.opt.input_w, self.opt.input_h),
                             #(self.opt.input_res, self.opt.input_res),
                             flags=cv2.INTER_LINEAR)  # 仿射变换  inp = (384, 1280, 3)
        inp = (inp.astype(np.float32) / 255.) # inp 归一化
        if self.split == 'train' and not self.opt.no_color_aug: # 颜色增强
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)

        inp = (inp - self.mean) / self.std # z_score标准化
        inp = inp.transpose(2, 0, 1) # inp = (3,384,1280) 交换通道
        
        num_joints = self.num_joints
        n_num_joints = self.n_num_joints
        planes = self.planes
        # 仿射变换
        trans_output = get_affine_transform(c, s, 0, [self.opt.output_w, self.opt.output_h]) # np.array([[ 2.61437908e-01, -0.00000000e+00,  0.00000000e+00],
                                                                                             #           [-2.32203508e-17,  2.61437908e-01, -3.66013072e-01]])
        trans_output_inv = get_affine_transform(c, s, 0, [self.opt.output_w, self.opt.output_h], inv=1) # np.array([[ 3.825, -0.   ,  0.   ],
                                                                                                        #           [ 0.   ,  3.825,  1.4  ]])
         
        hm = np.zeros((self.num_classes, self.opt.output_h, self.opt.output_w), dtype=np.float32) #三个类别行人，车，自行车 (3,96,320) 下采样了4倍
        hm_hp = np.zeros((num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32) # (9,96,320) 下采样了4倍
        dense_kps = np.zeros((num_joints, 2, self.opt.output_h, self.opt.output_w), dtype=np.float32) # (9, 2, 96, 320) 下采样了4倍
        dense_kps_mask = np.zeros((num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32) # (9, 96, 320) 下采样了4倍
        
        ###############################################################################
        n_points_hm_hp = np.zeros((6, n_num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32) # (6, 4, 96,320) 下采样了4倍
        n_points_dense_kps = np.zeros((6, n_num_joints, 2, self.opt.output_h, self.opt.output_w), dtype=np.float32) # (6, 4, 2, 96, 320) 每个面有4个点，每个点（x,y）
        n_points_dense_kps_mask = np.zeros((6, n_num_joints, self.opt.output_h, self.opt.output_w), dtype=np.float32) # (6, 4, 96, 320) 每个面4个点。
        ##############################################################################
        
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)  # (32,2) 2D size (w,h)
        dim = np.zeros((self.max_objs, 3), dtype=np.float32) # (32,3) 3D dimension (l,w,h)
        location = np.zeros((self.max_objs, 3), dtype=np.float32) # (32,3) (x,y,z)
        dep = np.zeros((self.max_objs, 1), dtype=np.float32) # (32,1) depth 
        ori = np.zeros((self.max_objs, 1), dtype=np.float32) # (32,1) oritation
        rotbin = np.zeros((self.max_objs, 2), dtype=np.int64) # (32,2) oritation bins 2个区间
        rotres = np.zeros((self.max_objs, 2), dtype=np.float32) # (32,2) oritation回归 2个区间
        rot_mask = np.zeros((self.max_objs), dtype=np.uint8) # rotation mask
        
        kps = np.zeros((self.max_objs, num_joints * 2), dtype=np.float32) #  9个关键点*2 (x,y) (32,18)
        
        #############################################################################
        planes_n_kps= np.zeros((self.max_objs, 6, n_num_joints * 2), dtype=np.float32) # 4个关键点*2 (x,y) (32,6,8)
        planes_n_kps_3d_coef= np.zeros((self.max_objs, 6, n_num_joints, 3), dtype=np.float32) # 4个关键点*2 (x,y) (32,6,4,3)
        planes_cpt= np.zeros((self.max_objs, 6, 3), dtype=np.float32) # 4个关键点*2 (x,y) (32,6,3)
        planes_cpt_coef= np.zeros((self.max_objs, 6, 3), dtype=np.float32) # 4个关键点*2 (x,y) (32,6,3)
        #############################################################################
        kps_cent = np.zeros((self.max_objs, 2), dtype=np.float32) # (32,2) 1个中心点(x,y)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32) #  (32,2)
        ind = np.zeros((self.max_objs), dtype=np.int64) # (32,) 
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8) #  (32,) 
        
        inv_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)   # (32,18)
        kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8)  # (32,18)
        coor_kps_mask = np.zeros((self.max_objs, self.num_joints * 2), dtype=np.uint8) #  # (32,18)
        hp_offset = np.zeros((self.max_objs * num_joints, 2), dtype=np.float32) # (288,2) 32*9
        hp_ind = np.zeros((self.max_objs * num_joints), dtype=np.int64)         # (288,)
        hp_mask = np.zeros((self.max_objs * num_joints), dtype=np.int64)        # (288,)
        
        #############################################################################
        planes_n_inv_mask = np.zeros((self.max_objs, 6, self.n_num_joints * 2), dtype=np.uint8)   # (32,6,8)
        planes_n_mask = np.zeros((self.max_objs, 6, self.n_num_joints * 2), dtype=np.uint8)  # (32,6,8)
        coor_planes_n_mask = np.zeros((self.max_objs, 6, self.n_num_joints * 2), dtype=np.uint8) #  (32,6,8)
        
        #planes_n_hp_offset = np.zeros((self.max_objs * 6 * n_num_joints, 2), dtype=np.float32) # (768,2) 32*6*4
        planes_n_hp_offset = np.zeros((self.max_objs , 6 , n_num_joints, 2), dtype=np.float32) # (32,6,4,2) 32*6*4
        
        # planes_n_hp_ind  = np.zeros((self.max_objs * 6 * n_num_joints), dtype=np.int64)         # (768,)
        planes_n_hp_ind  = np.zeros((self.max_objs, 6, n_num_joints), dtype=np.int64)          # ()
      
        
        #planes_n_hp_mask = np.zeros((self.max_objs * 6 * n_num_joints), dtype=np.int64)         # (768,)
        planes_n_hp_mask = np.zeros((self.max_objs, 6 , n_num_joints), dtype=np.int64)
        #############################################################################
        rot_scalar = np.zeros((self.max_objs, 1), dtype=np.float32)             # (32,1)
        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian
        calib=np.array(anns[0]['calib'],dtype=np.float32)
        calib=np.reshape(calib,(3,4))

        gt_det = []
        for k in range(num_objs):
            ann = anns[k] # 获得一张图像上的目标
            bbox = self._coco_box_to_bbox(ann['bbox']) # 转换成普通2D bbox
            cls_id = int(ann['category_id']) - 1 # 类别id
            pts = np.array(ann['keypoints'][:27], np.float32).reshape(num_joints, 3) # 获得9个关键点的坐标
            ##############################################################################
            n_pts = np.array(ann['six_planes'][:6], np.float32).reshape(6, n_num_joints, 3) # 获得6个平面上每个面的所有点。
            n_pts_coefficient = np.array(ann['six_planes_cof_list'][:6], np.float32).reshape(6, n_num_joints, 3)
            planes_n_kps_3d_coef[k,:,:,:] = n_pts_coefficient
            
            six_plane_cpt = np.array(ann['six_plane_cpt'][:6], np.float32).reshape(6,3)
            planes_cpt[k,:,:] = six_plane_cpt
            six_plane_cpt_cof = np.array(ann['six_plane_cpt_cof'][:6], np.float32).reshape(6,3)
            planes_cpt_coef[k,:,:] = six_plane_cpt_cof
            ##############################################################################
            alpha1=ann['alpha'] # 获得计算后的alpha角
            orien=ann['rotation_y'] # 获得旋转角
            loc = ann['location'] # 获得坐标点
            if flipped:
                alpha1=np.sign(alpha1)*np.pi-alpha1
                orien = np.sign(orien) * np.pi - orien
                loc[0]=-loc[0]
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                pts[:, 0] = width - pts[:, 0] - 1
                # n_pts[:,:,0:1] = width - n_pts[:,:,0:1] - 1
                for e in self.flip_idx: # 
                    pts[e[0]], pts[e[1]] = pts[e[1]].copy(), pts[e[0]].copy()
                    #n_pts[:,e[0],:], n_pts[:,e[1],:] = n_pts[:,e[1],:].copy(), n_pts[:,e[0],:].copy()
            
            # 仿射变换后重新确定缩框后的坐标，并和输出图大小判断截断范围。
            bbox[:2] = affine_transform(bbox[:2], trans_output)            # array([90.58824, 45.38562], dtype=float32)
            bbox[2:] = affine_transform(bbox[2:], trans_output)            # array([112.01089,  81.3159 ], dtype=float32)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1) # array([ 90.58824, 112.01089], dtype=float32)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1) # array([45.38562, 81.3159 ], dtype=float32)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0] # 缩框后的宽和高
            if (h > 0 and w > 0) or (rot != 0): # 若仿射变换后的框超出边界，则不计算。
                alpha = self._convert_alpha(alpha1) # 计算旋转角
                if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                    rotbin[k, 0] = 1 # 每个位置 区间置为1
                    rotres[k, 0] = alpha - (-0.5 * np.pi) # 第一个位置的角度 
                if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                    rotbin[k, 1] = 1  # 第一个位置区间置为1
                    rotres[k, 1] = alpha - (0.5 * np.pi)  # 第一个位置的角度 
                rot_scalar[k]=alpha # 每一个物体确定alpha标量
                
                radius = gaussian_radius((math.ceil(h), math.ceil(w))) # radius = 7.500311849300935
                radius = self.opt.hm_gauss if self.opt.mse_loss else max(0, int(radius)) # radius = 7
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32) # 缩框后的bbox的中心点 array([101.29956,  63.35076], dtype=float32)
                ct_int = ct.astype(np.int32)  # array([101,  63])
                wh[k] = 1. * w, 1. * h # array([21.422646, 35.930283], dtype=float32) 物体的宽和长。
                ind[k] = ct_int[1] * self.opt.output_w + ct_int[0] # ind[k] = 20261 物体中心点的像素位置
                
                reg[k] = ct - ct_int # 中心点的偏差 array([0.29956055, 0.3507614 ], dtype=float32)
                dim[k] = ann['dim']  # dim[1] = array([1.89, 0.48, 1.2 ], dtype=float32) 三维尺寸
                # dim[k][0]=math.log(dim[k][0]/1.63)
                # dim[k][1] = math.log(dim[k][1]/1.53)
                # dim[k][2] = math.log(dim[k][2]/3.88)
                dep[k] = loc[2]   # depth = 8.41 深度（距离）每个物体都有个深度值
                ori[k] = orien    # array([0.01], dtype=float32)每个物体都有个方向值
                location[k] = loc # array([1.9047173, 1.47     , 8.41     ], dtype=float32) 每个物体都有个空间位置
                reg_mask[k] = 1   # 物体mask。 有物体，mask置为1
                num_kpts = pts[:, 2].sum() # 第三维是坐标点的个数，num_kpts = 18个点(9 * 2) 
                # num_n_points= n_pts[:,:, 2].sum(axis=1) # array([8., 8., 8., 8., 8., 8.], dtype=float32)
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999 # 
                    reg_mask[k] = 0
                    
                rot_mask[k] = 1 # 有物体的地方rotmask 置为1 
                hp_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                hp_radius = self.opt.hm_gauss \
                    if self.opt.mse_loss else max(0, int(hp_radius)) # hp_radius = 7
                kps_cent[k,:]=pts[8,:2] # 每个物体的中心点坐标
                
                
                # planes data process version v1.
                '''
                for i in range(self.planes): # 6个平面
                    n_pts[i][:,:2] = n_points_affine_transform(n_pts[i], trans_output)
                    planes_n_kps[k, i, : ] = (n_pts[i][:, :2] - ct_int).flatten() # 每个投影点x,y距离中心点x,y的距离
                    planes_n_mask[k,i, : ] = 1 # 存在投影点距离，mask就置为1 
                    
                    x_mask = np.where((n_pts[i,:,0:1].squeeze() >= 0) & (n_pts[i,:,0:1].squeeze()  < self.opt.output_w) , 1, 0)
                    y_mask = np.where((n_pts[i,:,1:2].squeeze() >= 0) & (n_pts[i,:,1:2].squeeze()  < self.opt.output_h) , 1, 0)
                    planes_n_inv_mask[k, i, : ] = np.vstack((x_mask,y_mask)).flatten('F') 
                    coor_planes_n_mask[k, i, : ] = np.vstack((x_mask,y_mask)).flatten('F')
                    planes_n_pt_int = n_pts[i][:, :2].astype(np.int32) # 将点转成整数型
                    planes_n_hp_offset[k, i, :] = n_pts[i][:, :2] - planes_n_pt_int
                    
                    planes_n_hp_ind[k, i, :] = (n_pts[i][:, 1:2]  * np.expand_dims(y_mask, axis=1) * self.opt.output_w + n_pts[i][:, 0:1] * np.expand_dims(x_mask, axis=1)).squeeze()                    
                    planes_n_hp_mask[k, i, :]  = np.where(planes_n_hp_ind[k, i, :]> 0, 1, 0)
                    
                    if self.opt.planes_n_dense_hp:
                        # must be before draw center hm gaussian
                        for n in range(self.n_num_joints):
                            draw_dense_reg(n_points_dense_kps[i][n], hm[cls_id], ct_int, n_pts[i][n, :2] - planes_n_pt_int[n] , radius, is_offset=True)
                            draw_gaussian(n_points_dense_kps_mask[i][n], ct_int, radius)
                    for n in range(self.n_num_joints):
                        draw_gaussian(n_points_hm_hp[i][n], planes_n_pt_int[n], hp_radius)
                '''
                
                
                '''
                # v2 点设计与原论文一样
                for i in range(self.planes): # 6个平面
                    n_pts[i][:,:2] = n_points_affine_transform(n_pts[i], trans_output)
                    planes_n_kps[k, i, : ] = (n_pts[i][:, :2] - ct_int).flatten() # 每个投影点x,y距离中心点x,y的距离
                    
                    planes_n_mask[k,i, : ] = 1 # 存在投影点距离，mask就置为1 
                    
                    x_mask = np.where((n_pts[i,:,0:1].squeeze() >= 0) & (n_pts[i,:,0:1].squeeze()  < self.opt.output_w) , 1, 0)
                    y_mask = np.where((n_pts[i,:,1:2].squeeze() >= 0) & (n_pts[i,:,1:2].squeeze()  < self.opt.output_h) , 1, 0)
                    planes_n_inv_mask[k, i, : ] = np.vstack((x_mask,y_mask)).flatten('F') 
                    coor_planes_n_mask[k, i, : ] = np.vstack((x_mask,y_mask)).flatten('F')
                    planes_n_pt_int = n_pts[i][:, :2].astype(np.int32) # 将点转成整数型
                    
                    planes_n_hp_offset[k * 6 * n_num_joints+i*n_num_joints:k * 6 * n_num_joints+(i+1)*n_num_joints,:] = n_pts[i][:, :2] - planes_n_pt_int 
                    planes_n_hp_ind[k * 6 * n_num_joints+i*n_num_joints:k * 6 * n_num_joints+(i+1)*n_num_joints] = (n_pts[i][:, 1:2]  * self.opt.output_w + n_pts[i][:, 0:1]).squeeze() 
                    planes_n_hp_mask[k * 6 * n_num_joints+i*n_num_joints:k * 6 * n_num_joints+(i+1)*n_num_joints]  = np.where(planes_n_hp_ind[k * 6 * n_num_joints+i*n_num_joints:k * 6 * n_num_joints+(i+1)*n_num_joints]> 0, 1, 0)
                    if self.opt.planes_n_dense_hp:
                        # must be before draw center hm gaussian
                        for n in range(self.n_num_joints):
                            draw_dense_reg(n_points_dense_kps[i][n], hm[cls_id], ct_int, n_pts[i][n, :2] - planes_n_pt_int[n] , radius, is_offset=True)
                            draw_gaussian(n_points_dense_kps_mask[i][n], ct_int, radius)
                    for n in range(self.n_num_joints):
                        draw_gaussian(n_points_hm_hp[i][n], planes_n_pt_int[n], hp_radius)
                '''
                
               
                # v3
                for i in range(self.planes): # 6个平面
                    for j in range(self.n_num_joints):
                        n_pts[i][j,:2] = affine_transform(n_pts[i][j, :2], trans_output)
                        planes_n_kps[k, i, j * 2: j * 2 + 2] = (n_pts[i][j, :2] - ct_int) # 每个投影点x,y距离中心点x,y的距离, Note: 是否可以使用每个平面的中心点！！！
                        planes_n_mask[k, i, j * 2: j * 2 + 2] = 1 # 存在投影点距离，mask就置为1 
                        if n_pts[i][j, 2] > 0:
                            if n_pts[i][j, 0] >= 0 and n_pts[i][j, 0] < self.opt.output_w and \
                                    n_pts[i][j, 1] >= 0 and n_pts[i][j, 1] < self.opt.output_h: # 判断投影点是否在feature_map范围之内。
                        
                                    planes_n_inv_mask[k, i, j * 2: j * 2 + 2] = 1
                                    coor_planes_n_mask[k, i, j * 2: j * 2 + 2 ] = 1
                                    
                                    planes_n_pt_int = n_pts[i][j, :2].astype(np.int32) # 将点转成整数型
                                    planes_n_hp_offset[k, i, j, :] = n_pts[i][j, :2] - planes_n_pt_int
                                    
                                    planes_n_hp_ind[k, i, j] = (planes_n_pt_int[1]  * self.opt.output_w + planes_n_pt_int[0])                    
                                    planes_n_hp_mask[k, i, j]  = 1
                        
                                    if self.opt.planes_n_dense_hp:                             
                                        draw_dense_reg(n_points_dense_kps[i][j], hm[cls_id], ct_int, n_pts[i][j, :2] - planes_n_pt_int[j] , radius, is_offset=True)
                                        draw_gaussian(n_points_dense_kps_mask[i][j], ct_int, radius)
                                    
                                    draw_gaussian(n_points_hm_hp[i][j], planes_n_pt_int, hp_radius) # 每个key点画一个高斯半径
            
                
                
                for j in range(num_joints):
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output) # 每个点缩放后重新定位坐标。
                    #pts[j, :2] = affine_transform(pts[j, :2], trans_output_inv) # 变回原来的坐标。
                    kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int # 每个投影点x,y距离中心点x,y的距离
                    kps_mask[k, j * 2: j * 2 + 2] = 1 # 存在投影点距离，mask就置为1  
                    if pts[j, 2] > 0:
                        #pts[j, :2] = affine_transform(pts[j, :2], trans_output)
                        if pts[j, 0] >= 0 and pts[j, 0] < self.opt.output_w and \
                                pts[j, 1] >= 0 and pts[j, 1] < self.opt.output_h: # 判断投影点是否在feature_map范围之内。
                            #kps[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                            #kps_mask[k, j * 2: j * 2 + 2] = 1
                            inv_mask[k, j * 2: j * 2 + 2] = 1 # inv_mask 有置为1 
                            coor_kps_mask[k, j * 2: j * 2 + 2] = 1  # 坐标据中心点距离mask置为1 
                            pt_int = pts[j, :2].astype(np.int32) # 将点转成整数型
                            hp_offset[k * num_joints + j] = pts[j, :2] - pt_int # 每个物体坐标取整之后的偏差
                            hp_ind[k * num_joints + j] = pt_int[1] * self.opt.output_w + pt_int[0] # 索引25391
                            hp_mask[k * num_joints + j] = 1 # 有物体的地方的坐标置为1 
                            if self.opt.dense_hp:
                                # must be before draw center hm gaussian
                                draw_dense_reg(dense_kps[j], hm[cls_id], ct_int,
                                               pts[j, :2] - ct_int, radius, is_offset=True)
                                draw_gaussian(dense_kps_mask[j], ct_int, radius)
                            draw_gaussian(hm_hp[j], pt_int, hp_radius)
                # 体心中心点            
                if coor_kps_mask[k,16]==0 or coor_kps_mask[k,17]==0: # 最后中心点的偏差是否为0
                    coor_kps_mask[k,:] = coor_kps_mask[k,:] * 0 # 若为0 则置为0
                draw_gaussian(hm[cls_id], ct_int, radius)
                
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                               pts[:, :2].reshape(num_joints * 2).tolist() + n_pts[:,:,:2].reshape(planes*n_num_joints * 2).tolist()+ [cls_id])
                # gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                #                ct[0] + w / 2, ct[1] + h / 2, 1] +
                #               pts[:, :2].reshape(num_joints * 2).tolist() + [cls_id])
        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            kps_mask *= 0
            planes_n_mask *= 0
        meta = {'file_name': file_name}
        if flipped:
            coor_kps_mask = coor_kps_mask * 0
            coor_planes_n_mask = coor_planes_n_mask * 0 
            
            inv_mask=inv_mask*0
            planes_n_inv_mask = planes_n_inv_mask * 0
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh,
               'hps': kps, 'hps_mask': kps_mask, 
               'planes_n_kps':planes_n_kps, 'planes_n_kps_3d_coef':planes_n_kps_3d_coef,'planes_n_mask':planes_n_mask,
               'planes_cpt':planes_cpt,'planes_cpt_coef':planes_cpt_coef,
               'dim': dim,'rotbin': rotbin, 'rotres': rotres,'rot_mask': rot_mask,
               'dep':dep,'rotscalar':rot_scalar,'kps_cent':kps_cent,'calib':calib,
               'opinv':trans_output_inv,'meta':meta,"label_sel":label_sel,'location':location,'ori':ori,
               'coor_kps_mask':coor_kps_mask,'inv_mask':inv_mask, 'coor_planes_n_mask':coor_planes_n_mask, 'planes_n_inv_mask':planes_n_inv_mask}
        if self.opt.reg_offset:
            # 物体中心点的偏差
            ret.update({'reg': reg})
        if self.opt.hm_hp:
            # 9个顶点的高斯核
            ret.update({'hm_hp': hm_hp})
        if self.opt.n_points_hm_hp:
            # 6个平面每个点的高斯核
            ret.update({'n_points_hm_hp':n_points_hm_hp})
        if self.opt.reg_hp_offset:
            # hp_offset shape = (288, 2)
            # hp_ind shape = (288,)
            # hp_mask shape = (288,)
            
            ret.update({'hp_offset': hp_offset, 'hp_ind': hp_ind, 'hp_mask': hp_mask})
        if self.opt.reg_planes_hp_offset:     
            # v1 
            # planes_n_hp_mask = np.where(planes_n_hp_ind> 0, 1, 0) #根据ind确定该面的mask
            # v2
            # planes_n_hp_offset = np.mean(planes_n_hp_offset,axis=2,dtype=np.float32) #  取平均值(32, 6, 4, 2) -> (32, 6, 2)
            # planes_n_hp_ind = np.mean(planes_n_hp_ind,axis=2, dtype=np.int64) #取平面点的平均值表示该面的ind (32,6,4) ->(32,6)
            # planes_n_hp_mask = np.mean(planes_n_hp_mask,axis=2, dtype=np.int64) # (32,6,4) ->(32,6)
           
            ret.update({'planes_n_hp_offset': planes_n_hp_offset, 'planes_n_hp_ind': planes_n_hp_ind, 'planes_n_hp_mask': planes_n_hp_mask})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 40), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret
