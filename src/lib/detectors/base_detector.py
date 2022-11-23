from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

from models.model import create_model, load_model
from utils.image import get_affine_transform
from utils.debugger import Debugger
import os

from utils.points_function import compute_box_3d, project_to_image
class BaseDetector(object):
  def __init__(self, opt):
    if opt.gpus[0] >= 0:
      opt.device = torch.device('cuda')
    else:
      opt.device = torch.device('cpu')
    
    print('Creating model...')
    self.model = create_model(opt.arch, opt.heads, opt.head_conv)
    self.model = load_model(self.model, opt.load_model)
    self.model = self.model.to(opt.device)
    self.model.eval()

    self.mean = np.array(opt.mean, dtype=np.float32).reshape(1, 1, 3)
    self.std = np.array(opt.std, dtype=np.float32).reshape(1, 1, 3)
    self.max_per_image = 100
    self.num_classes = opt.num_classes
    self.scales = opt.test_scales
    self.opt = opt
    self.planes = 6
    self.pause = True
    self.image_path=' '
    const = torch.Tensor(
                        [[-1, 0], [0, -1], 
                         [-1, 0], [0, -1], 
                         [-1, 0], [0, -1], 
                         [-1, 0], [0, -1], 
                         [-1, 0], [0, -1], 
                         [-1, 0], [0, -1],
                         [-1, 0], [0, -1], 
                         [-1, 0], [0, -1]])
    self.const = const.unsqueeze(0).unsqueeze(0)
    self.const=self.const.to(self.opt.device)
    
    planes_const = torch.Tensor([[-1, 0], [0, -1]])
    planes_const = planes_const.repeat((self.opt.planes_n_kps,1)).unsqueeze(0).repeat((self.planes,1,1))
    self.planes_const = planes_const.unsqueeze(0).unsqueeze(0)  # [1, 1, 6, 8, 2]
    self.planes_const=self.planes_const.to(self.opt.device)

  def pre_process(self, image, scale, meta=None):
      
      height, width = image.shape[0:2]
      new_height = int(height * scale)
      new_width  = int(width * scale)
      if self.opt.fix_res:
        inp_height, inp_width = self.opt.input_h, self.opt.input_w
        c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
      else:
        inp_height = (new_height | self.opt.pad) + 1
        inp_width = (new_width | self.opt.pad) + 1
        c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
        s = np.array([inp_width, inp_height], dtype=np.float32)

      trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
      resized_image = cv2.resize(image, (new_width, new_height))
      inp_image = cv2.warpAffine(
        resized_image, trans_input, (inp_width, inp_height),
        flags=cv2.INTER_LINEAR)
      inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

      images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
      if self.opt.flip_test:
        images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
      images = torch.from_numpy(images)

      meta = {'c': c, 's': s,
              'out_height': inp_height // self.opt.down_ratio,
              'out_width': inp_width // self.opt.down_ratio}
      trans_output_inv = get_affine_transform(c, s, 0, [meta['out_width'], meta['out_height']],inv=1)
      trans_output_inv = torch.from_numpy(trans_output_inv)
      trans_output_inv=trans_output_inv.unsqueeze(0)
      meta['trans_output_inv']=trans_output_inv
      return images, meta

  def process(self, images, meta, return_time=False):
    raise NotImplementedError

  def post_process(self, dets, meta, scale=1):
    raise NotImplementedError

  def merge_outputs(self, detections):
    raise NotImplementedError

  def debug(self, debugger, images, dets, output, scale=1):
    raise NotImplementedError

  def show_results(self, debugger, image, results, calib, ground_truth=None, project_points=None):
   raise NotImplementedError

  def read_clib(self,calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
      if i == 2:
        calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
        calib = calib.reshape(3, 4)
        return calib
        
  def read_label(self, label_path, calib):
    f = open(label_path, 'r')
    gt_results = []
    gt_project_points = []
    image_shape = [370, 1224]
    for i, line in enumerate(f):
        obj_class =  line.split(' ')[0]
        if obj_class not in ['Car', 'Pedestrian', 'Cyclist']:
          continue
        dim = [float(line.split(' ')[8]), float(line.split(' ')[9]), float(line.split(' ')[10])]
        location = [float(line.split(' ')[11]), float(line.split(' ')[12]), float(line.split(' ')[13])]
        rotation_y = [float(line.split(' ')[14])]
        box_3d = compute_box_3d(dim, location, rotation_y)
        box_2d_as_point, vis_num, pts_center = project_to_image(box_3d, calib, image_shape) 
        project_points = box_2d_as_point[:,:2]
        gt_project_points.append(project_points)
        # kpts_2d = torch.tensor(box_2d_as_point[:,:2],dtype=torch.float32)
        gt_results.append([obj_class]+[float(i) for i in line.split(' ')[1:]])
    return gt_results, gt_project_points

  def run(self, image_or_path_or_tensor, meta=None):
    load_time, pre_time, net_time, dec_time, post_time = 0, 0, 0, 0, 0
    merge_time, tot_time = 0, 0
    debugger = Debugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3),
                        theme=self.opt.debugger_theme)
    start_time = time.time()
    pre_processed = False
    #print('image_or_path_or_tensor',image_or_path_or_tensor)
    if isinstance(image_or_path_or_tensor, np.ndarray):
      image = image_or_path_or_tensor
    elif type(image_or_path_or_tensor) == type (''):
      
      self.image_path=image_or_path_or_tensor    # ./kitti_format/data/kitti/training/image_2/000003.png

      image = cv2.imread(image_or_path_or_tensor)
      calib_path=os.path.join(self.opt.calib_dir, image_or_path_or_tensor[-10:-3]+'txt')
      calib_numpy=self.read_clib(calib_path)
      calib=torch.from_numpy(calib_numpy).unsqueeze(0).to(self.opt.device)

      data_tpye = self.image_path.split('/')[-3]
      if data_tpye == 'training':
        gt_label_path = self.image_path.replace('image_2','label_2').replace('.png','.txt')
        gt_results, gt_project_2d_points = self.read_label(gt_label_path, calib_numpy)
    else:
      image = image_or_path_or_tensor['image'][0].numpy()
      pre_processed_images = image_or_path_or_tensor
      pre_processed = True
    
    loaded_time = time.time()
    load_time += (loaded_time - start_time)
    
    detections = []
    for scale in self.scales:
      scale_start_time = time.time()
      if not pre_processed:

        images, meta = self.pre_process(image, scale, meta)
        meta['trans_output_inv']=meta['trans_output_inv'].to(self.opt.device)
      else:
        images = pre_processed_images['images'][scale][0]
        meta = pre_processed_images['meta'][scale]
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
      meta['calib']=calib
      images = images.to(self.opt.device)
      torch.cuda.synchronize()
      pre_process_time = time.time()
      pre_time += pre_process_time - scale_start_time
      
      output, dets, kp_num, planes, forward_time = self.process(images,meta,return_time=True)

      torch.cuda.synchronize()
      net_time += forward_time - pre_process_time
      decode_time = time.time()
      dec_time += decode_time - forward_time
      
      if self.opt.debug >= 2:
        self.debug(debugger, images, dets, output, scale)
      
      dets = self.post_process(dets, kp_num, planes, meta, scale)
      
      torch.cuda.synchronize()
      post_process_time = time.time()
      post_time += post_process_time - decode_time

      detections.append(dets)

    results = self.merge_outputs(detections)

    
    
    torch.cuda.synchronize()
    end_time = time.time()
    merge_time += end_time - post_process_time
    tot_time += end_time - start_time

    if self.opt.debug >= 1:
      self.show_results(debugger, image, kp_num, planes, results, calib_numpy, ground_truth=gt_results if data_tpye == 'training' else None, project_points = gt_project_2d_points if data_tpye == 'training' else None)
    
    return {'results': results, 'gt_results': gt_results, 'gt_project_2d_points':gt_project_2d_points, 'tot': tot_time, 'load': load_time,
            'pre': pre_time,    'net': net_time, 'dec': dec_time,
            'post': post_time, 'merge': merge_time}