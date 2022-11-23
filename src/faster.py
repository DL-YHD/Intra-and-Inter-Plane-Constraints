from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from detectors.car_pose import CarPoseDetector
from opts import opts
import shutil
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net', 'dec']

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.heads = {'hm': 3, 'wh': 2, 'hps': 18, 'planes_n_kps': 6*opt.planes_n_kps*2, 'rot': 8, 'dim': 3, 'prob': 1, 'reg': 2, 'hm_hp': 9, 'n_points_hm_hp': 6*opt.planes_n_kps, 'hp_offset': 2, 'planes_n_hp_offset': 6*opt.planes_n_kps*2}
    opt.hm_hp=False
    opt.n_points_hm_hp = False
    opt.reg_offset=False
    opt.reg_hp_offset=False
    opt.reg_planes_hp_offset = False
    opt.faster=True
    
    Detector = CarPoseDetector
    detector = Detector(opt)
    if os.path.exists(opt.results_dir):
        shutil.rmtree(opt.results_dir,True)

    if os.path.isdir(opt.demo):
      image_names = []
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
    else:
          if opt.demo[-3:] == 'txt':
              
              with open(opt.demo, 'r') as f:
                  lines = f.readlines()  
              image_names = [os.path.join(opt.data_dir + '/kitti/{0}/image_2/'.format(opt.test_data_used), img[:6] + '.png') for img in lines]
        
          else:
              image_names = [opt.demo]
    
    time_tol=0
    num=0
    for (image_name) in image_names:
      num+=1
      print(image_name)
      ret = detector.run(image_name)
     
      time_str = ''
      for stat in time_stats:
          time_tol=time_tol+ret[stat]
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
      print(time_str)
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
