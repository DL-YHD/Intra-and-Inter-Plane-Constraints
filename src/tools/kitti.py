from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import pickle
import json
import numpy as np
import cv2
nuscenes = False

parent_path = os.path.abspath(os.path.join(os.getcwd(), "../.."))

DATA_PATH = os.path.join(parent_path,'kitti_format/data/kitti/')

if nuscenes:
    DATA_PATH = os.path.join(parent_path,'kitti_format/data/nuscenes/')
DEBUG = False
# VAL_PATH = DATA_PATH + 'training/label_val/'
import os
import math
SPLITS = ['train1']
import _init_paths
from utils.ddd_utils import compute_box_3d, project_to_image, project_to_image3,alpha2rot_y
from utils.ddd_utils import draw_box_3d, unproject_2d_to_3d
from utils.ddd_utils import create_plane_points, project_to_image2
'''
#Values    Name      Description
----------------------------------------------------------------------------
   1    type         Describes the type of object: 'Car', 'Van', 'Truck',
                     'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram',
                     'Misc' or 'DontCare'
   1    truncated    Float from 0 (non-truncated) to 1 (truncated), where
                     truncated refers to the object leaving image boundaries
   1    occluded     Integer (0,1,2,3) indicating occlusion state:
                     0 = fully visible, 1 = partly occluded
                     2 = largely occluded, 3 = unknown
   1    alpha        Observation angle of object, ranging [-pi..pi]
   4    bbox         2D bounding box of object in the image (0-based index):
                     contains left, top, right, bottom pixel coordinates
   3    dimensions   3D object dimensions: height, width, length (in meters)
   3    location     3D object location x,y,z in camera coordinates (in meters)
   1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]
   1    score        Only for results: Float, indicating confidence in
                     detection, needed for p/r curves, higher is better.
'''


def _bbox_to_coco_bbox(bbox): # (point,point) ---> (point, length)
    return [(bbox[0]), (bbox[1]),
            (bbox[2] - bbox[0]), (bbox[3] - bbox[1])]


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
def read_clib3(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 3:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
def read_clib0(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 0:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib
cats = ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Person_sitting',
        'Tram', 'Misc', 'DontCare']
# det_cats=['Car', 'Pedestrian', 'Cyclist']
det_cats=['Car']
if nuscenes:
    cats = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 'pedestrian', 'motorcycle', 'bicycle',
            'traffic_cone', 'barrier']
    det_cats = cats
cat_ids = {cat: i + 1 for i, cat in enumerate(cats)} # {'Car': 1,'Pedestrian': 2, 'Cyclist': 3,'Van': 4,'Truck': 5,'Person_sitting': 6,'Tram': 7,'Misc': 8,'DontCare': 9}
# cat_info = [{"name": "pedestrian", "id": 1}, {"name": "vehicle", "id": 2}]
F = 721
H = 384  # 375
W = 1248  # 1242
EXT = [45.75, -0.34, 0.005]
CALIB = np.array([[F, 0, W / 2, EXT[0]], [0, F, H / 2, EXT[1]],
                  [0, 0, 1, EXT[2]]], dtype=np.float32)        # array([[ 7.210e+02,  0.000e+00,  6.240e+02,  4.575e+01],
                                                               # [ 0.000e+00,  7.210e+02,  1.920e+02, -3.400e-01],
                                                               # [ 0.000e+00,  0.000e+00,  1.000e+00,  5.000e-03]], dtype=float32)

res_value = 2 # resolution 取被2除尽的数且不超过2. 当resolution=2时，取的就是顶点坐标。 resolution = [ 2, 1, 0.5]

cat_info = []
points_dict = {2:4,1:9,0.5:25} # keypoints interval distance 
for i, cat in enumerate(cats):
    cat_info.append({'name': cat, 'id': i + 1})

for SPLIT in SPLITS:
    image_set_path = os.path.join(DATA_PATH,"training/image_2/")
    ann_dir = os.path.join(DATA_PATH,"training/label_2/")
    calib_dir = os.path.join(DATA_PATH,"training/calib/")
    splits = ['train_split_1','val_split_1','train_split_2','val_split_2']
    if nuscenes:
        splits = ['train_nuscenes', 'val_nuscenes']
    # splits = ['trainval', 'test']
    calib_type = {'train': 'training', 'val': 'training', 'trainval': 'training',
                  'test': 'testing', 'train_stereo': 'training'}
    for split in splits:
        ret = {'images': [], 'annotations': [], "categories": cat_info}
        image_set = open(DATA_PATH + '{}.txt'.format(split), 'r')
        image_to_id = {}
        for line in image_set:
            if line[-1] == '\n':
                line = line[:-1]
            image_id = int(line) # 获取图片id索引号
            calib_path = calib_dir  + '{}.txt'.format(line)

            calib0 = read_clib0(calib_path) # P0 : array([[707.0493,   0.    , 604.0814,   0.    ],
                                            #             [  0.    , 707.0493, 180.5066,   0.    ],
                                            #             [  0.    ,   0.    ,   1.    ,   0.    ]], dtype=float32)
            if image_id>7480 and image_id<14962: # 使用right image(kitti image 3), 
                calib = read_clib3(calib_path)
            else:
                calib = read_clib(calib_path) # calib = np.array([[ 7.070493e+02,  0.000000e+00,  6.040814e+02,  4.575831e+01],
                                              #                   [ 0.000000e+00,  7.070493e+02,  1.805066e+02, -3.454157e-01],
                                              #                   [ 0.000000e+00,  0.000000e+00,  1.000000e+00,  4.981016e-03]],
                                              #                   dtype=np.float32)

            image_info = {'file_name': '{}.png'.format(line),
                          'id': int(image_id),
                          'calib': calib.tolist()}   # {'file_name': '000000.png',
                                                     #         'id': 0,
                                                     #      'calib': [[707.04931640625, 0.0,             604.0814208984375,   45.75830841064453],
                                                     #                [0.0,             707.04931640625, 180.50660705566406,  -0.34541571140289307],
                                                     #                [0.0,             0.0,             1.0,                 0.004981015808880329]]}
            ret['images'].append(image_info)
            if split == 'test':
                continue
            ann_path = ann_dir + '{}.txt'.format(line)
            # print(ann_path)
            # if split == 'val':
            #   os.system('cp {} {}/'.format(ann_path, VAL_PATH))
            anns = open(ann_path, 'r')
            for ann_ind, txt in enumerate(anns): # ann_ind = 0, txt = 'Pedestrian 0.00 0 -0.20 712.40 143.00 810.73 307.92 1.89 0.48 1.20 1.84 1.47 8.41 0.01\n'
                tmp = txt[:-1].split(' ')        # ['Pedestrian', '0.00', '0', '-0.20', '712.40', '143.00','810.73', '307.92', '1.89', '0.48', '1.20', '1.84', '1.47', '8.41', '0.01']
                cat_id = cat_ids[tmp[0]]         # cat_id = 2  类别id
                truncated = int(float(tmp[1]))   # truncated = 0 截断类型
                occluded = int(tmp[2])           # occluded = 0 遮挡类型
                alpha = float(tmp[3])            # alpha = -0.2 
                dim = [float(tmp[8]), float(tmp[9]), float(tmp[10])] # dimension = [1.89, 0.48, 1.2]
                location = [float(tmp[11]), float(tmp[12]), float(tmp[13])] # location = [1.84, 1.47, 8.41]
                rotation_y = float(tmp[14])                                    # rotation_y = 0.01
                num_keypoints = 0
                box_2d_as_point=[0]*27 # [0,0,0, 0,0,0] 创建3*9 个初始坐标点
                bbox=[0.,0.,0.,0.]
                calib_list = np.reshape(calib, (12)).tolist() # 将 array 转成 list
                if tmp[0] in det_cats: # 判断该物体是否是需要识别的类别。
                    #image = cv2.imdecode(np.fromfile(os.path.join(image_set_path, image_info['file_name']),dtype=np.uint8),cv2.IMREAD_COLOR)
                    image = cv2.imread(os.path.join(image_set_path, image_info['file_name'])) # 读取图片 shape=[370,1224]
                    bbox = [float(tmp[4]), float(tmp[5]), float(tmp[6]), float(tmp[7])]  # bbox = [712.4, 143.0, 810.73, 307.92] 获取2D边界框
                    
                    #=============================================================================================================#
                    resolution = res_value # resolution 取被2除尽的数且不超过2. 当resolution=2时，取的就是顶点坐标。 resolution = [ 2, 1, 0.5]
                    #=============================================================================================================#
                    N = 6 * int((2/resolution + 1)**2) # 6个面总共的点数,2 是bin的长度 
                    six_planes_list = []
                    six_planes_cof_list = []
                    six_planes_cpt_list = []
                    six_planes_cpt_cof_list = []
                    
                    plane_vis_num_list = []
                    six_planes_points_3d, six_planes_points_3d_indexs, cpt, cpt_index, six_plane_cpt, six_plane_cpt_cof= create_plane_points(dim, location, rotation_y, resolution)
                    
                    for plane_cpt, cpt_cof in zip(six_plane_cpt, six_plane_cpt_cof):
                        plane_cpt = plane_cpt.tolist()[0]
                        cpt_cof = cpt_cof.tolist()[0]
                        six_planes_cpt_list.append(plane_cpt)
                        six_planes_cpt_cof_list.append(cpt_cof)
                        
                    for n_points_3d, n_points_3d_index in zip(six_planes_points_3d,six_planes_points_3d_indexs): # 6个平面和一个中心点
                          n_points_2d_as_point, n_points_vis_num, pts_center= project_to_image2(n_points_3d, calib, image.shape, cpt)
                          n_points_2d_as_point=np.reshape(n_points_2d_as_point,(1,3*(N//6)))
                          
                          n_points_3d_index=np.reshape(n_points_3d_index,(1,3*(N//6)))
                          n_points_2d_as_point=n_points_2d_as_point.tolist()[0] # array转换成list
                          n_points_3d_index=n_points_3d_index.tolist()[0] # array转换成list
                          six_planes_list.append(n_points_2d_as_point)
                          six_planes_cof_list.append(n_points_3d_index)
                          plane_vis_num_list.append(n_points_vis_num)
                         
            
                    box_3d = compute_box_3d(dim, location, rotation_y) # 将世界坐标系中的3D坐标转换成相机坐标系
                 
                    box_2d_as_point,vis_num,pts_center = project_to_image(box_3d, calib, image.shape) # 获得9个2D 投影的关键点坐标， 可见关键点的个数，中心坐标点的投影
                    box_2d_as_point=np.reshape(box_2d_as_point,(1,27)) # reshape成1*27
                    ###box_2d_as_point=box_2d_as_point.astype(np.int)###
                    box_2d_as_point=box_2d_as_point.tolist()[0] # array转换成list
                    num_keypoints=vis_num
                    
                    
                    off_set=(calib[0,3]-calib0[0,3])/calib[0,0] # off_set = (45.75831 - 0) / 707.0493 = 0.06471728
                    location[0] += off_set###################################################confuse
                    alpha = rotation_y - math.atan2(pts_center[0, 0] - calib[0, 2], calib[0, 0]) # alpha = 0.01 - arctan((763.7633-604.0814)/707.0493) = -0.21211633507492916
                    ann = {'segmentation': [[0,0,0,0,0,0]],
                           'num_keypoints':num_keypoints,
                           'plane_num_keypoints':plane_vis_num_list,
                           'area':1,
                           'iscrowd': 0,
                           'keypoints': box_2d_as_point,
                           'six_planes':six_planes_list,
                           'six_planes_cof_list':six_planes_cof_list,
                           'six_plane_cpt':six_planes_cpt_list,
                           'six_plane_cpt_cof':six_planes_cpt_cof_list,
                           'image_id': image_id,
                           'bbox': _bbox_to_coco_bbox(bbox),
                           'category_id': cat_id,
                           'id': int(len(ret['annotations']) + 1),
                           'dim': dim,
                           'rotation_y': rotation_y,
                           'alpha': alpha,
                           'location':location,
                           'calib':calib_list,
                            }
                    ret['annotations'].append(ann)
        print("# images: ", len(ret['images']))
        print("# annotations: ", len(ret['annotations']))
        # import pdb; pdb.set_trace()
        out_path = '{}annotations_split_{}/kitti_{}_{}_points.json'.format(DATA_PATH, split.split('_')[2], split.split('_')[0], points_dict[resolution])
        json.dump(ret, open(out_path, 'w'))

