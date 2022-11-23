from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2



def create_plane_points(dimension, location, rotation_y, resolution):
    # resolution 间隔dim*resolution比例,采样一次。
    
    #lr_index = [0, 1, 2, 3, 4, 5, 6, 7]
    #bt_index = [0, 2, 4, 6, 1, 3, 5, 7]
    #fb_index = [0, 4, 1, 5, 2, 6, 3, 7]
    
    half_l = dimension[2] / 2
    half_h = dimension[0] / 2
    half_w = dimension[1] / 2
    
    # 创建3D Box 8 个顶点
    # l, w, h = dimension[2], dimension[1], dimension[0]
    # X_corners = [l/2,  l/2, -l/2, -l/2, l/2,  l/2, -l/2, -l/2,   0] # 最后增加了一个中心点坐标 x.
    # Y_corners = [  0,    0,    0,    0,  -h,   -h,   -h,   -h,   -h/2] # 最后增加了一个中心点坐标 y.
    # Z_corners = [w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2,   0] # 最后增加了一个中心点坐标 z.
    
    
    # X_corners_ind = [1/2,  1/2, -1/2, -1/2,  1/2,  1/2, -1/2, -1/2,   0] # 旋转平移后的 x索引.
    # Y_corners_ind = [1/2,  1/2,  1/2,  1/2, -1/2, -1/2, -1/2, -1/2,   0] # 旋转平移后的 y索引
    # Z_corners_ind = [1/2, -1/2, -1/2,  1/2,  1/2, -1/2, -1/2,  1/2,   0] # 旋转平移后的 z索引.
    
    # corners_3D_Box  = np.array([X_corners, Y_corners, Z_corners], dtype=np.float32) # 得到世界坐标系下的3D Box 8 个顶点
    # corners_3D_Box_ind  = np.array([X_corners_ind, Y_corners_ind, Z_corners_ind], dtype=np.float32) # 得到世界坐标系下的3D Box 8 个顶点
    # obj view
    # x_corners = []
    # y_corners = []
    # z_corners = []
    
    corners = []
    points_index= []
    
    plane_center_point = []
    plane_center_point_index = []
    

    for i in [1, -1]: # 1:right plane ,  -1: left plane
        flag = 1
        if i != flag:
            flag = -1
            # 添加平面中心点
            plane_center_point.append([half_l*1, half_h*0,half_w*0])
            plane_center_point_index.append([0.5, 0, 0])
        for j in np.arange(1,-1-resolution,-resolution): # 1:bottom plane , -1: top plane
            for k in np.arange(1,-1-resolution,-resolution): # 1:front plane ,-1: back plane                            
                # x_corners.append(half_l*i)
                # y_corners.append(half_h*round(j,2))
                # z_corners.append(half_w*round(k,2))
                corners.append([half_l*i, half_h*round(j,2),half_w*round(k,2)])
                points_index.append([0.5*i,0.5*round(j,2),0.5*round(k,2)])
                
    # 添加平面中心点            
    plane_center_point.append([half_l*(-1), half_h*0,half_w*0])
    plane_center_point_index.append([-0.5, 0, 0])
    
    for j in [1, -1]: #  1:bottom plane, -1: top plane
        flag = 1
        if j != flag:
            flag = -1
            # 添加平面中心点
            plane_center_point.append([half_l*0, half_h*1,half_w*0])
            plane_center_point_index.append([0, 0.5, 0])
            
        for k in np.arange(1,-1-resolution,-resolution): # 1:front plane ,-1: back plane
            for i in np.arange(1,-1-resolution,-resolution): #   1:left plane, -1: right plane 
                # x_corners.append(half_l*round(i,2))
                # y_corners.append(half_h*j)
                # z_corners.append(half_w*round(k,2))
                corners.append([half_l*round(i,2), half_h*j,half_w*round(k,2)])
                points_index.append([0.5*round(i,2),0.5*j,0.5*round(k,2)])
    # 添加平面中心点          
    plane_center_point.append([half_l*0, half_h*(-1),half_w*0])
    plane_center_point_index.append([0, -0.5, 0])            
    
    for k in [1, -1]: # 1:front plane ,-1:  back plane
        flag = 1
        if k != flag:
            flag = -1
            # 添加平面中心点
            plane_center_point.append([half_l*0, half_h*0,half_w*1])
            plane_center_point_index.append([0, 0, 0.5])
        for i in np.arange(1,-1-resolution,-resolution): #    1:left plane , -1: right plane 
            for j in np.arange(1,-1-resolution,-resolution): #    1:bottom plane,  -1: top plane            
                # x_corners.append(half_l*round(i,2))
                # y_corners.append(half_h*round(j,2))
                # z_corners.append(half_w*k)
                corners.append([half_l*round(i,2), half_h*round(j,2),half_w*k])
                points_index.append([0.5*round(i,2),0.5*round(j,2),0.5*k])
    
    # 添加平面中心点            
    plane_center_point.append([half_l*0, half_h*0,half_w*(-1)])
    plane_center_point_index.append([0, 0, -0.5])
    
    # 最后加入中心点坐标（底面中心点）            
    # x_corners.append(0)
    # y_corners.append(0)
    # z_corners.append(0)

    corners.append([0,0,0])
    points_index.append([0,0,0])
    # corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32) # 得到世界坐标系下的3D 点坐标
    
    corners =  np.array(corners, dtype=np.float32) # 得到世界坐标系下的3D 点坐标
    points_index = np.array(points_index, dtype=np.float32) # 得到世界坐标系下的3D 点所引
    
    plane_center_point =  np.array(plane_center_point, dtype=np.float32) # 得到世界坐标系下的3D 点坐标
    plane_center_point_index = np.array(plane_center_point_index, dtype=np.float32) # 得到世界坐标系下的3D 点所引
    # 确定每个面的点
    # interval = len(corners[1])//6 
    interval = corners.shape[0]//6 
    
    # rotate if R is passed in
    if rotation_y is not None:
        c, s = np.cos(rotation_y), np.sin(rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32) # 创建旋转矩阵     
        #corners = np.dot(R, corners)
        corners = np.dot(R, corners.transpose(1, 0))
        plane_center_point = np.dot(R, plane_center_point.transpose(1, 0))
        #corners_3d = np.dot(R, corners_3D_Box)  # 先旋转
       
  
        
    
    # shift if location is passed in
    if location is not None:
        corners = corners + np.array(location, dtype=np.float32).reshape(3, 1) # 再平移 shape = (3,9)
        plane_center_point = plane_center_point + np.array(location, dtype=np.float32).reshape(3, 1) # 再平移 shape = (3,9)
        corners[1] = corners[1] -  half_h # 将坐标中心点移动到物体中心位置
        plane_center_point[1] = plane_center_point[1] -  half_h # 将坐标中心点移动到物体中心位置
        #corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) # 再平移 shape = (3,9)
        
    # 得到相机坐标系下的3D 点坐标
    corners = corners.transpose(1, 0)
    plane_center_point = plane_center_point.transpose(1, 0)
    #corners_3d = corners_3d.transpose(1, 0)
    
    right_plane   = corners[0*interval:1*interval]
    left_plane    = corners[1*interval:2*interval]
    bottom_plane  = corners[2*interval:3*interval]
    top_plane     = corners[3*interval:4*interval]
    front_plane   = corners[4*interval:5*interval]
    back_plane    = corners[5*interval:6*interval]
    
    vox_center_point  = corners[6*interval:6*interval+1]
    
    right_plane_p_ind    = points_index[0*interval:1*interval]
    left_plane_p_ind     = points_index[1*interval:2*interval]
    bottom_plane_p_ind   = points_index[2*interval:3*interval]
    top_plane_p_ind      = points_index[3*interval:4*interval]
    front_plane_p_ind    = points_index[4*interval:5*interval]
    back_plane_p_ind     = points_index[5*interval:6*interval]
    
    center_point_p_ind   = points_index[6*interval:6*interval+1]
    
    
    right_plane_cpt   = plane_center_point[0:1]
    left_plane_cpt    = plane_center_point[1:2]
    bottom_plane_cpt  = plane_center_point[2:3]
    top_plane_cpt     = plane_center_point[3:4]
    front_plane_cpt   = plane_center_point[4:5]
    back_plane_cpt    = plane_center_point[5:6]
    
    right_plane_cpt_ind    = plane_center_point_index[0:1]
    left_plane_cpt_ind     = plane_center_point_index[1:2]
    bottom_plane_cpt_ind   = plane_center_point_index[2:3]
    top_plane_cpt_ind      = plane_center_point_index[3:4]
    front_plane_cpt_ind    = plane_center_point_index[4:5]
    back_plane_cpt_ind     = plane_center_point_index[5:6]

    plane = [right_plane,left_plane,bottom_plane,top_plane,front_plane,back_plane]
    plane_p_ind = [right_plane_p_ind,left_plane_p_ind,bottom_plane_p_ind,top_plane_p_ind,front_plane_p_ind,back_plane_p_ind]
    
    plane_cpt = [right_plane_cpt, left_plane_cpt, bottom_plane_cpt, top_plane_cpt, front_plane_cpt, back_plane_cpt]
    plane_cpt_ind = [right_plane_cpt_ind, left_plane_cpt_ind, bottom_plane_cpt_ind, top_plane_cpt_ind, front_plane_cpt_ind, back_plane_cpt_ind]
    
          
    return plane, plane_p_ind, vox_center_point, center_point_p_ind, plane_cpt, plane_cpt_ind



def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32) # 创建旋转矩阵
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2,  l/2, -l/2, -l/2, l/2,  l/2, -l/2, -l/2,   0] # 最后增加了一个中心点坐标 x.
  y_corners = [  0,    0,    0,    0,  -h,   -h,   -h,   -h,   -h/2] # 最后增加了一个中心点坐标 y.
  z_corners = [w/2, -w/2, -w/2,  w/2, w/2, -w/2, -w/2,  w/2,   0] # 最后增加了一个中心点坐标 z.

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32) # 得到世界坐标系下的3D 点坐标
  corners_3d = np.dot(R, corners)  # 先旋转
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1) # 再平移 shape = (3,9)
  
  return corners_3d.transpose(1, 0) # 得到相机坐标系下的3D 点坐标

def project_to_image(pts_3d, P, img_shape):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  h=img_shape[0] # h = 370
  w=img_shape[1] # w = 1224
  pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
                                                                      # 将3D 坐标矩阵后面添加值为1的一列.
                                                                      # array([[ 2.44237   ,  1.47      ,  8.643988  ,  1.        ],
                                                                      #        [ 2.43757   ,  1.47      ,  8.164012  ,  1.        ],
                                                                      #        [ 1.23763   ,  1.47      ,  8.176012  ,  1.        ],
                                                                      #        [ 1.24243   ,  1.47      ,  8.655988  ,  1.        ],
                                                                      #        [ 2.44237   , -0.41999996,  8.643988  ,  1.        ],
                                                                      #        [ 2.43757   , -0.41999996,  8.164012  ,  1.        ],
                                                                      #        [ 1.23763   , -0.41999996,  8.176012  ,  1.        ],
                                                                      #        [ 1.24243   , -0.41999996,  8.655988  ,  1.        ],
                                                                      #        [ 1.84      ,  0.52500004,  8.41      ,  1.        ]],
                                                                      #     dtype=float32)
  pts_2d_center = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0) # 3*4 与 4*9 矩阵相乘得到3*9矩阵， 转置之后为9*3矩阵。
                                                                         # array([[6994.3066  , 2599.314   ,    8.648969],
                                                                         #        [6700.9688  , 2512.6753  ,    8.168993],
                                                                         #        [5859.801   , 2514.8413  ,    8.180993],
                                                                         #        [6153.139   , 2601.48    ,    8.660969],
                                                                         #        [6994.3066  , 1262.9907  ,    8.648969],
                                                                         #        [6700.9688  , 1176.3519  ,    8.168993],
                                                                         #        [5859.801   , 1178.5181  ,    8.180993],
                                                                         #        [6153.139   , 1265.1569  ,    8.660969],
                                                                         #        [6427.0537  , 1888.916   ,    8.414981]], dtype=float32)
  pts_2d_center = pts_2d_center[:, :2] / pts_2d_center[:, 2:]  # 得到投影的2D点坐标。算法：前两列的数/最后一列的数
                                                               # array([[808.68677, 300.53455],
                                                               #        [820.2931 , 307.5869 ],
                                                               #        [716.2701 , 307.40048],
                                                               #        [710.4447 , 300.36826],
                                                               #        [808.68677, 146.0279 ],
                                                               #        [820.2931 , 144.00208],
                                                               #        [716.2701 , 144.05562],
                                                               #        [710.4447 , 146.07567],
                                                               #        [763.7633 , 224.47063]], dtype=float32)
  pts_2d=pts_2d_center[:9,:] # 9个关键点的2D坐标 
  pts_center=pts_2d_center[(9-1):9,:]  # 中心点坐标
  x_pts=pts_2d[:,0:1] # 获得所有2D点的x坐标
  y_pts=pts_2d[:,1:2] # 获得所有2D点的y坐标
  
  is_vis=np.ones(x_pts.shape) # 9*1 的全1矩阵， 初值为1，认为所有点都是可见的
  # 判断x,y投影之后的坐标是否越界， 若是则将值设为0，即不可见。
  is_vis[x_pts>w]=0 
  is_vis[y_pts>h]=0
  is_vis[x_pts < 0] = 0
  is_vis[y_pts < 0] = 0
  # if is_vis[8,0]==0:
  #     is_vis*=0
  vis_num=is_vis.sum() # 求和，统计可见关键点的个数
  is_vis=is_vis*2 # 可见点的个数
  pts_2d=np.column_stack((pts_2d,is_vis))
  # import pdb; pdb.set_trace()
  return pts_2d,vis_num,pts_center

def project_to_image2(pts_3d, P, img_shape, pts_center):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  h=img_shape[0] # h = 370
  w=img_shape[1] # w = 1224
  
  pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_center_homo = np.concatenate([pts_center, np.ones((pts_center.shape[0], 1), dtype=np.float32)], axis=1)
                                                                      
  pts_2d_center = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_center = np.dot(P, pts_center_homo.transpose(1, 0)).transpose(1, 0) 
                                                                         
  pts_2d = pts_2d_center[:, :2] / pts_2d_center[:, 2:]
  pts_center = pts_center[:, :2] / pts_center[:, 2:]
  
  x_pts=pts_2d[:,0:1] # 获得所有2D点的x坐标
  y_pts=pts_2d[:,1:2] # 获得所有2D点的y坐标
  
  is_vis=np.ones(x_pts.shape) # N*1 的全1矩阵， 初值为1，认为所有点都是可见的
  # 判断x,y投影之后的坐标是否越界， 若是则将值设为0，即不可见。
  is_vis[x_pts>w]=0 
  is_vis[y_pts>h]=0
  is_vis[x_pts < 0] = 0
  is_vis[y_pts < 0] = 0
  # if is_vis[8,0]==0:
  #     is_vis*=0
  vis_num=is_vis.sum() # 求和，统计可见关键点的个数
  is_vis=is_vis*2 # 可见点的个数
  pts_2d=np.column_stack((pts_2d,is_vis))
  # import pdb; pdb.set_trace()
  return pts_2d,vis_num, pts_center

def project_to_image3(pts_3d, P,img_shape):
  # pts_3d: n x 3的
  # P: 3 x 4
  # return: n x 2
  h=img_shape[0]
  w=img_shape[1]
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d_center = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d_center = pts_2d_center[:, :2] / pts_2d_center[:, 2:]
  pts_2d=pts_2d_center[:9,:]

  pts_center=pts_2d_center[8:9,:]
  x_pts=pts_2d[:,0:1]
  y_pts=pts_2d[:,1:2]
  is_vis=np.ones(x_pts.shape)
  is_vis[x_pts>w]=0
  is_vis[y_pts>h]=0
  is_vis[x_pts < 0] = 0
  is_vis[y_pts < 0] = 0
  # if is_vis[8,0]==0:
  #     is_vis*=0
  vis_num=is_vis.sum()
  is_vis=is_vis*2
  f = P[0, 0]
  cx, cy = P[ 0, 2], P[1, 2]
  pts_2d[:,0]=(pts_2d[:,0]-cx)/f
  pts_2d[:, 1] = (pts_2d[:, 1] - cy)/ f

  pts_2d=np.column_stack((pts_2d,is_vis))
  # import pdb; pdb.set_trace()
  return pts_2d,vis_num,pts_center
def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(0, 0, 255)):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
               (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
               (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d

def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot(center, alpha, dim, depth, calib):
  # single image
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d

def n_points_project_3d_bbox(location, dim, rotation_y, calib):
    n_points_3d = create_plane_points(dim, location, rotation_y) 
    n_points_2d = project_to_image(n_points_3d, calib) 
    return n_points_2d

if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)