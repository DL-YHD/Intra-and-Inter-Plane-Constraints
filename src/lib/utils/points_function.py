import numpy as np
import torch 
import random

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

def svd_function(const, kp_norm, index, kps_3d_coef):
	A = torch.cat([const.float(), kp_norm.float()],dim=1)  # (18, 3)
	B = torch.zeros_like(kp_norm) 
	C = torch.zeros_like(kp_norm) 
	# for i in range(9): 
	#     B[i*2:i*2+1]   = l * kps_3d_coef[i,0:1] * cosori + w * kps_3d_coef[i,2:3] * sinori
	#     B[i*2+1:i*2+2] = h * kps_3d_coef[i,1:2]
	# for i in range(9):
	#     C[i*2:i*2+2] =   l * kps_3d_coef[i,0:1] * (-sinori) + w * kps_3d_coef[i,2:3] * cosori   
	
	A_bak = torch.zeros_like(A)
	for i in index:
	 	A_bak[i*2:i*2+1] = A[i*2:i*2+1]
	 	A_bak[i*2+1:i*2+2] = A[i*2+1:i*2+2]
	for i in index:
	 	B[i*2:i*2+1]   = l * kps_3d_coef[i,0:1] * cosori + w * kps_3d_coef[i,2:3] * sinori
	 	B[i*2+1:i*2+2] = h * kps_3d_coef[i,1:2]
	
	for i in index:
	    C[i*2:i*2+2] =  l * kps_3d_coef[i,0:1] * (-sinori) + w * kps_3d_coef[i,2:3] * cosori

	'''
	B[0:1] = l * 0.5 * cosori + w * 0.5 * sinori
	B[1:2] = h * 0.5
	B[2:3] = l * 0.5 * cosori - w * 0.5 * sinori
	B[3:4] = h * 0.5
	B[4:5] = -l * 0.5 * cosori - w * 0.5 * sinori
	B[5:6] = h * 0.5   
	B[6:7] = -l * 0.5 * cosori + w * 0.5 * sinori
	B[7:8] = h * 0.5
	B[8:9] = l * 0.5 * cosori + w * 0.5 * sinori 
	B[9:10] = -h * 0.5
	B[10:11] = l * 0.5 * cosori - w * 0.5 * sinori
	B[11:12] = -h * 0.5   
	B[12:13] = -l * 0.5 * cosori - w * 0.5 * sinori
	B[13:14] = -h * 0.5
	B[14:15] = -l * 0.5 * cosori + w * 0.5 * sinori
	B[15:16] = -h * 0.5
	B[16:17] = 0
	B[17:18] = 0
	
	
	
	C[0:1] = -l * 0.5 * sinori + w * 0.5 * cosori
	C[1:2] = -l * 0.5 * sinori + w * 0.5 * cosori
	C[2:3] = -l * 0.5 * sinori - w * 0.5 * cosori
	C[3:4] = -l * 0.5 * sinori - w * 0.5 * cosori
	C[4:5] = l * 0.5 * sinori - w * 0.5 * cosori
	C[5:6] = l * 0.5 * sinori - w * 0.5 * cosori
	C[6:7] = l * 0.5 * sinori + w * 0.5 * cosori
	C[7:8] = l * 0.5 * sinori + w * 0.5 * cosori
	C[8:9] =  -l * 0.5 * sinori + w * 0.5 * cosori
	C[9:10] = -l * 0.5 * sinori + w * 0.5 * cosori
	C[10:11] = -l * 0.5 * sinori - w * 0.5 * cosori
	C[11:12] = -l * 0.5 * sinori - w * 0.5 * cosori
	C[12:13] = l * 0.5 * sinori - w * 0.5 * cosori
	C[13:14] = l * 0.5 * sinori - w * 0.5 * cosori
	C[14:15] = l * 0.5 * sinori + w * 0.5 * cosori
	C[15:16] = l * 0.5 * sinori + w * 0.5 * cosori
	C[16:17] = 0
	C[17:18] = 0
	'''

	A = A_bak 
	
	B = B - kp_norm * C # (18)
	AT = A.permute(1,0)  # (18,3) -> (3,18)
	
	#AT = AT.view(b * c, 3, 18) # (32,3,18)
	#A = A.view(b * c, 18, 3) # (32, 18, 3)
	B = B.view(B.size(0), 1).float() # (18,1)
	# mask = mask.unsqueeze(2)
	pinv = torch.mm(AT, A) # (3,18) * (18, 3) = (3, 3)
	pinv = torch.inverse(pinv)  # b*c 3 3 (32,3,3)
	
	
	pinv = torch.mm(pinv, AT) # (32, 3, 18)
	pinv = torch.mm(pinv, B)  # (32, 3, 1)
	
	return pinv.T

# dim = [1.89, 0.48, 1.2]
# location = [1.84, 1.47, 8.41]
# rotation_y = 0.01

dim = [2.71, 2.12, 5.66]
location = [10.90, 1.13, 21.30]
rotation_y = -0.01

# dim = [1.50, 1.62, 3.88] 
# location = [-12.54, 1.64, 19.72] 
# rotation_y = -0.42

# dim = [1.59, 0.53, 1.89]
# location = [2.75, 1.68, 3.14] 
# rotation_y = -1.55

# dim = [1.44, 1.64, 3.78]
# location = [-3.03, 1.57, 13.30]
# rotation_y = 1.68


# dim = [1.67, 1.64, 4.32]
# location = [-2.61, 1.13, 31.73]
# rotation_y = -1.30

# dim =  [1.63, 1.48, 2.37]
# location = [3.23, 1.59, 8.55]
# rotation_y = -1.47

# dim = [1.68, 1.67, 4.29]
# location = [-12.66, 1.13, 38.44]
# rotation_y = 1.73


# dim = [1.8700, 1.6700, 3.6900]
# location = [-16.5300, 1.5550, 58.4900]
# rotation_y = 1.5700

# dim = [1.49, 1.76, 4.01] 
# location = [-15.71, 2.16, 38.26]
# rotation_y = 1.57

# dim = [1.38, 1.80, 3.41]
# location = [-15.89, 2.23, 51.17] 
# rotation_y = 1.58











# image_shape = [370, 1224]

# calib = np.array([[ 7.0705e+02,  0.0000e+00,  6.0408e+02,  4.5758e+01],
#                   [ 0.0000e+00,  7.0705e+02,  1.8051e+02, -3.4542e-01],
#                   [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  4.9810e-03]])


# box_3d = compute_box_3d(dim, location, rotation_y) # 将世界坐标系中的3D坐标转换成相机坐标系
# box_2d_as_point, vis_num, pts_center = project_to_image(box_3d, calib, image_shape) 

# l = torch.tensor(dim[2]) # 预测的长
# h = torch.tensor(dim[0]) # 预测的高
# w = torch.tensor(dim[1]) # 预测的宽

# kpts_3d = torch.tensor(box_3d)

# kpts_2d = torch.tensor(box_2d_as_point[:,:2],dtype=torch.float32)

# 加噪声
# SEED = 123
# torch.manual_seed(SEED)
# np.random.seed(0)
# a = np.random.randint(20, 50, (9, 2))
# kpts_2d += torch.tensor(a,dtype=torch.float32)
# print(a)

# center point coef
# kps_3d_coef = torch.tensor([[1/2, 1/2, 1/2],
#                             [1/2, 1/2,-1/2],                          
#                             [-1/2,1/2,-1/2],
#                             [-1/2,1/2, 1/2],
                            
#                             [1/2,-1/2,1/2],
#                             [1/2,-1/2,-1/2],
#                             [-1/2,-1/2,-1/2],
#                             [-1/2,-1/2,1/2],
#                             [0,0,0]])

# front_cpt_coef = kps_3d_coef.clone()
# front_cpt_coef[:,0] = kps_3d_coef[:,0] - 1/2
        
# back_cpt_coef = kps_3d_coef.clone()
# back_cpt_coef[:,0] = kps_3d_coef[:,0] + 1/2
               
# bottom_cpt_coef = kps_3d_coef.clone()
# bottom_cpt_coef[:,1] = kps_3d_coef[:,1] - 1/2
        
# top_cpt_coef = kps_3d_coef.clone()
# top_cpt_coef[:,1]  = kps_3d_coef[:,1] + 1/2
               
# right_cpt_coef = kps_3d_coef.clone()
# right_cpt_coef[:,2]   = kps_3d_coef[:,2] - 1/2
        
# left_cpt_coef = kps_3d_coef.clone()
# left_cpt_coef[:,2]  = kps_3d_coef[:,2] + 1/2


# bottom point coef
# kps_3d_coef = torch.tensor([[1/2, 0, 1/2],
#                             [1/2, 0,-1/2],                          
#                             [-1/2,0,-1/2],
#                             [-1/2,0, 1/2],
                            
#                             [1/2,-1,1/2],
#                             [1/2,-1,-1/2],
#                             [-1/2,-1,-1/2],
#                             [-1/2,-1,1/2],
#                             [0,-1/2,0]])

# top point coef
# kps_3d_coef = torch.tensor([[1/2, 1, 1/2],
#                             [1/2, 1,-1/2],                          
#                             [-1/2,1,-1/2],
#                             [-1/2,1, 1/2],
                            
#                             [1/2,0,1/2],
#                             [1/2,0,-1/2],
#                             [-1/2,0,-1/2],
#                             [-1/2,0,1/2],
#                             [0,1/2,0]])


# right point coef
# kps_3d_coef = torch.tensor([[0,  1/2, 1/2],
#                             [0,  1/2,-1/2],                          
#                             [-1, 1/2,-1/2],
#                             [-1, 1/2, 1/2],
                            
#                             [0, -1/2,1/2],
#                             [0, -1/2,-1/2],
#                             [-1, -1/2,-1/2],
#                             [-1, -1/2,1/2],
#                             [-1/2,0,0]])

# left point coef
# kps_3d_coef = torch.tensor([[1,  1/2, 1/2],
#                             [1,  1/2,-1/2],                          
#                             [0, 1/2,-1/2],
#                             [0, 1/2, 1/2],
                            
#                             [1, -1/2,1/2],
#                             [1, -1/2,-1/2],
#                             [0, -1/2,-1/2],
#                             [0, -1/2,1/2],
#                             [1/2,0,0]])


# back point coef
# kps_3d_coef = torch.tensor([[1/2, 1/2, 0],
#                             [1/2, 1/2,-1],                          
#                             [-1/2,1/2,-1],
#                             [-1/2,1/2, 0],
                            
#                             [1/2,-1/2,0],
#                             [1/2,-1/2,-1],
#                             [-1/2,-1/2,-1],
#                             [-1/2,-1/2,0],
#                             [0,0,-1/2]])


# front point coef
# kps_3d_coef = torch.tensor([[1/2, 1/2, 1],
#                             [1/2, 1/2,0],                          
#                             [-1/2,1/2,0],
#                             [-1/2,1/2, 1],
                            
#                             [1/2,-1/2,1],
#                             [1/2,-1/2,0],
#                             [-1/2,-1/2,0],
#                             [-1/2,-1/2,1],
#                             [0,0,1/2]])

# const = torch.tensor([[-1, 0], [0, -1], 
#                       [-1, 0], [0, -1], 
#                       [-1, 0], [0, -1], 
#                       [-1, 0], [0, -1], 
#                       [-1, 0], [0, -1], 
#                       [-1, 0], [0, -1],
#                       [-1, 0], [0, -1], 
#                       [-1, 0], [0, -1], 
#                       [-1, 0], [0, -1]])

# calib = torch.tensor(calib)


# cx, cy = calib[0, 2].unsqueeze(0), calib[ 1, 2].unsqueeze(0)
# cxy = torch.cat((cx.float(), cy.float()))

# kpoint = kpts_2d
# fs = calib[0, 0].float()
# f = fs.repeat(18).reshape(9,2)

# kp_norm = (kpoint - cxy) / f
# kp_norm = kp_norm.reshape(-1,1) 

# cosori = torch.cos(torch.tensor(rotation_y)) # 
# sinori = torch.sin(torch.tensor(rotation_y)) # 




# 随机选取N个点
# index = torch.from_numpy(np.random.choice(a = [0,1,2,3,4,5,6,7,8], size=9, replace = False))

# cpt_location = svd_function(const, kp_norm, index, kps_3d_coef)

# right_cpt_location = svd_function(const, kp_norm, index, right_cpt_coef)
# left_cpt_location = svd_function(const, kp_norm, index, left_cpt_coef)
# bottom_cpt_location = svd_function(const, kp_norm, index, bottom_cpt_coef)
# top_cpt_location = svd_function(const, kp_norm, index, top_cpt_coef)
# back_cpt_location = svd_function(const, kp_norm, index, back_cpt_coef)
# front_cpt_location = svd_function(const, kp_norm, index, front_cpt_coef)
# plane_locs_gt = torch.stack((front_cpt_location,back_cpt_location,bottom_cpt_location,top_cpt_location,right_cpt_location,left_cpt_location,),dim=1)

# print(plane_locs_gt)
# (right_cpt_location - left_cpt_location) * (bottom_cpt_location - top_cpt_location) * (front_cpt_location - back_cpt_location)
# 8个点求伪逆
# loc =   np.array([[1.9005],[0.5232],[8.4150]])
