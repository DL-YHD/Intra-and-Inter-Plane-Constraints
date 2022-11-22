# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# train 4 points
# CUDA_VISIBLE_DEVICES=0 python ./src/main.py --data_dir ./kitti_format --split split_1 --exp_id res18_4_points_split_1 --load_model ./kitti_format/exp/res18_4_points_split_1/model_30.pth --arch res_18 --batch_size 8 --master_batch_size 4 --lr 1.25e-4 --lr_step 90,120 --gpus 0 --num_epochs 200

# train 9 points
# CUDA_VISIBLE_DEVICES=0 python ./src/main.py --data_dir ./kitti_format --split split_1 --exp_id res18_9_points_split_1 --load_model ./kitti_format/exp/res18_9_points_split_1/model_30.pth --arch res_18 --batch_size 8 --master_batch_size 4 --lr 1.25e-4 --lr_step 90,120 --gpus 0 --num_epochs 200

# train 25 points
CUDA_VISIBLE_DEVICES=0 python ./src/main.py --data_dir ./kitti_format --split split_1 --exp_id res18_25_points_split_1 --load_model ./kitti_format/exp/res18_25_points_split_1/model_30.pth --arch res_18 --batch_size 8 --master_batch_size 4 --lr 1.25e-4 --lr_step 90,120 --gpus 0 --num_epochs 200

