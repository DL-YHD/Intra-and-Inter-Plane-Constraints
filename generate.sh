# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python ./src/faster.py --demo ./kitti_format/data/kitti/val_split_1.txt --data_dir ./kitti_format --test_data_used training --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/res18_4_points_split_1/model_last.pth --gpus 0 --arch res_18
# python ./src/faster.py --demo ./kitti_format/data/kitti/val_split_1.txt --data_dir ./kitti_format --test_data_used training --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/res18_9_points_split_1/model_last.pth --gpus 0 --arch res_18
# python ./src/faster.py --demo ./kitti_format/data/kitti/val_split_1.txt --data_dir ./kitti_format --test_data_used training --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/res18_25_points_split_1/model_best_0.5_weights.pth --gpus 0 --arch res_18

# python ./src/faster.py --demo ./kitti_format/data/kitti/val_split_1.txt --data_dir ./kitti_format --test_data_used training --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/dla34_4_points_split_1/model_last_5.pth --gpus 0 --arch dla_34
# python ./src/faster.py --demo ./kitti_format/data/kitti/val_split_1.txt --data_dir ./kitti_format --test_data_used training --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/dla34_9_points_split_1/model_last.pth --gpus 0 --arch dla_34
# python ./src/faster.py --demo ./kitti_format/data/kitti/val_split_1.txt --data_dir ./kitti_format --test_data_used training --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/dla34_25_points_split_1/model_100.pth --gpus 0 --arch dla_34
