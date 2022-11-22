# source
CUDA_VISIBLE_DEVICES=0 python ./src/faster.py --vis --demo ./kitti_format/data/kitti/train_split_1.txt --data_dir ./kitti_format --calib_dir ./kitti_format/data/kitti/training/calib/ --load_model ./kitti_format/exp/res18_25_points_split_1/model_last_4.pth --test_data_used training --gpus 0 --arch res_18
