# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
CUDA_VISIBLE_DEVICES=0 python ./src/main.py --data_dir ./kitti_format --split split_1 --exp_id dla34_4_points_split_1 --arch dla_34 --load_model  ./kitti_format/exp/res18_25_points_split_1/model_30.pth --batch_size 2 --master_batch_size 4 --lr 1.25e-4 --lr_step 90,120 --gpus 0,1 --num_epochs 200
