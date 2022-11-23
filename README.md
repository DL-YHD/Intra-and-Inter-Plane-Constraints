## Vertex points are not Enough:Monocular 3D Object Detection via Intra- and Inter-Plane Constraints

Our work is implemented on the RTM3D open source code.
[**KM3D**](https://arxiv.org/abs/2009.00764), [**RTM3D**](https://arxiv.org/abs/2001.03343)


## Experimental environment 
All experiments are tested with Ubuntu 20.04, Pytorch 1.0.0, CUDA 10.0, Python 3.6, single NVIDIA 1070

## Installation
Please refer to [INSTALL.md](readme/INSTALL.md)

## Dataset preparation
Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows: 
```
PlaneCons
├── kitti_format
│   ├── data
│   │   ├── kitti
│   │   |   ├── annotations_split_1 / kitti_train_4_points.json
│   │   |   ├── annotations_split_2 /
│   │   |   ├── training
│   │   |   |   ├── calib /000000.txt .....
│   │   |   |   ├── image_2 /000000.png.....007480.png
│   │   |   |   ├── label_2 /000000.txt.....007480.txt
│   │   |   ├── testing
│   │   |   |   ├── calib /000000.txt .....
│   │   |   |   ├── image_2 /000000.png.....007480.png
│   │   |   |   ├── image_3 /000000.png.....007480.png
|   |   |   ├── train.txt val.txt train_split_1.txt val_split_1.txt 
├── src
├── demo_kitti_format
├── readme
├── requirements.txt
```
# Data Preparation
   ~~~
   cd ./src/tools
   python kitti.py
   ~~~
You can modify the resolution parameter(you can choose [0.5,1,2] three different values) in the [kitti.py](src/tools/kitti.py) file to get preprocessed labels with different number of key points. 
We set the distance between the vertices to 2.

When resolution=2, the vertices of the 3D box are obtained.

When resolution=1, one key point is taken for each plane of the 3D box at an interval of 1 unit, i.e. 9 key points.
 
When resolution=0.5, one key point is taken for each plane of the 3D box at an interval of 1/4 unit.

# Training & Testing & Evaluation
## Training by python with multiple GPUs in a machine
When you  are going to train different numbers of keypoint models, first you must set the n_num_joints parameter in the [kittihp.py](src/lib/datasets/dataset/kittihp.py) and set the corresponding network parameters in the [opts.py](src/lib/opts.py) file.

then
Run following command to train model with ResNet-18 backbone.
   ~~~
   sh ./train_res.sh
   ~~~
Run following command to train model with DLA-34 backbone.
   ~~~
   sh ./train_dla.sh
   ~~~

## Results generation
Run following command for results generation.
   ~~~
   sh ./generate.sh
   ~~~

## Evaluation
Run following command for evaluation.
   ~~~
   sh ./eval.sh
   ~~~

## Visualization
Run following command for visualization.
   ~~~
   sh ./vis.sh
   ~~~

You can modify the parameters in the .sh file

## pre-trained model
We provide our pre-trained model with the following link to Google Cloud Drive:[Google Cloud Drive](https://drive.google.com/file/d/1Y6dDGk5jp4cuUPF-oE2UiWUztuo1ggqk/view?usp=sharing)

## Acknowledgement
- [**KM3D**](https://github.com/Banconxuan/RTM3D)
## License

Plane Constraints is released under the MIT License (refer to the LICENSE file for details).
Portions of the code are borrowed from, [KM3D](https://github.com/Banconxuan/RTM3D),[CenterNet](https://github.com/xingyizhou/CenterNet), [dla](https://github.com/ucbdrive/dla) (DLA network), [DCNv2](https://github.com/CharlesShang/DCNv2)(deformable convolutions), [iou3d](https://github.com/sshaoshuai/PointRCNN) and [kitti_eval](https://github.com/prclibo/kitti_eval) (KITTI dataset evaluation). Please refer to the original License of these projects (See [NOTICE](NOTICE)).
## Citation

If you find this project useful for your research, please use the following BibTeX entry.
    
