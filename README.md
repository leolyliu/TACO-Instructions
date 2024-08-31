# :sparkles: [CVPR 2024] TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding :sparkles:

[CVPR 2024] Official repository of "TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding".

### :page_with_curl:[Paper](https://arxiv.org/pdf/2401.08399.pdf) | :house:[Project](https://taco2024.github.io/) | :movie_camera:[Video](https://youtu.be/bIgHylU1oZo) | :file_folder:[Dataset (pre-released version)](https://1drv.ms/f/s!Ap-t7dLl7BFUfmNkrHubnoo8LCs?e=1h0Xhe) | :file_folder:[Dataset](https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/AM4Ia7pK_b0DURAVyxpHLuY?rlkey=e76q06hyj9yqbahhipmf5ij1o&st=c30zhh8s&dl=0)

#### Authors

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, Li Yi

## Data Instruction

### Pre-released Version

The [pre-released version](https://1drv.ms/f/s!Ap-t7dLl7BFUfmNkrHubnoo8LCs?e=1h0Xhe) contains 244 high-quality motion sequences spanning 137 <tool, action, object> triplets. Please refer to the "Data Visualization" section for data usage.

We back up the data at [BaiduNetDisk](https://pan.baidu.com/s/1gANrhzdUyvsUGXcDB4xMfQ?pwd=kg7j). Some of the files are split due to file size limitations. To get the original zip files, please use the following commands:

```
cat Allocentric_RGB_Videos_split.* > Allocentric_RGB_Videos.zip
cat Egocentric_Depth_Videos_split.* > Egocentric_Depth_Videos.zip
```

Dataset contents:

* **244** high-quality motions sequences spanning **137** ```<tool, action, object>``` triplets
* **206** High-resolution object models (10K~100K faces per object mesh)
* Hand-object pose and mesh annotations
* Egocentric RGB-D videos
* **8** allocentric RGB videos

### Whole Dataset

The [whole dataset](https://www.dropbox.com/scl/fo/6wux06w26exuqt004eg1a/AM4Ia7pK_b0DURAVyxpHLuY?rlkey=e76q06hyj9yqbahhipmf5ij1o&st=c30zhh8s&dl=0) contains 2316 motion sequences. Please refer to the "Data Visualization" section for data usage.

Dataset contents:

* **2316** motions sequences spanning **151** ```<tool, action, object>``` triplets
* **206** High-resolution object models (10K~100K faces per object mesh)
* Hand-object pose and mesh annotations
* Egocentric RGB-D videos
* **12** allocentric RGB videos
* Camera parameters
* Automatic Hand-object 2D segmentations
* Automatic marker-removed images

If you have questions about the dataset, please contact ```yun-liu22@mails.tsinghua.edu.cn```.

## Data Organization

The files of the dataset are organized as follows:

```x
|-- Allocentric_RGB_Videos
  |-- <triplet_1>
    |-- <sequence_1>
      |-- 22070938.mp4
      |-- 22139905.mp4
      ...
    |-- <sequence_2>
    ...
  |-- <triplet_2>
  ...
|-- Egocentric_Depth_Videos
  |-- <triplet_1>
    |-- <sequence_1>
      egocentric_depth.avi
    |-- <sequence_2>
    ...
  |-- <triplet_2>
  ...
|-- Egocentric_RGB_Videos
  |-- <triplet_1>
    |-- <sequence_1>
      color.mp4
    |-- <sequence_2>
    ...
  |-- <triplet_2>
  ...
|-- Hand_Poses
  |-- <triplet_1>
    |-- <sequence_1>
      left_hand_shape.pkl
      left_hand.pkl
      right_hand_shape.pkl
      right_hand.pkl
    |-- <sequence_2>
    ...
  |-- <triplet_2>
  ...
|-- Object_Poses
  |-- <triplet_1>
    |-- <sequence_1>
      target_<target_name>.npy
      tool_<tool_name>.npy
    |-- <sequence_2>
    ...
  |-- <triplet_2>
  ...
|-- Object_Models
  |-- 001_cm.obj
  ...
  |-- 218_cm.obj
```

## Data Visualization

[1] Environment Setup:

Our code is tested on Ubuntu 20.04 with NVIDIA GeForce RTX 3090. The driver version is 535.146.02. The CUDA version is 12.2.

Please install the environment using the following commands:

```x
conda create -n taco python=3.9
conda activate taco
<install PyTorch >= 1.7.1, we use PyTorch 1.11.0>
<install PyTorch3D >= 0.6.1, we use PyTorch3D 0.7.2>
pip install -r requirements.txt
```

[2] Download [MANO models](https://mano.is.tue.mpg.de/), and put ```MANO_LEFT.pkl``` and ```MANO_RIGHT.pkl``` in the folder ```dataset_utils/manopth/mano/models```.

[3] Visualize Hand-Object Poses:

```x
cd dataset_utils
python visualization.py --dataset_root <dataset root directory> --object_model_root <object model root directory> --triplet <triplet name> --sequence_name <sequence name> --save_path <path to save the visualization result> --device <device for the rendering process>
```

For example, if you select the following data sequence:

```x
python visualization.py --dataset_root <dataset root directory> --object_model_root <object model root directory> --triplet "(stir, spoon, bowl)" --sequence_name "20231105_019" --save_path "./example.gif" --device "cuda:0"
```

You can obtain the following visualization result:

<img src="https://raw.githubusercontent.com/leolyliu/TACO-Instructions/master/assets/example.gif" width="1024"/>

[4] Parse Egocentric Depth Videos:

Please use the following command for each video:

```x
ffmpeg -i <egocentric depth video path> -f image2 -start_number 0 -vf fps=fps=30 -qscale:v 2 <egocentric depth image save path>
```

For example:

```x
mkdir ./decode
ffmpeg -i ./egocentric_depth.avi -f image2 -start_number 0 -vf fps=fps=30 -qscale:v 2 ./decode/%05d.png
```

Each depth image is a 1920x1080 uint16 array. The depth scale is 1000 (i.e. depth values are stored in millimeters).

## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{liu2024taco,
  title={Taco: Benchmarking generalizable bimanual tool-action-object understanding},
  author={Liu, Yun and Yang, Haolin and Si, Xu and Liu, Ling and Li, Zipeng and Zhang, Yuxiang and Liu, Yebin and Yi, Li},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={21740--21751},
  year={2024}
}
```

## License

This work is licensed under a [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Email

If you have any questions, please contact ```yun-liu22@mails.tsinghua.edu.cn```.
