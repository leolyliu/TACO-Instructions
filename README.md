# :sparkles: [CVPR 2024] TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding :sparkles:

[CVPR 2024] Official repository of "TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding".

### :page_with_curl:[Paper](https://arxiv.org/pdf/2401.08399.pdf) | :house:[Project](https://taco2024.github.io/) | :movie_camera:[Video](https://youtu.be/bIgHylU1oZo) | :file_folder:[Dataset (pre-released version)](https://1drv.ms/f/s!Ap-t7dLl7BFUfmNkrHubnoo8LCs?e=1h0Xhe) | :file_folder:[Dataset](https://1drv.ms/f/c/5411ece5d2edad9f/EkeIFARuXYVNqkYfROjOVD8BhySYm5fzK7-8OkPLBYjz5g?e=7oLmIq)

#### Authors

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, Li Yi

## Data Instruction

The [pre-released version](https://1drv.ms/f/s!Ap-t7dLl7BFUfmNkrHubnoo8LCs?e=1h0Xhe) contains 244 high-quality motion sequences spanning 138 <tool, action, object> triplets. Please refer to the "Data Visualization" section for data usage.

The [whole dataset](https://1drv.ms/f/c/5411ece5d2edad9f/EkeIFARuXYVNqkYfROjOVD8BhySYm5fzK7-8OkPLBYjz5g?e=7oLmIq) contains the overall 2546 motion sequences, annotations will be released very soon. If you have questions about the dataset, please contact ```yun-liu22@mails.tsinghua.edu.cn```.

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
<install PyTorch >= 1.7.1>
<install PyTorch3D >= 0.6.1>
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
