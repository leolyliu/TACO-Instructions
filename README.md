# :sparkles: [CVPR 2024] TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding :sparkles:

[CVPR 2024] Official repository of "TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding".

### :page_with_curl:[Paper](https://arxiv.org/pdf/2401.08399.pdf) | :house:[Project](https://taco2024.github.io/) | :movie_camera:[Video](https://youtu.be/bIgHylU1oZo) | :file_folder:[Dataset (pre-released version)](https://1drv.ms/f/s!Ap-t7dLl7BFUfmNkrHubnoo8LCs?e=1h0Xhe)

#### Authors

Yun Liu, Haolin Yang, Xu Si, Ling Liu, Zipeng Li, Yuxiang Zhang, Yebin Liu, Li Yi

## Data Visualization

[1] Environment Setup:

TODO

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

## Citation

If you find our work useful in your research, please consider citing:

```
@article{liu2024taco,
  title={TACO: Benchmarking Generalizable Bimanual Tool-ACtion-Object Understanding},
  author={Liu, Yun and Yang, Haolin and Si, Xu and Liu, Ling and Li, Zipeng and Zhang, Yuxiang and Liu, Yebin and Yi, Li},
  journal={arXiv preprint arXiv:2401.08399},
  year={2024}
}
```

## Email

If you have any questions, please contact ```yun-liu22@mails.tsinghua.edu.cn```.
