import os
from os.path import join, isfile, isdir
import argparse
import numpy as np
import cv2
import json
from tqdm import tqdm
from video_utils import mp42imgs


def process(args, sequence_names, SN):
    dataset_root = args.dataset_root
    allocentric_RGB_dir = join(dataset_root, "Allocentric_RGB_Videos")
    allocentric_camera_params_dir = join(dataset_root, "Allocentric_Camera_Parameters")
    save_dir = args.save_dir

    for sequence_name in tqdm(sequence_names):
        sequence_triplet = None
        for triplet in os.listdir(allocentric_RGB_dir):
            if isdir(join(allocentric_RGB_dir, triplet, sequence_name)):
                sequence_triplet = triplet
                break
        assert not sequence_triplet is None
        img = mp42imgs(join(allocentric_RGB_dir, sequence_triplet, sequence_name, SN + ".mp4"), return_rgb=False, max_cnt=5)[0]  # bgr
        camera_params = json.load(open(join(allocentric_camera_params_dir, sequence_triplet, sequence_name, "calibration.json"), "r"))[SN]
        intrinsic = np.float32(camera_params["K"]).reshape(3, 3)  # (3, 3)
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.float32(camera_params["R"]).reshape(3, 3)
        extrinsic[:3, 3] = np.float32(camera_params["T"])

        world_axis_points = np.float32([
            [0, 0, 0],
            [0.3, 0, 0],
            [0, 0.3, 0],
            [0, 0, 0.3],
        ])
        world_axis_points_in_camera_space = (world_axis_points @ extrinsic[:3, :3].T) + extrinsic[:3, 3]
        world_axis_points_xyd = world_axis_points_in_camera_space @ intrinsic.T
        world_axis_points_uv = world_axis_points_xyd[:, :2] / world_axis_points_xyd[:, 2:]
        world_axis_points_uv = world_axis_points_uv.astype(np.int32)
        cv2.arrowedLine(img, world_axis_points_uv[0], world_axis_points_uv[1], (0, 0, 255), 5)  # x-axis
        cv2.arrowedLine(img, world_axis_points_uv[0], world_axis_points_uv[2], (0, 255, 0), 5)  # y-axis
        cv2.arrowedLine(img, world_axis_points_uv[0], world_axis_points_uv[3], (255, 0, 0), 5)  # z-axis
        image_size = (img.shape[1], img.shape[0])
        img = cv2.resize(img, (image_size[0] // 8, image_size[1] // 8))
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(join(save_dir, sequence_name + "_" + SN + "_0000.png"), img)


def get_args():
    parser = argparse.ArgumentParser()
    ###################################################################
    parser.add_argument('--dataset_root', type=str, default="/cephfs_yili/backup/datasets/HOI-mocap_backup_20240423/TACO_Dataset_V1")
    parser.add_argument('--save_dir', type=str, default="../assets/world_coordinate_system_visualizations")
    parser.add_argument('--device', type=str, default="cuda:0")
    ###################################################################

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    sequence_names = [
        "20230917_006",
        "20230919_038",
        "20230923_019",
        "20230926_017",
        "20230927_032",
        "20230928_033",
        "20230929_016",
        "20230930_015",
        "20231002_058",
        "20231005_032",
        "20231006_145",
        "20231010_194",
        "20231013_215",
        "20231015_118",
        "20231019_220",
        "20231020_009",
        "20231024_116",
        "20231026_006",
        "20231027_052",
        "20231031_182",
        "20231102_069",
        "20231103_068",
        "20231104_140",
        "20231105_053",
    ]
    SN = "22139908"
    process(args, sequence_names, SN)
