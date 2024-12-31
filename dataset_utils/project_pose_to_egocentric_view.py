import os
from os.path import join, isfile, isdir, dirname, abspath
import argparse
import numpy as np
import trimesh
import cv2
import imageio
import pickle
from tqdm import tqdm
from pyt3d_wrapper import Pyt3DWrapper
from pytorch3d.renderer import PointLights
from hand_pose_loader import read_hand_shape, mano_params_to_hand_info
from video_utils import mp42imgs, overlay_two_imgs


def visualize(tool_model, target_model, tool_poses, target_poses, right_hand_meshes, left_hand_meshes, egocentric_intrinsic, egocentric_extrinsics, egocentric_imgs=None, save_path=None, sampling_rate=1, device="cuda:0"):
    N = tool_poses.shape[0]
    assert target_poses.shape[0] == N

    image_size = (1920, 1080) if egocentric_imgs is None else (egocentric_imgs[0].shape[1], egocentric_imgs[0].shape[0])
    
    tool_pts = tool_model.vertices
    tool_faces = tool_model.faces
    target_pts = target_model.vertices
    target_faces = target_model.faces
    
    pyt3d_wrapper = Pyt3DWrapper(image_size=image_size, use_fixed_cameras=True, intrin=egocentric_intrinsic, extrin=np.eye(4), device=device, lights=PointLights(device=device, location=[[0.0, 0.0, 2]]))
    
    print("###### start visualization ... ######")
    
    assert not save_path is None
    os.makedirs(dirname(save_path), exist_ok=True)
    rgb_imgs = []
    
    for frame_idx in tqdm(range(0, N, sampling_rate)):
        # get object poses in this frame
        tool_pose = tool_poses[frame_idx]
        target_pose = target_poses[frame_idx]
        
        # construct object meshes in this frame
        tool_mesh = trimesh.Trimesh(vertices=(tool_pts.copy() @ tool_pose[:3, :3].T) + tool_pose[:3, 3], faces=tool_faces.copy())
        target_mesh = trimesh.Trimesh(vertices=(target_pts.copy() @ target_pose[:3, :3].T) + target_pose[:3, 3], faces=target_faces.copy())
        right_hand_mesh = right_hand_meshes[frame_idx]
        left_hand_mesh = left_hand_meshes[frame_idx]
        meshes = [right_hand_mesh, left_hand_mesh, tool_mesh, target_mesh]

        # setup camera extrinsic
        pyt3d_wrapper.setup_intrin_extrin(intrin=egocentric_intrinsic, extrin=egocentric_extrinsics[frame_idx])
        
        # render
        render_result = pyt3d_wrapper.render_meshes(meshes)
        img = (render_result[0]*255).astype(np.uint8)
        # overlay
        if not egocentric_imgs is None:
            img = overlay_two_imgs(egocentric_imgs[frame_idx], img)
        
        rgb_imgs.append(cv2.resize(img, (image_size[0] // 2, image_size[1] // 2)))
    
    # save
    imageio.mimsave(save_path, rgb_imgs, duration=(1000/(30//sampling_rate)), loop=0)

    print("###### finish visualization !!! ######")


def process(args):
    dataset_root = args.dataset_root
    object_model_root = args.object_model_root
    triplet = args.triplet
    sequence_name = args.sequence_name
    save_path = args.save_path
    device = args.device
    
    object_pose_dir = join(dataset_root, "Object_Poses", triplet, sequence_name)
    hand_pose_dir = join(dataset_root, "Hand_Poses", triplet, sequence_name)
    egocentric_intrinsic_filepath = join(dataset_root, "Egocentric_Camera_Parameters", triplet, sequence_name, "egocentric_intrinsic.txt")
    egocentric_frame_extrinsic_filepath = join(dataset_root, "Egocentric_Camera_Parameters", triplet, sequence_name, "egocentric_frame_extrinsic.npy")
    egocentric_rgb_filepath = join(dataset_root, "Egocentric_RGB_Videos", triplet, sequence_name, "color.mp4")
    
    # get tool and target names
    tool_name, target_name = None, None
    for file_name in os.listdir(object_pose_dir):
        if file_name.startswith("tool_"):
            tool_name = file_name.split(".")[0].split("_")[-1]
        elif file_name.startswith("target_"):
            target_name = file_name.split(".")[0].split("_")[-1]
    assert (not tool_name is None) and (not target_name is None)
    
    # load object models and poses
    tool_model = trimesh.load(join(object_model_root, tool_name + "_cm.obj"))
    tool_model.vertices *= 0.01  # unit: m
    target_model = trimesh.load(join(object_model_root, target_name + "_cm.obj"))
    target_model.vertices *= 0.01  # unit: m
    tool_poses = np.load(join(object_pose_dir, "tool_" + tool_name + ".npy"))
    target_poses = np.load(join(object_pose_dir, "target_" + target_name + ".npy"))
    N_frame = tool_poses.shape[0]
    assert target_poses.shape[0] == N_frame
    
    # load hand shapes and poses
    right_hand_beta = pickle.load(open(join(hand_pose_dir, "right_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy()
    left_hand_beta = pickle.load(open(join(hand_pose_dir, "left_hand_shape.pkl"), "rb"))["hand_shape"].reshape(10).detach().cpu().numpy()
    right_hand_vertices, _, right_hand_model_faces = mano_params_to_hand_info(join(hand_pose_dir, "right_hand.pkl"), mano_beta=right_hand_beta, side="right", max_cnt=None, return_pose=False, return_faces=True, device=device)
    left_hand_vertices, _, left_hand_model_faces = mano_params_to_hand_info(join(hand_pose_dir, "left_hand.pkl"), mano_beta=left_hand_beta, side="left", max_cnt=None, return_pose=False, return_faces=True, device=device)
    assert right_hand_vertices.shape[0] == N_frame
    assert left_hand_vertices.shape[0] == N_frame
    right_hand_meshes = []
    left_hand_meshes = []
    for i in range(N_frame):
        right_hand_meshes.append(trimesh.Trimesh(vertices=right_hand_vertices[i].copy(), faces=right_hand_model_faces.copy()))
        left_hand_meshes.append(trimesh.Trimesh(vertices=left_hand_vertices[i].copy(), faces=left_hand_model_faces.copy()))
    
    # load egocentric intrinsic, camera poses, frames
    egocentric_intrinsic = np.loadtxt(egocentric_intrinsic_filepath)  # shape = (3, 3)
    egocentric_extrinsics = np.load(egocentric_frame_extrinsic_filepath)  # world_to_camera, shape = (N_frame, 4, 4)
    egocentric_imgs = mp42imgs(egocentric_rgb_filepath, return_rgb=True, max_cnt=None)
    if len(egocentric_imgs) != N_frame:
        print("[error] losing frames in the egocentric video, skip!")
        return None

    # visualize object poses as an RGB video
    visualize(tool_model, target_model, tool_poses, target_poses, right_hand_meshes, left_hand_meshes, egocentric_intrinsic, egocentric_extrinsics, egocentric_imgs, save_path=save_path, sampling_rate=1, device=device)


def get_args():
    parser = argparse.ArgumentParser()
    ###################################################################
    parser.add_argument('--dataset_root', type=str, default="/cephfs_yili/backup/datasets/HOI-mocap_backup_20240423/TACO_Dataset_V1")
    parser.add_argument('--object_model_root', type=str, default="/cephfs_yili/backup/datasets/HOI-mocap_backup_20240423/object_models_released")
    parser.add_argument('--triplet', type=str, default="(brush, brush, box)")
    parser.add_argument('--sequence_name', type=str, default="20231006_163")
    parser.add_argument('--save_path', type=str, default="./egocentric_example.gif")
    parser.add_argument('--device', type=str, default="cuda:0")
    ###################################################################

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    process(args)
