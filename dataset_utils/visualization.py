import os
from os.path import join, isfile, isdir, dirname, abspath
import argparse
import numpy as np
import trimesh
import cv2
import pickle
from tqdm import tqdm
from pyt3d_wrapper import Pyt3DWrapper
from pytorch3d.renderer import PointLights
from hand_pose_loader import read_hand_shape, mano_params_to_hand_info


IMAGE_SIZE = (1024, 750)
INTRINSIC = np.float32([
    [9533.359863759411, 0.0, 2231.699969508665],
    [0.0, 9593.722282299485, 1699.3865932992662],
    [0.0,0.0,1.0]
]) / 4.0
CAMERA_TO_WORLD = np.float32([
    [-9.861640182402281463e-01, -6.260122508879048531e-02, 1.534979340110796397e-01, -2.296398434650619436e-01],
    [-1.649820266894134468e-01, 2.803133276113436434e-01, -9.456243277501432676e-01, 1.459474531833096833e+00],
    [1.616972472681078854e-02, -9.578650870455813759e-01, -2.867629945118802537e-01, 1.091449991261020935e+00],
    [0, 0, 0, 1],
])
EXTRINSIC = np.linalg.inv(CAMERA_TO_WORLD)


def visualize(tool_model, target_model, tool_poses, target_poses, right_hand_meshes, left_hand_meshes, save_path=None, sampling_rate=1, device="cuda:0"):
    N = tool_poses.shape[0]
    assert target_poses.shape[0] == N
    
    tool_pts = tool_model.vertices
    tool_faces = tool_model.faces
    target_pts = target_model.vertices
    target_faces = target_model.faces
    
    pyt3d_wrapper = Pyt3DWrapper(image_size=IMAGE_SIZE, use_fixed_cameras=True, intrin=INTRINSIC, extrin=EXTRINSIC, device=device, lights=PointLights(device=device, location=[[0.0, 0.0, 2]]))
    
    print("###### start visualization ... ######")
    
    assert not save_path is None
    os.makedirs(dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, 30 // sampling_rate, IMAGE_SIZE)
    
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
        
        # render
        render_result = pyt3d_wrapper.render_meshes(meshes)
        img = (render_result[0]*255).astype(np.uint8)
        
        # save
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video_writer.write(img)
    
    video_writer.release()

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
    
    # visualize object poses as an RGB video
    visualize(tool_model, target_model, tool_poses, target_poses, right_hand_meshes, left_hand_meshes, save_path=save_path, sampling_rate=1, device=device)


def get_args():
    parser = argparse.ArgumentParser()
    ###################################################################
    parser.add_argument('--dataset_root', type=str, default="/cephfs_yili/backup/datasets/HOI-mocap_backup_20240423/TACO_Dataset_Examples")
    parser.add_argument('--object_model_root', type=str, default="/cephfs_yili/backup/datasets/HOI-mocap_backup_20240423/object_models_released")
    parser.add_argument('--triplet', type=str, default="(stir, spoon, bowl)")
    parser.add_argument('--sequence_name', type=str, default="20231105_019")
    parser.add_argument('--save_path', type=str, default="./example.mp4")
    parser.add_argument('--device', type=str, default="cuda:0")
    ###################################################################

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    process(args)
