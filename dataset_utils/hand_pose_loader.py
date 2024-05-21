import os
from os.path import join, abspath, dirname, isfile
import numpy as np
import pickle
import torch
from manopth.manopth.manolayer import ManoLayer
import open3d as o3d


def read_hand_shape(hand_shape_path):
    assert hand_shape_path.endswith(".pkl")
    if not isfile(hand_shape_path):
        return np.zeros(10)
    beta = pickle.load(open(hand_shape_path, "rb"))["hand_shape"]
    beta = np.float32(beta)
    return beta


def mano_params_to_hand_info(hand_pose_path, mano_beta=None, side="right", max_cnt=None, return_pose=False, return_faces=False, device="cuda:0"):
    """
    hand_pose_path: an integrated "pkl" file or a directory including N_frame "pkl" files (named from hand_00001.pkl to hand_{xxxxx}.pkl), no redundant files
    mano_beta: a file path or a numpy array (shape = (10,))
    side: "left" / "right"

    return: a tuple:
        * hand_vertices: a numpy array, shape = (N_frame, 778, 3)
        * hand_joints: a numpy array, shape = (N_frame, 21, 3)
        * (optional) hand_faces: a numpy array, shape = (N_face, 3)
    """

    betas = torch.from_numpy(mano_beta).unsqueeze(0).to(torch.float32).to(device)  # (1, 10)

    mano_layer = ManoLayer(mano_root=join(dirname(abspath(__file__)), 'manopth/mano/models'), use_pca=False, ncomps=45, side=side, center_idx=0)
    mano_layer.to(device)

    theta_list, trans_list = [], []
    
    if hand_pose_path.endswith(".pkl"):
        # print("[mano_params_to_hand_vertices] use the integrated pose file")
        hand_pose_data = pickle.load(open(hand_pose_path, "rb"))
        keys = list(hand_pose_data.keys())
        keys.sort()
        for key in keys:
            theta_list.append(hand_pose_data[key]["hand_pose"].detach().cpu().numpy())
            trans_list.append(hand_pose_data[key]["hand_trans"].detach().cpu().numpy())
    else:
        hand_pose_fns = os.listdir(hand_pose_path)
        hand_pose_fns.sort()
        if not max_cnt is None:
            hand_pose_fns = hand_pose_fns[:max_cnt]

        for hand_pose_fn in hand_pose_fns:
            hand_pose_data = pickle.load(open(join(hand_pose_path, hand_pose_fn), "rb"))
            theta_list.append(hand_pose_data["hand_pose"].detach().cpu().numpy())  # (48,)
            trans_list.append(hand_pose_data["hand_trans"].detach().cpu().numpy())  # (3,)
            
    batch_theta = torch.from_numpy(np.float32(theta_list)).to(device)  # (N_frame, 48)
    batch_trans = torch.from_numpy(np.float32(trans_list)).to(device)  # (N_frame, 3)
    
    betas = betas.repeat(batch_theta.shape[0], 1)  # (N_frame, 10)
    hand_verts_pred, hand_joints_pred, full_pose = mano_layer(batch_theta, betas)
    hand_verts_pred = hand_verts_pred / 1000.0
    hand_joints_pred = hand_joints_pred / 1000.0
    hand_verts_pred += batch_trans.unsqueeze(1)  # (N_frame, 778, 3)
    hand_joints_pred += batch_trans.unsqueeze(1)  # (N_frame, 21, 3)
    
    hand_vertices = hand_verts_pred.detach().cpu().numpy()
    hand_joints = hand_joints_pred.detach().cpu().numpy()
    
    if not return_pose:
        if return_faces:
            return hand_vertices, hand_joints, mano_layer.th_faces.detach().cpu().numpy().reshape(-1, 3)
        return hand_vertices, hand_joints
    else:
        assert not return_faces
        return hand_vertices, hand_joints, batch_theta.detach().cpu().numpy(), batch_trans.detach().cpu().numpy()


if __name__ == "__main__":
    ml = ManoLayer(mano_root=join(dirname(abspath(__file__)), 'manopth/mano/models'), use_pca=False, ncomps=45, side="right", center_idx=0)
    verts, _, _ = ml(torch.zeros(1, 48), torch.zeros(1, 10))
    verts = verts[0] / 1000.0
    print(verts)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts.detach().cpu().numpy())
    o3d.io.write_point_cloud("./TPOSE.ply", pcd)
