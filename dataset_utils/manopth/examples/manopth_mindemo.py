import torch
from manopth.manolayer import ManoLayer
from manopth import demo
import numpy as np

batch_size = 1
# Select number of principal components for pose space
ncomps = 6

# Initialize MANO layer
mano_layer = ManoLayer(
    mano_root='mano/models', use_pca=True, ncomps=ncomps, flat_hand_mean=False)

# Generate random shape parameters
# random_shape = torch.rand(batch_size, 10)
random_shape = torch.zeros(batch_size, 10)

# Generate random pose parameters, including 3 values for global axis-angle rotation
# random_pose = torch.ones(batch_size, ncomps + 3)*1.2
random_pose = torch.rand(batch_size, ncomps + 3)


# random_pose = torch.tensor([[-np.pi*0.5,0.0,0.0,0,0,0,0,0,0]])
# random_pose = torch.zeros(batch_size, ncomps + 3)

# Forward pass through MANO layer
hand_verts, hand_joints = mano_layer(random_pose, random_shape)
# hand_joints[0][18:, :] = 0
demo.display_hand({
    'verts': hand_verts,
    'joints': hand_joints
},
                  mano_faces=mano_layer.th_faces)
# print(hand_joints)
# def cal_angle(x,y):
#     Lx = np.sqrt(x.dot(x))
#     Ly = np.sqrt(y.dot(y))
#     cos_angle = x.dot(y)/(Lx*Ly)
#     return np.arccos(cos_angle)*180/np.pi



# hand_joints = np.asarray(hand_joints)
# #0-1 5 9 13 17
# Thumb = hand_joints[0][1, :] - hand_joints[0][0, :]
# Index = hand_joints[0][5, :] - hand_joints[0][0, :]
# Middle = hand_joints[0][9, :] - hand_joints[0][0, :]
# Ring = hand_joints[0][13, :] - hand_joints[0][0, :]
# Pinky = hand_joints[0][17, :] - hand_joints[0][0, :]

# # print(random_pose[0][0:3])
# aaR = np.asarray(random_pose[0][0:3])
# print(np.linalg.norm(aaR)*180/np.pi)
# print(np.linalg.norm(Thumb))
# print(Thumb,Index,Middle,Ring,Pinky)
# print(cal_angle(Thumb,Index))
# print(cal_angle(Index, Middle))
# print(cal_angle(Middle, Ring))
# print(cal_angle(Ring, Pinky))

