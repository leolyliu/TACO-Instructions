"""
a simple wrapper for pytorch3d rendering
"""

import numpy as np
import torch
import pytorch3d
import time
from copy import deepcopy
# Data structures and functions for rendering
from pytorch3d.renderer import (
    PointLights,
    AmbientLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesVertex,
    PerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene


COLOR_LIST = [
    [1.0, 0.0, 0.0],  # hand1
    [0.0, 1.0, 0.0],  # hand2
    [0.0, 0.0, 1.0],  # object1
    [1.0, 0.0, 1.0],  # object2
]


class MeshRendererWrapper:
    "a simple wrapper for the pytorch3d mesh renderer"
    def __init__(self, image_size=(1200, 900),
                 faces_per_pixel=1,
                 device='cuda:0',
                 blur_radius=0, lights=None,
                 materials=None, max_faces_per_bin=50000):
        self.image_size = image_size
        self.faces_per_pixel=faces_per_pixel
        self.max_faces_per_bin=max_faces_per_bin # prevent overflow, see https://github.com/facebookresearch/pytorch3d/issues/348
        self.blur_radius = blur_radius
        self.device = device
        self.lights=lights if lights is not None else AmbientLights(
            ambient_color=((0.5, 0.5, 0.5),), device=device
        )
        self.materials = materials
        self.renderer = self.setup_renderer()

    def setup_renderer(self):
        # for sillhouette rendering
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=self.image_size[::-1],  # input: (h, w)
            blur_radius=self.blur_radius,
            # blur_radius=np.log(1. / 1e-4 - 1.) * sigma, # this will create large sphere for each face
            faces_per_pixel=self.faces_per_pixel,
            clip_barycentric_coords=False,
            max_faces_per_bin=self.max_faces_per_bin
        )
        shader = SoftPhongShader(
            device=self.device,
            lights=self.lights,
            materials=self.materials)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings),
                shader=shader
        )
        return renderer

    def render(self, meshes, cameras, ret_mask=False):
        images = self.renderer(meshes, cameras=cameras)
        # print(images.shape)
        if ret_mask:
            mask = images[0, ..., 3].cpu().detach().numpy()
            return images[0, ..., :3].cpu().detach().numpy(), mask > 0
        return images[0, ..., :3].cpu().detach().numpy()


class Pyt3DWrapper:
    def __init__(self, image_size, device='cuda:0', colors=COLOR_LIST, use_fixed_cameras=False, eyes=None, intrin=None, extrin=None, lights=None):
        self.renderer = MeshRendererWrapper(image_size, device=device, lights=lights)
        self.image_size = image_size
        self.use_fixed_cameras = use_fixed_cameras
        self.device = device
        if use_fixed_cameras:
            self.setup_intrin_extrin(intrin, extrin)
        elif not eyes is None:
            self.cameras = self.get_surround_cameras(eyes=eyes, n_poses=len(eyes), device=device)
        else:
            self.cameras = self.get_surround_cameras(device=device)
        self.colors = deepcopy(colors)
    
    def setup_intrin_extrin(self, intrin, extrin):
        focal_length = torch.tensor((-intrin[0, 0], -intrin[1, 1]), dtype=torch.float32).unsqueeze(0)
        cam_center = torch.tensor((intrin[0, 2], intrin[1, 2]), dtype=torch.float32).unsqueeze(0)
        R = torch.from_numpy(extrin[:3, :3].T).unsqueeze(0)
        T = torch.from_numpy(extrin[:3, 3]).unsqueeze(0)
        pyt3d_version = pytorch3d.__version__
        if pyt3d_version >= '0.6.0':
            self.cameras = [PerspectiveCameras(focal_length=focal_length, principal_point=cam_center, image_size=((self.image_size[1], self.image_size[0]),), device=self.device, R=R, T=T, in_ndc=False)]
        else:
            self.cameras = [PerspectiveCameras(focal_length=focal_length, principal_point=cam_center, image_size=((self.image_size[1], self.image_size[0]),), device=self.device, R=R, T=T)]
    
    @staticmethod
    def get_surround_cameras(radius=0.5, eyes=None, n_poses=30, up=(0.0, 0.0, 1.0), device='cuda:0'):
        fx, fy = 1000, 1000  # focal length
        cx, cy = 512, 512  # camera centers
        color_w, color_h = 1024, 1024  # mock
        cam_center = torch.tensor((cx, cy), dtype=torch.float32).unsqueeze(0)
        focal_length = torch.tensor((fx, fy), dtype=torch.float32).unsqueeze(0)

        cameras = []
        if eyes is None:
            eyes = []
            for theta in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
                if np.abs(up[1]) > 0:
                    eye = [np.cos(theta + np.pi / 2) * radius, 0.2, -np.sin(theta + np.pi / 2) * radius]
                else:
                    eye = [np.cos(theta + np.pi / 2) * radius, np.sin(theta + np.pi / 2) * radius, 0.2]
                print(eye)
                eyes.append(eye)
        
        for eye in eyes:
            R, T = look_at_view_transform(
                eye=(eye,),
                at=([0.0, 0.0, 0.0],),
                up=(up,),
            )
            pyt3d_version = pytorch3d.__version__
            if pyt3d_version >= '0.6.0':
                cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center,
                                        image_size=((color_h, color_w),),
                                        device=device,
                                        R=R, T=T, in_ndc=False)
            else:
                cam = PerspectiveCameras(focal_length=focal_length, principal_point=cam_center,
                                        image_size=((color_h, color_w),),
                                        device=device,
                                        R=R, T=T)
            cameras.append(cam)
        return cameras

    def render_meshes(self, meshes, specified_vertices=None):
        """
        render a list of meshes
        :param meshes: a list of psbody meshes
        :return: rendered image
        """
        colors = deepcopy(self.colors)
        pyt3d_mesh = self.prepare_render(meshes, colors, specified_vertices)
        rends = []
        for cam in self.cameras:
            img = self.renderer.render(pyt3d_mesh, cam)
            rends.append(img)
        return rends

    def prepare_render(self, meshes, colors, specified_vertices, static_contact_color=[0.9, 0.2, 0.2]):
        py3d_meshes = []
        if specified_vertices is None:
            specified_vertices = [None] * len(meshes)
        for mesh, sp_vs, color in zip(meshes, specified_vertices, colors):
            vc = np.zeros_like(mesh.vertices)
            vc[:, :] = color
            if not sp_vs is None:  # render contact
                vc[sp_vs] = static_contact_color
            text = TexturesVertex([torch.from_numpy(vc).float().to(self.device)])
            py3d_mesh = Meshes([torch.from_numpy(mesh.vertices).float().to(self.device)], [torch.from_numpy(mesh.faces.astype(int)).long().to(self.device)],
                               text)
            py3d_meshes.append(py3d_mesh)
        joined = join_meshes_as_scene(py3d_meshes)
        return joined
