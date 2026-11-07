#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, rotation_from_normals  # ADAM CHANGE: rotation_from_normals
from torch import nn
import os
import json
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
except:
    pass

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.completion_index = torch.empty(0, dtype=torch.long)  # ADAM CHANGE: see create_from_pcd; empty when there is no completion
        self.instance_id = torch.empty(0, dtype=torch.long)  # fixed semantic identity, distinct from seed provenance / pose
        self.completion_policy = "legacy"
        self.training_state = {}  # ADAM CHANGE: train.py's resume state, carried by the checkpoint
        # ADAM CHANGE: rigid object pose of the completion Gaussians (enable_object_pose): per prior object a quaternion
        # and a translation about the object's centre. The completed Gaussians' stored xyz/rotation stay canonical and
        # get_xyz/get_rotation apply the pose, so the renderer, the losses and save_ply see the moved object.
        self.object_pose = None  # nn.Parameter [K, 7]: w x y z tx ty tz
        self.object_of = torch.empty(0, dtype=torch.long)  # [N] pose index per Gaussian, -1 ordinary
        self.object_centre = torch.empty(0, 3)  # [K, 3]
        self.object_pose_lr = 0.0
        self.object_pose_render_grad = True  # False: the rendering losses do not move the pose (fit term only)
        self.object_evidence = None  # [N] bool: the seeds whose rendering gradient reaches the pose (observed ones); None = all
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            dict(instance_id=self.instance_id, completion_policy=self.completion_policy, completion_index=self.completion_index,  # ADAM CHANGE: identities, exposure and resume state
                 exposure=self._exposure, exposure_mapping=self.exposure_mapping,
                 exposure_optimizer=self.exposure_optimizer.state_dict(),
                 training_state=self.training_state,
                 object_pose=None if self.object_pose is None else self.object_pose.detach(),
                 object_of=self.object_of, object_centre=self.object_centre, object_pose_lr=self.object_pose_lr,
                 object_evidence=self.object_evidence),
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) not in (12, 13):
            raise ValueError("Unsupported Gaussian checkpoint format")
        metadata = model_args[12] if len(model_args) == 13 else None  # ADAM CHANGE
        if metadata is None and self.completed.any():
            raise ValueError("Legacy completion checkpoint has no source identities; cannot resume safely")
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args[:12]
        self.completion_index = (metadata["completion_index"].to(self._xyz.device) if metadata else
                                 torch.full((len(self._xyz),), -1, dtype=torch.long, device=self._xyz.device))
        if len(self.completion_index) != len(self._xyz):
            raise ValueError("Checkpoint completion identities do not match its Gaussians")
        self.instance_id = (metadata.get("instance_id", torch.full_like(self.completion_index, -1)).to(self._xyz.device)
                            if metadata else torch.full_like(self.completion_index, -1))
        if metadata and metadata.get("completion_policy", "legacy") != self.completion_policy:
            raise ValueError("checkpoint completion policy differs from the requested policy")
        if len(self.instance_id) != len(self._xyz):
            raise ValueError("checkpoint instance identities do not match its Gaussians")
        if metadata is not None:
            self._exposure = nn.Parameter(metadata["exposure"].to(self._xyz.device))
            self.exposure_mapping = metadata["exposure_mapping"]
            self.training_state = metadata["training_state"]
            if metadata.get("object_pose") is not None:  # ADAM CHANGE: the pose group must exist before the optimizer state loads
                self.object_pose = nn.Parameter(metadata["object_pose"].to(self._xyz.device))
                self.object_of = metadata["object_of"].to(self._xyz.device)
                self.object_centre = metadata["object_centre"].to(self._xyz.device)
                self.object_pose_lr = metadata["object_pose_lr"]
                evidence = metadata.get("object_evidence")
                self.object_evidence = None if evidence is None else evidence.to(self._xyz.device)
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if metadata is not None:
            self.exposure_optimizer.load_state_dict(metadata["exposure_optimizer"])

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        if self.object_pose is None or not len(self.object_pose):
            return self.rotation_activation(self._rotation)
        m = self.object_of >= 0  # ADAM CHANGE: the object's rotation composed onto the canonical disc orientation
        q = self.rotation_activation(self.object_pose[:, :4])[self.object_of[m]]
        q = self._evidence_gate(q, m)
        r = self.rotation_activation(self._rotation[m].detach())
        composed = torch.stack([
            q[:, 0] * r[:, 0] - (q[:, 1:] * r[:, 1:]).sum(1),
            *(q[:, 0, None] * r[:, 1:] + r[:, 0, None] * q[:, 1:] + torch.cross(q[:, 1:], r[:, 1:], dim=1)).unbind(1),
        ], 1)
        return self.rotation_activation(self._rotation).index_put((m.nonzero()[:, 0],), composed)

    def _evidence_gate(self, value, m):
        """Per-seed pose values [M, ...] with the gradient path to the pose cut where rendering may not move it."""
        if self.object_pose_render_grad and self.object_evidence is None:
            return value
        if not self.object_pose_render_grad:
            return value.detach()
        keep = self.object_evidence[m].reshape(-1, *([1] * (value.dim() - 1)))
        return torch.where(keep, value, value.detach())

    @property
    def get_xyz(self):
        if self.object_pose is None or not len(self.object_pose):
            return self._xyz
        R, t, c = self.object_transforms()  # ADAM CHANGE: x = R (p - c) + c + t on the completion Gaussians, p canonical
        m = self.object_of >= 0
        k = self.object_of[m]
        Rk, tk = self._evidence_gate(R[k], m), self._evidence_gate(t[k], m)
        moved = torch.einsum("nij,nj->ni", Rk, self._xyz[m].detach() - c[k]) + c[k] + tk
        return self._xyz.index_put((m.nonzero()[:, 0],), moved)

    def object_transforms(self):
        """Rotation [K, 3, 3], translation [K, 3] and centre [K, 3] of every posed object, differentiable in the pose."""
        return build_rotation(self.object_pose[:, :4]), self.object_pose[:, 4:], self.object_centre

    def enable_object_pose(self, object_of, lr):
        """Make the completion Gaussians rigid bodies: object_of [N] is each Gaussian's pose index (-1 ordinary). The
        pose starts at identity about each object's centroid and joins the optimizer as its own group."""
        self.object_of = object_of.to(self._xyz.device)
        self.object_pose_lr = lr
        K = int(object_of.max().item()) + 1 if (object_of >= 0).any() else 0
        centres = torch.zeros(K, 3, device=self._xyz.device)
        for k in range(K):
            members = self._xyz[self.object_of == k].detach()
            if len(members):
                centres[k] = members.mean(0)
        self.object_centre = centres
        pose = torch.zeros(K, 7, device=self._xyz.device)
        pose[:, 0] = 1
        self.object_pose = nn.Parameter(pose)
        if self.optimizer is not None:
            self.optimizer.add_param_group({'params': [self.object_pose], 'lr': lr, "name": "object_pose"})
    
    @property
    def completed(self):  # ADAM CHANGE: boolean mask of the completion Gaussians
        return self.completion_index >= 0

    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(
        self, 
        pcd : BasicPointCloud, 
        cam_infos : int, 
        spatial_lr_scale : float,
        completion=None,
        completion_opacity=0.9,
        completion_thickness=0.002,
        completion_objects=None,
        completion_weights=None
    ):
        self.spatial_lr_scale = spatial_lr_scale
        
        # ADAM CHANGE: completion points are appended after the init cloud so the lines below treat them as
        # ordinary points (SH colour, kNN scale); what differs for them is overridden afterwards with [-n:]
        n = 0 if completion is None else len(completion.points)
        if n:
            pcd = BasicPointCloud(points=np.concatenate([pcd.points, completion.points]),
                                  colors=np.concatenate([pcd.colors, completion.colors]), normals=None)
            print("Number of completion points : ", n)
            
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        if n:  # ADAM CHANGE: discs on the surface, thin along the normal; in-plane scale stays the kNN rule
            rots[-n:] = rotation_from_normals(torch.tensor(np.asarray(completion.normals)).float().cuda())
            scales[-n:, 2] = np.log(completion_thickness)


        opacities = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        if n:  # fixed prior weights scale the seed opacity; legacy files have unit weight
            weights = torch.ones(n, device="cuda") if completion_weights is None else torch.as_tensor(completion_weights, device="cuda")
            opacities[-n:] = completion_opacity * weights[:, None]
        opacities = self.inverse_opacity_activation(opacities)
        # ADAM CHANGE: row of the completion PLY each Gaussian came from, -1 for ordinary ones
        self.completion_index = torch.full((fused_point_cloud.shape[0],), -1, dtype=torch.long, device="cuda")
        self.instance_id = torch.full_like(self.completion_index, -1)
        if n:
            self.completion_index[-n:] = torch.arange(n, device="cuda")


        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.object_pose is not None:  # ADAM CHANGE: restored pose keeps its group so the optimizer state matches
            l.append({'params': [self.object_pose], 'lr': self.object_pose_lr, "name": "object_pose"})

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            try:
                self.optimizer = SparseGaussianAdam(l, lr=0.0, eps=1e-15)
            except:
                # A special version of the rasterizer is required to enable sparse adam
                self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        if self.pretrained_exposures is None:
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()  # ADAM CHANGE: posed positions when an object pose is active
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = (self.get_rotation if self.object_pose is not None else self._rotation).detach().cpu().numpy()  # ADAM CHANGE

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()] + [('instance_id', 'i4')]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        for i, name in enumerate(self.construct_list_of_attributes()):
            elements[name] = attributes[:, i]
        elements["instance_id"] = self.instance_id.cpu().numpy() if len(self.instance_id) else -1
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)
        # ADAM CHANGE: completion Gaussians keep their opacity; no training pixel could raise them back after a reset
        if self.completion_policy == "legacy":
            opacities_new = torch.where(self.completed[:, None], self.get_opacity, opacities_new)
        opacities_new = self.inverse_opacity_activation(opacities_new)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False):
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        ids = np.asarray(plydata.elements[0]["instance_id"]) if "instance_id" in plydata.elements[0].data.dtype.names else np.full(len(xyz), -1)
        self.instance_id = torch.as_tensor(ids.astype(np.int64), device="cuda")

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "object_pose":  # ADAM CHANGE: per object, not per Gaussian
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.instance_id = self.instance_id[valid_points_mask]
        self.completion_index = self.completion_index[valid_points_mask]  # ADAM CHANGE
        if len(self.object_of):
            self.object_of = self.object_of[valid_points_mask]
        if self.object_evidence is not None:
            self.object_evidence = self.object_evidence[valid_points_mask]
        self.tmp_radii = self.tmp_radii[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == "object_pose":  # ADAM CHANGE: per object, not per Gaussian
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_completion_index, new_instance_id):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.instance_id = torch.cat((self.instance_id, new_instance_id))
        self.completion_index = torch.cat((self.completion_index, new_completion_index))  # ADAM CHANGE
        if len(self.object_of):  # only ordinary Gaussians densify
            self.object_of = torch.cat((self.object_of, torch.full_like(new_completion_index, -1)))
        if self.object_evidence is not None:
            self.object_evidence = torch.cat((self.object_evidence, torch.zeros_like(new_completion_index, dtype=torch.bool)))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)
        new_completion_index = self.completion_index[selected_pts_mask].repeat(N)  # ADAM CHANGE

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii, new_completion_index, self.instance_id[selected_pts_mask].repeat(N))

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]
        new_completion_index = self.completion_index[selected_pts_mask]  # ADAM CHANGE

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii, new_completion_index, self.instance_id[selected_pts_mask])

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if self.completion_policy == "legacy" and self.completion_index.numel():  # legacy seeds do not densify
            # (radii > 0).nonzero() is an index tensor [M, 1], not a boolean mask: filter the indices. The guard is for load_ply
            # models, where completion_index is empty; after create_from_pcd it always has one entry per Gaussian
            idx = update_filter.reshape(-1)
            update_filter = idx[~self.completed[idx]][:, None]
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
