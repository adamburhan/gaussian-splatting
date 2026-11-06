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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        # ADAM CHANGE: ScanNet++ folder of the undistorted DSLR test views (tools/split_views.py writes the _ext one)
        self.dslr_folder = "dslr_undistorted_by_iphone"
        self.train_images = "iphone/rgb"
        self.train_sparse = "iphone/colmap"
        self.train_masks = "iphone/rgb_masks"
        # ADAM CHANGE: depth map format, "encoded" (16-bit inverse depth + depth_params.json, the 3DGS default) or "mm" (metric uint16 millimetres)
        self.depth_unit = "encoded"
        # ADAM CHANGE: completion Gaussians from the shape prior (a PLY with x y z nx ny nz red green blue object)
        self.completion = ""                # path; empty = none
        self.completion_opacity = 0.9       # initial opacity (ordinary init points get 0.1)
        self.completion_thickness = 0.002    # scale along the normal [m]: a disc on the surface
        # ADAM CHANGE: which gradients the completion Gaussians receive. all: trained like any Gaussian;
        # colour_frozen: DC and SH gradients zeroed, geometry and opacity train; shape_only: observation-loss
        # gradients zeroed on every parameter, only a shape loss (with shape_targets completion | all) may move
        # them; without a shape loss this is the fully frozen arm
        self.completion_gradients = "colour_frozen"
        # ADAM CHANGE: shape losses (utils/shape_loss.py) on the Gaussians near the prior's object surfaces
        self.shape = ""                     # prior directory (objects.json + distance/<label>.npz); empty = none
        self.shape_loss = "centre"          # centre | surface | band
        self.shape_targets = "ordinary"     # which Gaussians the loss may act on: ordinary | completion | all
        self.shape_opacity = "none"         # weight each Gaussian's penalty by its opacity: none | detached | trainable
        self.shape_radius = 0.15            # ownership: centre within this distance of an object's surface [m]
        self.shape_radius_scale = 0.0       # ownership radius is at least this fraction of the object's box diagonal
        self.shape_robust = "none"          # distance penalty: none (d^2) | huber
        self.shape_robust_scale = 0.03      # huber: quadratic within this distance [m], linear beyond
        self.shape_ownership = "surface"    # surface: any surface within radius | unseen: also within radius of the object's completion samples
        self.shape_refresh = 100            # iterations between ownership updates
        self.shape_spread = 1.0             # surface: weight of the variance-along-the-normal term
        self.shape_tolerance = 0.005        # surface: free thickness [m]
        self.shape_delta = 0.02             # band: free distance to the surface [m]
        self.shape_samples = 16             # band: reparameterised draws per Gaussian
        self.shape_colour_weight = 0.0      # seed colour loss: completion DC toward its object's ordinary neighbours' mean; 0 = off
        # ADAM CHANGE: the completion Gaussians as rigid bodies whose pose trains (needs --completion and --shape)
        self.object_pose = "none"           # none | rigid: a quaternion and translation per prior object, about its centroid
        self.object_observed = ""           # PLY of the objects' observed map points (x y z ... object), the pose's visible evidence
        self.object_fit_weight = 1.0        # weight of the observed-points-to-mesh Huber term
        self.object_fit_samples = 2048      # observed points sampled per object and iteration
        self.object_fit_scale = 0.03        # Huber scale [m]
        self.object_pose_prior_t = 0.05     # Gaussian prior on the translation [m]: 0.5 |t|^2 / s^2 (0 = none)
        self.object_pose_prior_deg = 10.0   # Gaussian prior on the rotation angle [deg] (0 = none)
        self.object_pose_evidence = "all"   # all: rendering losses and the fit term move the pose | fit: the fit term only
                                            # | observed: rendering gradients only through the seeds a camera measured
        self.shape_colour_radius = 0.10     # seed colour loss: neighbourhood radius [m]
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.object_pose_lr = 0.001         # ADAM CHANGE: Adam step of the object pose group (quaternion and metres)
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.shape_weight = 1.0  # ADAM CHANGE: weight of the shape loss (squared metres per Gaussian)
        self.random_background = False
        self.optimizer_type = "default"
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
