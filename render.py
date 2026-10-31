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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import numpy as np
from PIL import Image
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh):
    out = os.path.join(model_path, name, "ours_{}".format(iteration))
    render_path, depth_path, alpha_path = (os.path.join(out, d) for d in ("renders", "depth", "alpha"))
    for d in (render_path, depth_path, alpha_path):
        makedirs(d, exist_ok=True)

    # ADAM CHANGE: files keep the source image names so the evaluators can pair them with their ground truth;
    # ground truth is not copied. Next to the RGB, the benchmark's depth convention (docs/contract.md): the
    # opacity-normalised expected depth sum(w z) / sum(w) in metres as uint16 mm, 0 where the accumulated
    # opacity sum(w) is below 0.5, and that opacity as uint8. Both come from one extra unclamped pass
    # rasterizing [1, z, z] as colours over a black background, z the view-space depth of each centre.
    black = torch.zeros(3, dtype=torch.float32, device="cuda")
    for view in tqdm(views, desc="Rendering progress"):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]
        xyz = gaussians.get_xyz
        view_t = view.world_view_transform  # stored transposed: camera coordinates = [x, 1] @ view_t
        z = (xyz @ view_t[:3, 2] + view_t[3, 2]).clamp(min=0.2)  # the rasterizer culls z < 0.2
        accumulated = render(view, gaussians, pipeline, black, override_color=torch.stack([torch.ones_like(z), z, z], 1),
                             clamp_output=False)["render"]
        alpha, z_sum = accumulated[0], accumulated[1]
        expected_depth = torch.where(alpha >= 0.5, z_sum / alpha.clamp(min=1e-3), torch.zeros_like(alpha))

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]

        stem = os.path.splitext(view.image_name)[0]
        torchvision.utils.save_image(rendering, os.path.join(render_path, stem + ".png"))
        Image.fromarray((expected_depth * 1000).round().clamp(0, 65535).cpu().numpy().astype(np.uint16)).save(os.path.join(depth_path, stem + ".png"))
        Image.fromarray((alpha * 255).round().clamp(0, 255).cpu().numpy().astype(np.uint8)).save(os.path.join(alpha_path, stem + ".png"))
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, load_train=not skip_train)  # ADAM CHANGE

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.train_test_exp, separate_sh)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE)