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

import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, read_completion
from utils.general_utils import safe_state, get_expon_lr_func
from utils.shape_loss import ShapePrior, LOSSES, target_mask, seed_colour, object_fit  # ADAM CHANGE
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    shape_prior = ShapePrior(dataset.shape, dataset.shape_radius, dataset.shape_refresh, ownership=dataset.shape_ownership,
                             radius_scale=dataset.shape_radius_scale) if dataset.shape else None  # ADAM CHANGE
    if shape_prior is not None and dataset.shape_ownership == "unseen":  # ADAM CHANGE: the completion samples are the unseen surface
        if not dataset.completion:
            sys.exit("shape_ownership unseen needs --completion")
        cloud, objects = read_completion(dataset.completion)
        names = json.load(open(Path(dataset.completion).with_suffix(".json")))["objects"]
        shape_prior.set_seeds(torch.tensor(np.asarray(cloud.points)).float().cuda(), torch.tensor(objects).cuda(), names)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint, weights_only=False)
        gaussians.restore(model_params, opt)
    observed = None
    if dataset.object_pose == "rigid":  # ADAM CHANGE: completion Gaussians as rigid bodies, their pose trained
        if shape_prior is None or not dataset.completion:
            sys.exit("object_pose rigid needs --shape and --completion")
        cloud, objects, seen = read_completion(dataset.completion, observed=True)
        names = json.load(open(Path(dataset.completion).with_suffix(".json")))["objects"]
        index = torch.tensor([shape_prior.labels.index(n) if n in shape_prior.labels else -1 for n in names], device="cuda")
        if gaussians.object_pose is None:  # fresh run; a checkpoint restores its own pose
            object_of = torch.full((gaussians.get_xyz.shape[0],), -1, dtype=torch.long, device="cuda")
            completed = gaussians.completed
            rows = gaussians.completion_index[completed]
            object_of[completed] = index[torch.tensor(objects, device="cuda")[rows]]
            gaussians.enable_object_pose(object_of, opt.object_pose_lr)
            if dataset.object_pose_evidence == "observed":
                if not seen.any():
                    sys.exit("object_pose_evidence observed needs a completion with observed samples (filters=all)")
                evidence = torch.zeros(len(object_of), dtype=torch.bool, device="cuda")
                evidence[completed] = torch.tensor(seen, device="cuda")[rows]
                gaussians.object_evidence = evidence
                print(f"object pose evidence: {int(evidence.sum())} observed seeds of {int(completed.sum())}")
        gaussians.object_pose_render_grad = dataset.object_pose_evidence != "fit"
        observed = [None] * len(shape_prior.labels)
        if dataset.object_observed:
            points, owners = read_completion(dataset.object_observed)
            points = torch.tensor(np.asarray(points.points)).float().cuda()
            owners = index[torch.tensor(owners, device="cuda")]
            observed = [points[owners == k] for k in range(len(shape_prior.labels))]
        pose_log = open(os.path.join(dataset.model_path, "object_pose.jsonl"), "a")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    state = gaussians.training_state  # ADAM CHANGE: resume the view order, RNG streams and shape ownership
    if state:
        cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
        viewpoint_stack = [cameras[name] for name in state["remaining_views"]]
        viewpoint_indices = state["viewpoint_indices"]
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
        if shape_prior is not None and "shape_labels" in state:
            if shape_prior.labels != state["shape_labels"]:
                raise ValueError("Checkpoint shape labels disagree with the active prior")
            owner = state["shape_owner"]
            shape_prior.owner = owner.to(gaussians.get_xyz.device) if owner is not None else None
            shape_prior.last_refresh = state["shape_last_refresh"]
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        # ADAM CHANGE: the object pose: the prior's grids follow the objects (detached: the surface loss does not
        # pull the pose), the observed points supply the pose's gradient through the fit term
        fit_term = None
        if observed is not None:
            R, t, c = gaussians.object_transforms()
            shape_prior.pose = (R.detach(), t.detach(), c)
            fit_raw, n_fit, fit_per_object = object_fit(shape_prior, observed, R, t, c, dataset.object_fit_samples, dataset.object_fit_scale)
            fit_term = dataset.object_fit_weight * fit_raw
            q = torch.nn.functional.normalize(gaussians.object_pose[:, :4], dim=1)
            angle = 2 * torch.asin(q[:, 1:].norm(dim=1).clamp(max=1))  # [K] rotation angle [rad]
            if dataset.object_pose_prior_t > 0:  # the pose stays near the placement unless the evidence insists
                fit_term = fit_term + 0.5 * (t.norm(dim=1) ** 2).sum() / dataset.object_pose_prior_t ** 2
            if dataset.object_pose_prior_deg > 0:
                fit_term = fit_term + 0.5 * (angle ** 2).sum() / (dataset.object_pose_prior_deg * np.pi / 180) ** 2
            render_loss = loss
            loss = loss + fit_term

        # ADAM CHANGE: shape loss on the Gaussians owned by a prior object (utils/shape_loss.py)
        shape_term, colour_term = None, None
        if shape_prior is not None and opt.shape_weight > 0:
            owner = shape_prior.maybe_refresh(gaussians.get_xyz, iteration)
            owner = torch.where(target_mask(gaussians, dataset.shape_targets), owner, torch.full_like(owner, -1))
            shape_raw, n_shape = LOSSES[dataset.shape_loss](shape_prior, gaussians.get_xyz, gaussians.get_scaling, gaussians.get_rotation, owner,
                                                            opacity=gaussians.get_opacity, opacity_weight=dataset.shape_opacity,
                                                            spread=dataset.shape_spread, tolerance=dataset.shape_tolerance,
                                                            delta=dataset.shape_delta, samples=dataset.shape_samples,
                                                            robust=dataset.shape_robust, robust_scale=dataset.shape_robust_scale)
            shape_term = opt.shape_weight * shape_raw
            if dataset.shape_colour_weight > 0:  # ADAM CHANGE: seed colour loss on the completion Gaussians' DC
                colour_raw, n_colour = seed_colour(shape_prior, gaussians.get_xyz, gaussians._features_dc, shape_prior.owner,
                                                   gaussians.completed, radius=dataset.shape_colour_radius)
                colour_term = dataset.shape_colour_weight * colour_raw
            if iteration % 1000 == 0:
                print(f"\n[ITER {iteration}] shape loss {shape_raw.item():.6f} on {n_shape} Gaussians; owned: "
                      + ", ".join(f"{k} {v}" for k, v in shape_prior.stats().items())
                      + (f"; colour loss {colour_raw.item():.6f} on {n_colour} seeds" if colour_term is not None else ""))

        if observed is not None:  # ADAM CHANGE: pose log, after every term exists
            if iteration % 100 == 0:
                entry = dict(
                    iteration=iteration, fit=float(fit_raw), objects=n_fit,
                    fit_per_object={int(k): float(v) for k, v in fit_per_object.items()},
                    translation_cm=(t.detach().norm(dim=1) * 100).tolist(),
                    rotation_deg=(angle.detach() * 180 / np.pi).tolist())
                if iteration == 100 or iteration % 1000 == 0:  # gradient paths: which loss reaches the pose, per term
                    grads = {}
                    for name, term in (("render", render_loss), ("fit", fit_term), ("surface", shape_term)):
                        if term is None:
                            continue
                        g = torch.autograd.grad(term, gaussians.object_pose, retain_graph=True, allow_unused=True)[0]
                        grads[name] = None if g is None else g.norm(dim=1).tolist()
                    entry["pose_grad_norms"] = grads
                pose_log.write(json.dumps(entry) + "\n")
                pose_log.flush()

        # ADAM CHANGE: gradient policy for the completion Gaussians (see arguments: completion_gradients)
        if dataset.completion_gradients == "shape_only":
            loss.backward()  # observation losses only; the shape term is a separate graph on the parameters
            for param in (gaussians._xyz, gaussians._features_dc, gaussians._features_rest, gaussians._opacity, gaussians._scaling, gaussians._rotation):
                param.grad[gaussians.completed] = 0
            if shape_term is not None:  # without a shape loss this is the fully frozen arm
                shape_term.backward()
        else:
            if shape_term is not None:
                loss = loss + shape_term
            loss.backward()
            if dataset.completion_gradients == "colour_frozen" and gaussians.completed.any():
                for param in (gaussians._features_dc, gaussians._features_rest):
                    param.grad[gaussians.completed] = 0
        if colour_term is not None:  # ADAM CHANGE: its own graph on the seeds' DC, applied after the gradient policy
            colour_term.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in saving_iterations):  # ADAM CHANGE: after the step, so the saved model matches the checkpoint
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            if (iteration in checkpoint_iterations):  # ADAM CHANGE: resumable checkpoint, written atomically, only the latest kept
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                gaussians.training_state = dict(
                    remaining_views=[camera.image_name for camera in viewpoint_stack],
                    viewpoint_indices=viewpoint_indices, python_rng=random.getstate(),
                    numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all())
                if shape_prior is not None:
                    gaussians.training_state.update(shape_labels=shape_prior.labels, shape_owner=shape_prior.owner,
                                                    shape_last_refresh=shape_prior.last_refresh)
                path = Path(scene.model_path) / f"chkpnt{iteration}.pth"
                temporary = path.with_suffix(".pth.tmp")
                torch.save((gaussians.capture(), iteration), temporary)
                temporary.replace(path)
                for old in path.parent.glob("chkpnt*.pth"):
                    if int(old.stem[6:]) < iteration:
                        old.unlink()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=0)  # ADAM CHANGE: python / numpy / torch seed (safe_state)
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
