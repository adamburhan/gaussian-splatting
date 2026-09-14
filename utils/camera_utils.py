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

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
from PIL import Image
import cv2

WARNED = False

def resample(array, intrinsics, size, interpolation):
    """Resample `array` onto a `size` (w, h) pinhole grid with the camera's own K, undistorting it if
    the camera has OPENCV distortion. The array may have any resolution: both grids are scaled copies
    of the full-resolution camera, and pixel centres sit at integer + 0.5 (COLMAP convention)."""
    if intrinsics is None or intrinsics["distortion"] is None:
        return cv2.resize(array, size, interpolation=interpolation)
    sx, sy = size[0] / intrinsics["width"], size[1] / intrinsics["height"]
    K = np.array([[intrinsics["fx"] * sx, 0, intrinsics["cx"] * sx - 0.5],
                  [0, intrinsics["fy"] * sy, intrinsics["cy"] * sy - 0.5],
                  [0, 0, 1]])
    map_x, map_y = cv2.initUndistortRectifyMap(K, intrinsics["distortion"], None, K, size, cv2.CV_32FC1)
    map_x = (map_x + 0.5) * (array.shape[1] / size[0]) - 0.5
    map_y = (map_y + 0.5) * (array.shape[0] / size[1]) - 0.5
    return cv2.remap(array, map_x, map_y, interpolation)

def loadCam(args, id, cam_info, resolution_scale, is_nerf_synthetic, is_test_dataset):
    image = Image.open(cam_info.image_path)
    if cam_info.intrinsics is not None and cam_info.intrinsics["distortion"] is not None:
        image = Image.fromarray(resample(np.asarray(image), cam_info.intrinsics, image.size, cv2.INTER_LINEAR))
    if cam_info.mask_path != "":
        mask = np.asarray(Image.open(cam_info.mask_path).convert("L"))
        mask = resample(mask, cam_info.intrinsics, image.size, cv2.INTER_NEAREST)
        image.putalpha(Image.fromarray(mask))  # PILtoTorch hands a 4th channel to Camera as alpha_mask

    if cam_info.depth_path != "":
        try:
            if cam_info.depth_unit == "mm":
                depth = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 1000
                invdepthmap = np.where(depth > 0, 1 / np.maximum(depth, 1e-3), 0).astype(np.float32)
            elif is_nerf_synthetic:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / 512
            else:
                invdepthmap = cv2.imread(cam_info.depth_path, -1).astype(np.float32) / float(2**16)

        except FileNotFoundError:
            print(f"Error: The depth file at path '{cam_info.depth_path}' was not found.")
            raise
        except IOError:
            print(f"Error: Unable to open the image file '{cam_info.depth_path}'. It may be corrupted or an unsupported format.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred when trying to read depth at {cam_info.depth_path}: {e}")
            raise
    else:
        invdepthmap = None
        
    orig_w, orig_h = image.size
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution
    

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if invdepthmap is not None and cam_info.depth_unit == "mm":
        # sensor depth (e.g. 256x192 iPhone LiDAR) reaches the training resolution here by nearest
        # neighbour so depth edges are not blended; Camera's linear resize is then the identity
        invdepthmap = resample(invdepthmap, cam_info.intrinsics, resolution, cv2.INTER_NEAREST)

    principal_point = None
    if cam_info.intrinsics is not None:
        principal_point = (cam_info.intrinsics["cx"] / cam_info.intrinsics["width"],
                           cam_info.intrinsics["cy"] / cam_info.intrinsics["height"])

    return Camera(resolution, colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, depth_params=cam_info.depth_params,
                  image=image, invdepthmap=invdepthmap,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  train_test_exp=args.train_test_exp, is_test_dataset=is_test_dataset, is_test_view=cam_info.is_test,
                  principal_point=principal_point)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_nerf_synthetic, is_test_dataset):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, is_nerf_synthetic, is_test_dataset))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry