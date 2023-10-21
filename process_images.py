import os
import glob
import json
import pickle

import cv2
import numpy as np
from tqdm import tqdm

from calibration import Calibration, ImageLoader
from odometry import estimate_egomotion
#from visualization import visualize_cameras


class RayMaker():
    def __init__(self, width: int, height: int, intrinsic: np.array):
        #parameters
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        #ray directions in camera coordinate system
        u, v = np.meshgrid(np.arange(width),
                       np.arange(height))
        dx = (u - cx) / fx
        dy = (v - cy) / fy
        dz = np.ones_like(dx)

        #normalize
        dnorm = np.sqrt(dx**2 + dy**2 + dz**2)
        dx /= dnorm
        dy /= dnorm
        dz /= dnorm

        self.ray_dirs_cam = np.dstack([dx, dy, dz])

    def make(self, extrinsic: np.array):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        #ray directions
        ray_dirs = np.einsum('ij,klj->kli', R, self.ray_dirs_cam)

        #ray origins
        ray_oris = np.broadcast_to(t, ray_dirs.shape)

        return ray_oris, ray_dirs


def process(CamCal: Calibration,
            image_directory: str,
            output_name: str):
    #data loader for images
    images = ImageLoader(directory=image_directory,
                         CamCal=CamCal,
                         grayscale=True)

    print("Performing visual odometry to obtain extrinsic matrices...")
    trajectory = estimate_egomotion(CamCal, images)

    images.grayscale=False
    images.normalize=True

    print("Processing images...")
    H, W = CamCal.Size

    #Make rays
    rays = RayMaker(width=W, height=H, intrinsic=CamCal.CameraMatrix)

    dataset = np.empty((len(images)*H*W, 9), dtype=np.float32)
    for ind in tqdm(range(len(images))):
        img = images[ind]
        ext = trajectory[ind]

        #get ray origins and directions
        ray_oris, ray_dirs = rays.make(ext)

        #combine ray and pixel data
        pixels = np.hstack([ray_oris.reshape(-1, 3),
                            ray_dirs.reshape(-1, 3),
                            img.reshape(-1, 3)])

        dataset[ind*H*W: (ind+1)*H*W] = pixels

    with open(output_name, "wb") as datafile:
        pickle.dump(dataset, datafile)

    metadata = {"width": W,
                "height": H,
                "focal_x": CamCal.CameraMatrix[0, 0],
                "focal_y": CamCal.CameraMatrix[1, 1],
                "reduction_factor": CamCal.RedFac,
                "number_of_images": len(images)}

    with open("metadata.json", "w") as metafile:
        json.dump(metadata, metafile)

    print(f"{len(images)} images saved to {output_name}")
    return trajectory


def test_images(file_name, H, W, index=0):
    pixels = np.load(file_name, allow_pickle=True)

    img = pixels[index*H*W: (index+1)*H*W, 6:]
    img = img.reshape((H, W, 3))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    while True:
        cv2.imshow("image", img)
        if cv2.waitKey(33) == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    #load calibration data
    CamCal = Calibration()
    CamCal.load("calib.pkl")

    #process test images
    out_name = "training_data.pkl"
    cam_poses = process(CamCal=CamCal,
                        image_directory="images/sequence2",
                        output_name=out_name)

    with open("camera_poses.npy", "wb") as ego:
        np.save(ego, cam_poses)

    #check processed test images
    test_images(out_name, CamCal.Size[0], CamCal.Size[1], index=2)
