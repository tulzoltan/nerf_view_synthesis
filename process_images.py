import os
import glob
import json
import pickle

import cv2
import numpy as np

from calibration import Calibration


class RayMaker():
    def __init__(self, width, height, intrinsic):
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

    def make(self, extrinsic):
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]

        #ray directions
        ray_dirs = np.einsum('ij,klj->kli', R,
                             self.ray_dirs_cam)

        #ray origins
        ray_oris = np.broadcast_to(t, ray_dirs.shape)

        return ray_oris, ray_dirs


def load_image(img_name, CamCal):
    #load
    img = cv2.imread(img_name)

    #downsample
    img = DownSampleImage(img, CamCal.RedFac)

    #undistort
    img = CamCal.undistort(img)

    #convert to RGB and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.normalize(img, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return img


def format_pixel_data(CamCal, trajectory, img_dir):
    H, W = CamCal.Size

    #get image files
    images = glob.glob(os.path.join(img_dir, "*.jpg"))

    #Make rays
    rays = RayMaker(width=W, height=H, intrinsic=CamCal.CameraMatrix)

    dataset = np.empty((len(images)*H*W, 9), dtype=np.float32)
    for ind in range(len(images)):
        ext = trajectory[ind]

        #get ray origins and directions
        ray_oris, ray_dirs = rays.make(ext)

        #load image
        img = load_image(img_name, CamCal)

        #combine
        pixels = np.hstack([ray_oris.reshape(-1, 3),
                            ray_dirs.reshape(-1, 3),
                            img.reshape(-1, 3)])

        dataset[ind*H*W: (ind+1)*H*W] = pixels

    output_name = "pixdat.pkl"
    with open(output_name, "wb") as file:
        pickle.dump(dataset, file)

    print(f"images saved to {output_name}")
    print(f"number of pixels: {len(dataset)}")


def test_images(H, W, index=0):
    pixels = np.load("pixdat.pkl", allow_pickle=True)
    img = pixels[index*H*W: (index+1)*H*W]
    cv2.imshow("image", img)
    cv2.waitKey(2000)


if __name__ == "__main__":
    img_dir = os.path.join(os.getcwd(), "images/sequence1")

    #load calibration data
    CamCal = Calibration()
    CamCal.load("calib.pkl")

    #Get egomotion data
    trajectory = np.load("egomotion.npy", allow_pickle=True)

    format_pixel_data(CamCal, trajectory, img_dir)

    test_images(CamCal.Size[0], CamCal.Size[1])
    cv2.destroyAllWindows()
