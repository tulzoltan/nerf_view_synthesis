import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from process_images import RayMaker
from nerf import NerfModel, render_rays


def create_extrinsic(old_exts, 
                     i: int = 0):
    assert i in range(0, len(old_exts)-1)

    ratio = 0.5
    c1, c2 = ratio, 1 - ratio
    new_ext = np.eye(4)

    #new translation vector
    new_ext[:3, 3] = c1*old_exts[i, :3, 3] + c2*old_exts[i+1, :3, 3]

    #new rotation matrix
    for k in range(3):
        #combine column vectors
        new_ext[:3, k] = c1*old_exts[i, :3, k] + c2*old_exts[i+1, :3, k]
        #orthoginalize
        for l in range(k):
            coef = new_ext[:3, l] @ new_ext[:3, k]
            new_ext[:3, k] -= coef * new_ext[:3, l]
        #normalize
        new_ext[:3, k] /= np.sqrt(np.sum(new_ext[:3, k]**2))

    return new_ext


@torch.no_grad()
def test(model, hn, hf, ray_oris, ray_dirs, H, W, device='cpu', chunk_size=10, n_bins=192):
    """
    Parameters:
        model: trained neural network
        hn: distance from near plane
        hf: distance from far plane
        ray_oris: ray origins for each pixel in the image
        ray_dirs: ray directions for each pixel in the image
        H: image height
        W: image width
        device: device to be used for testing (gpu or cpu)
        chunk_size: separate image into chunks for memory efficiency
        n_bins: number of bins for density estimation
    """
    
    data = []
    for i in range(int(np.ceil(H/chunk_size))):
        #iterate over chunks
        ray_oris_ = ray_oris[i*W*chunk_size: (i+1)*W*chunk_size].to(device)
        ray_dirs_ = ray_dirs[i*W*chunk_size: (i+1)*W*chunk_size].to(device)
        regenerated_px_vals = render_rays(model, ray_oris_, ray_dirs_, 
                                          hn=hn, hf=hf, n_bins=n_bins)
        data.append(regenerated_px_vals)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    return img


class ViewGenerator():
    def __init__(self, width, height, fx, fy, cx, cy):
        intrinsic = np.array([[fx,  0, cx],
                              [ 0, fy, cy],
                              [ 0,  0,  1]])

        self.rays = RayMaker(width=width, height=height,
                             intrinsic=intrinsic)

    def generate(self, ext: np.ndarray):
        print(f"Generating view for extrinsic matrix\n{ext}\n...")

        ray_oris, ray_dirs = self.rays.make(ext)
        ray_oris = torch.tensor(ray_oris.reshape(-1, 3).astype(np.float32))
        ray_dirs = torch.tensor(ray_dirs.reshape(-1, 3).astype(np.float32))

        img = test(model, hn=NEAR, hf=FAR, ray_oris=ray_oris,
                   ray_dirs=ray_dirs, H=HEIGHT, W=WIDTH, 
                   device=DEVICE, n_bins=NUM_BINS)

        return img


if __name__ == "__main__":
    import json

    out_dir = "novel_views"
    wgt_dir = "weights"

    metadata = json.load(open("metadata.json"))

    #parameters
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    HIDDEN_DIM = 64
    HEIGHT = metadata["height"]
    WIDTH = metadata["width"]
    BATCH_SIZE = 1024
    NUM_BINS = 48
    NEAR = 0
    FAR = 7

    load_name = f"BASE3v2_HD{HIDDEN_DIM}_NB{NUM_BINS}_N{NEAR}_F{FAR}"

    #load data
    print("Loading egomotion data ...")
    egomotion = np.load("camera_poses.npy",
                        allow_pickle=True)

    gen = ViewGenerator(width=WIDTH,
                        height=HEIGHT,
                        fx=metadata["focal_x"],
                        fy=metadata["focal_y"],
                        cx=WIDTH/2.0,
                        cy=HEIGHT/2.0)

    #set up NN model
    print("Loading neural network ...")
    model = NerfModel(hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.eval()

    #load weights
    load_file = os.path.join(wgt_dir, load_name+".pth.tar")
    if os.path.exists(load_file):
        model.load_state_dict(torch.load(load_file)["state_dict"])
    else:
        print(f"File {load_file} not found.")
        import sys
        sys.exit()

    #generate novel view
    #ind = np.random.randint(0, len(egomotion)-1)
    ind = 7
    ext = create_extrinsic(egomotion, ind)

    with open("new_camera_poses.npy", "wb") as ncp:
        np.save(ncp, np.expand_dims(ext, axis=0))

    img_new = gen.generate(ext)

    #generate known images for check
    img0 = gen.generate(egomotion[ind])
    img1 = gen.generate(egomotion[ind+1])

    #make plot
    f, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(img0)
    ax[0].set_title(f"training image {ind}")
    ax[1].imshow(img1)
    ax[1].set_title(f"training image {ind+1}")
    ax[2].imshow(img_new)
    ax[2].set_title("generated image")
    plot_name = os.path.join(out_dir, f"newv2TESTIMG_N{NEAR}_F{FAR}.png")
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close()

    print(f"Image saved to file {plot_name}")
