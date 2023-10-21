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


def create_extrinsic(old_exts: np.array,
                     i: int = 0) -> np.array():
    assert i in range(0, len(old_exts)-1)

    new_ext = np.eye(4)

    #new translation vector
    new_ext[:3, 3] = (old_exts[i, :3, 3] + old_exts[i+1, :3, 3]) / 2

    #new rotation matrix
    for k in range(3):
        #add old eigenvectors
        new_ext[:3, k] = old_exts[i, :3, k] + old_exts[i+1, :3, k]
        #normalize
        new_ext[:3, k] /= np.sqrt(np.sum(new_ext[:3, k]**2))

    return nex_ext


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


if __name__ == "__main__":
    import json

    out_dir = "novel_views"
    wgt_dir = "weights"

    metadata = json.load(open("metadata.json"))

    #parameters
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    HIDDEN_DIM = 32
    HEIGHT = metadata["height"]
    WIDTH = metadata["width"]
    BATCH_SIZE = 1024
    NUM_BINS = 48
    NEAR = 2
    FAR = 6

    load_name = f"BASE_HD{HIDDEN_DIM}_NB{NUM_BINS}_N{NEAR}_F{FAR}"

    #load data
    print("Loading egomotion data ...")
    egomotion = np.load("egomotion.npy", allow_pickle=True)
    ind = np.random.randint(0, len(egomotion)-1)
    ext = create_extrinsic(egomotion, ind)
    print(ext)

    fx = metadata["focal_x"]
    fy = metadata["focal_y"]
    cx = WIDTH / 2
    cy = HEIGHT / 2
    rays = RayMaker(width=WIDTH,
                    height=HEIGHT,
                    intrinsic=np.array([[fx,  0, cx],
                                        [ 0, fy, cy],
                                        [ 0,  0,  1]]))
    rays.make(ext)

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

    #test model
    print("Testing ...")

    #generate image
    img = test(model, hn=NEAR, hf=FAR, ray_oris=ray_oris, rayw_dirs=ray_dirs, 
                H=HEIGHT, W=WIDTH, device=DEVICE, n_bins=NUM_BINS)


    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax[0].imshow(img)
    ax[0].set_title("original image")
    plot_name = os.path.join(output_dir, f"{pref}_{cname}_IMG{ind}_N{NEAR}_F{FAR}.png")
    plt.savefig(plot_name, bbox_inches="tight")
    plt.close()

