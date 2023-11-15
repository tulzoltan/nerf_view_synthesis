import os
import json
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


"""
Based on the work of Mildenhall et al.:
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
arXiv: 2003.08934
github: https://github.com/bmild/nerf

and modified according to
MiP-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields
arXiv: 2103.13415
github: https://github.com/google/mipnerf/
"""


def conical_frustrum_gaussian(d, t0, t1, radius, diag = True):
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2*mu*hw**2) / (3*mu**2 + hw**2)
    t_var = hw**2/3 - 4*hw**4*(12*mu**2-hw**2) / (15*(3*mu**2+hw**2)**2)
    r_var = radius**2*(mu**2/4 + 5*hw**2/12 - 4*hw**4/(15*(3*mu**2+hw**2)))

    #expectation value
    mean = d[..., None, :] * t_mean[..., None]

    #2D normalization factor for projections
    d_mag_sq = torch.maximum(torch.tensor(1e-10).to(d.device),
                             torch.sum(d**2, axis=-1, keepdims=True))

    if diag:
        #diagonalized projection
        dd = d**2
        Imdd = 1 - dd / d_mag_sq

        #covariance
        long_cov = t_var[..., None] * dd[..., None, :]
        trans_cov = r_var[..., None] * Imdd[..., None, :]

    else:
        #projection matrices
        Id = torch.eye(d.shape[-1])
        dd = d[..., :, None] * d[..., None, :]
        Imdd = Id - d[..., :, None] * (d/d_mag_sq)[..., None, :]

        #covariance
        long_cov  = t_var[..., None, None] *   dd[..., None, :, :]
        trans_cov = r_var[..., None, None] * Imdd[..., None, :, :]

    cov = long_cov + trans_cov

    return mean, cov


def integrated_positional_encoding(x, x_cov, L, diag = True):
    """
    Args:
        x: variables to be encoded
        x_cov: covariance matrices of the variables to be encoded
        L: truncation for encoding
        diag: specifies whether the covariance matrices are in diagonal form
    """
    device = x.device

    if diag:
        scales = torch.tensor([2**i for i in range(L)]).to(device)
        shape = list(x.shape[:-1]) + [-1]
        y = torch.reshape(x[..., None, :] * scales[:, None], shape)
        y_var = torch.reshape(x_cov[..., None, :] * scales[:, None]**2, shape)

    else:
        dims = x.shape[-1]
        basis = [2**i * torch.eye(dims) for i in range(L)]
        basis = torch.cat(basis, dim=1).to(device)
        y = x @ basis
        #get the diagonal of P.T @ cov_mat @ P
        y_var = torch.sum((x_cov @ basis) * basis, dim=-2)

    #expected sine and cosine
    evar = torch.exp(-y_var/2)
    return torch.cat([evar*torch.sin(y), evar*torch.cos(y)], dim=1)


def positional_encoding(x, L):
    """
    Args:
        x: an array of variables to be encoded
        L: truncation of encoding
    """
    out = []
    for j in range(L):
        out.append(torch.sin(2**j*x))
        out.append(torch.cos(2**j*x))
    return torch.cat(out, dim=1)


class MipNerfModel(nn.Module):
    def __init__(self,
                 embedding_dim_pos=10,
                 embedding_dim_dir=4,
                 hidden_dim=128):
        super(MipNerfModel, self).__init__()

        self.block1 = nn.Sequential(
                nn.Linear(embedding_dim_pos*6, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                )

        self.block2 = nn.Sequential(
                nn.Linear(embedding_dim_pos*6+hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim+1)
                )

        self.block3 = nn.Sequential(
                nn.Linear(embedding_dim_dir*6+hidden_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Linear(hidden_dim//2, 3)
                )

        self.density_activation = nn.Softplus()
        self.density_bias = -1.0
        self.color_activation = nn.Sigmoid()
        self.color_padding = 0.001

        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dir = embedding_dim_dir

    def forward(self, x, x_cov, d):
        #apply positional encoding to position and direction vectors
        #emb_x: [batch_size, embedding_dim_pos*6]
        #emb_x = positional_encoding(x, self.embedding_dim_pos)
        emb_x = integrated_positional_encoding(x, x_cov, self.embedding_dim_pos)
        emb_d = positional_encoding(d, self.embedding_dim_dir)

        #density estimation
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h = tmp[:, :-1]
        sigma = self.density_activation(tmp[:, -1] + self.density_bias)

        #color estimation
        c = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.color_activation(c)
        c = c * (1 + 2*self.color_padding) - self.color_padding
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    ones = torch.ones((accumulated_transmittance.shape[0], 1),
                      device=alphas.device)
    return torch.cat((ones, accumulated_transmittance[:, :-1]), dim=-1)


def cast_rays(t_vals, ray_oris, ray_dirs, radius, diag = True):
    """
    Casts a cone-shaped ray. The conical frustrums are modelled with
    gaussian distributions.
    Args:
        t_vals: parameter value along the ray
        ray_oris: ray origin vectors
        ray_dirs: ray direction vectors
        radius: radius of the cone on the pixel plane
        diag: specifies whether the covariance matrix is in diagonal form
    """
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]

    means, covs = conical_frustrum_gaussian(ray_dirs, t0, t1, radius, diag)
    means = ray_oris[..., None, :] + means

    return means, covs


def render_rays(nerf_model, ray_oris, ray_dirs, radius,
                hn=0, hf=1, n_bins=192, diag=True):
    """
    Parameters:
        nerf_model: model for producing color and density information
        ray_oris: ray origins
        ray_dirs: ray directions
        radius: radius of the cones on the camera plane
        hn: distance from near plane
        hf: distance from far plane
        n_bins: number of bins for density estimation
        diag: specifies whether the covariance matrices are in diagonal form

    Returns:
        pix_col: pixel color
    """
    device = ray_oris.device

    #generate random points along each ray to sample
    #COMMENT: n_bins -> n_bins+1 in t def; check how delta is defined in paper
    #tensor([1e10]) might need to be removed
    t = torch.linspace(hn, hf, n_bins+1, device=device).expand(ray_oris.shape[0], n_bins+1)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u #[batch_size, n_bins]

    delta = t[:, 1:] - t[:, :-1]

    #dx = torch.sqrt(torch.sum((ray_dirs[:, :-1, :, :] - ray_dirs[:, 1:, :, :])**2, -1))
    #dx = torch.cat([dx, dx[:, -2:-1, :]], 1)
    #radius = dx[..., None] / np.sqrt(3)

    #compute the position of sample points in 3D space
    x, x_cov = cast_rays(t, ray_oris, ray_dirs, radius, diag=diag)

    #expand the ray_dirs tensor to match the shape of x
    ray_dirs = ray_dirs.expand(n_bins, ray_dirs.shape[0], 3).transpose(0, 1)

    #reshape
    x_cov = x_cov.reshape(-1, 3) if diag else x_cov.reshape(-1, 3, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)

    #generate color and density
    colors, sigma = nerf_model(x.reshape(-1, 3), x_cov, ray_dirs)
    colors = colors.reshape(x.shape)
    sigma  = sigma.reshape(x.shape[:-1])

    #compute pixel values as a weighted sum of colors along each ray
    alpha = 1 - torch.exp(-sigma*delta) #[batch_size, n_bins]
    weights = compute_accumulated_transmittance(1-alpha).unsqueeze(2)*alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)

    #regularization for white background
    weight_sum = weights.sum(-1).sum(-1)

    c = c + 1 - weight_sum.unsqueeze(-1)

    return c


def train(nerf_model, optimizer, scheduler, data_loader, radius, device='cpu', hn=0, hf=1, epochs=1, n_bins=192):
    """
    Parameters:
        nerf_model: NN model to be trained
        optimizer: optimizer used for training
        scheduler: learning rate scheduler
        data_loader: object that handles training data
        radius:
        device: device to be used for training (gpu or cpu)
        hn: distance from near cropping plane
        hf: distance from far cropping plane
        epochs: number of training epochs
        n_bins: number of bins used for density estimation

    Returns:
        training_loss: training loss for each epoch
    """

    training_loss = []
    count = 0
    for _ in tqdm(range(epochs)):
        for batch in tqdm(data_loader):
            ray_oris = batch[:,  :3].to(device)
            ray_dirs = batch[:, 3:6].to(device)
            ground_truth_px_vals = batch[:, 6:].to(device)

            #generate pixels
            regenerated_px_vals = render_rays(
                                nerf_model, ray_oris, ray_dirs, radius,
                                hn=hn, hf=hf, n_bins=n_bins)

            #Loss function
            loss = ((ground_truth_px_vals - regenerated_px_vals) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

            count += 1
            if count == 100:
                plt.plot(training_loss)
                plt.show()
                plt.close()
        scheduler.step()

    return training_loss


def get_path(new_dir: str, root: str = os.getcwd()) -> str:
    new_path = os.path.join(root, new_dir)

    if not os.path.exists(new_path):
        os.mkdir(new_path)

    return new_path


if __name__ == "__main__":

    #set directories
    out_dir = get_path("novel_views")
    wgt_dir = get_path("weights")

    #get metadata
    metadata = json.load(open("metadata.json"))

    #parameters
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    HIDDEN_DIM = 64 #256
    HEIGHT = metadata["height"]
    WIDTH = metadata["width"]
    RADIUS = 1.0 / np.sqrt(3)
    BATCH_SIZE = 1024
    NUM_BINS = 48 #192
    EPOCHS = 4 #16
    NEAR = 0
    FAR = 7

    Qload = False
    save_name = f"mipBASE3_HD{HIDDEN_DIM}_NB{NUM_BINS}_N{NEAR}_F{FAR}"
    load_name = f"mipBASE3_HD{HIDDEN_DIM}_NB{NUM_BINS}_N{NEAR}_F{FAR}"

    #load data
    print("Loading datasets ...")
    train_dataset = torch.from_numpy(
                        np.load("training_data.pkl",
                            allow_pickle=True))
    data_loader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    #set up NN model
    print("Loading neural network ...")
    model = MipNerfModel(hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)

    #load weights if requested
    def load_checkpoint(checkpoint):
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    if Qload:
        load_file = os.path.join(wgt_dir, load_name + ".pth.tar")
        if os.path.exists(load_file):
            load_checkpoint(torch.load(load_file))

    #train model
    print("Commencing training ...")
    loss = train(model, optimizer, scheduler, data_loader, radius=RADIUS,
                 epochs=EPOCHS, device=DEVICE, hn=NEAR, hf=FAR,
                 n_bins=NUM_BINS)

    #save progress
    save_file = os.path.join(wgt_dir, save_name + ".pth.tar")
    checkpoint = {"state_dict": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict()}
    torch.save(checkpoint, save_file)

    plt.figure()
    plt.plot(loss)
    if Qload:
        plt.title(f"Loss in {EPOCHS} epochs")
    elif EPOCHS > 1:
        plt.title(f"Loss in first {EPOCHS} epochs")
    else:
        plt.title(f"Loss in first epoch")
    fig_name = os.path.join(out_dir, save_name + (f"loss_EP{EPOCHS}"))
    plt.savefig(fig_name, bbox_inches='tight')
    plt.close()

