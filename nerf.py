import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


"""
Written following the work of Mildenhall et al.:
NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
arXiv: 2003.08934
github: https://github.com/bmild/nerf
"""


class NerfModel(nn.Module):
    def __init__(self,
                 embedding_dim_pos=10,
                 embedding_dim_dir=4,
                 hidden_dim=128):
        super(NerfModel, self).__init__()

        self.block1 = nn.Sequential(
                nn.Linear(embedding_dim_pos*6+3, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(), 
                )
        #density estimation
        self.block2 = nn.Sequential(
                nn.Linear(embedding_dim_pos*6+hidden_dim+3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim+1)
                )
        #color estimation
        self.block3 = nn.Sequential(
                nn.Linear(embedding_dim_dir*6+hidden_dim+3, hidden_dim//2),
                nn.ReLU(),
                )
        self.block4 = nn.Sequential(
                nn.Linear(hidden_dim//2, 3),
                nn.Sigmoid(),
                )
        self.embedding_dim_pos = embedding_dim_pos
        self.embedding_dim_dir = embedding_dim_dir
        self.relu = nn.ReLU()

    @staticmethod
    def positional_encoding(x, L):
        out = [x]
        for j in range(L):
            out.append(torch.sin(2**j*x))
            out.append(torch.cos(2**j*x))
        return torch.cat(out, dim=1)

    def forward(self, o, d):
        #emb_x: [batch_size, embedding_dim_pos*6]
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_dir)
        h = self.block1(emb_x)
        tmp = self.block2(torch.cat((h, emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        h = self.block3(torch.cat((h, emb_d), dim=1))
        c = self.block4(h)
        return c, sigma


def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    ones = torch.ones((accumulated_transmittance.shape[0], 1),
                      device=alphas.device)
    return torch.cat((ones, accumulated_transmittance[:, :-1]), dim=-1)


def render_rays(nerf_model, ray_oris, ray_dirs, hn=0, hf=0.5, n_bins=192):
    """
    Parameters:
        nerf_model: model for producing color and density information
        ray_oris: ray origins
        ray_dirs: ray directions
        hn: distance from near plane
        hf: distance from far plane
        n_bins: number of bins for density estimation

    Returns:
        pix_col: pixel color
    """
    device = ray_oris.device

    #generate random points along each ray to sample
    t = torch.linspace(hn, hf, n_bins, device=device).expand(ray_oris.shape[0], n_bins)
    mid = (t[:, :-1] + t[:, 1:]) / 2.
    lower = torch.cat((t[:, :1], mid), -1)
    upper = torch.cat((mid, t[:, -1:]), -1)
    u = torch.rand(t.shape, device=device)
    t = lower + (upper - lower) * u #[batch_size, n_bins]

    delta = torch.cat(
            (t[:, 1:] - t[:, :-1],
             torch.tensor([1e10], device=device).expand(ray_oris.shape[0], 1)),
            -1)

    #compute the position of sample points in 3D space
    x = ray_oris.unsqueeze(1) + t.unsqueeze(2) * ray_dirs.unsqueeze(1)

    #expans the ray_dirs tensor to match the shape of x
    ray_dirs = ray_dirs.expand(n_bins, ray_dirs.shape[0], 3).transpose(0, 1)

    colors, sigma = nerf_model(x.reshape(-1, 3), ray_dirs.reshape(-1, 3))
    colors = colors.reshape(x.shape)
    sigma  = sigma.reshape(x.shape[:-1])

    #compute pixel values as a weighted sum of colors along each ray
    alpha = 1 - torch.exp(-sigma*delta) #[batch_size, n_bins]
    weights = compute_accumulated_transmittance(1-alpha).unsqueeze(2)*alpha.unsqueeze(2)
    c = (weights * colors).sum(dim=1)

    #regularization for white background
    weight_sum = weights.sum(-1).sum(-1)

    pix_col = c + 1 - weight_sum.unsqueeze(-1)

    return pix_col


def train(nerf_model, optimizer, scheduler, data_loader, device='cpu', hn=0, hf=1, epochs=int(1e5), n_bins=192, H=400, W=400):
    """
    Parameters:
        nerf_model: the model that is to be trained
        optimizer: optimizer used for training
        scheduler: learning rate scheduler
        data_loader: object that handles training data
        device: device to be used for training (gpu or cpu)
        hn: distance from near cropping plane
        hf: distance from far cropping plane
        epochs: number of training epochs
        n_bins: number of bins used for density estimation
        H: image height
        W: image width

    Returns:
        training_loss: training loss for each epoch
    """

    training_loss = []
    for ind in tqdm(range(epochs)):
        for batch in data_loader:
            ray_oris = batch[:,  :3].to(device)
            ray_dirs = batch[:, 3:6].to(device)
            ground_truth_px_vals = batch[:, 6:].to(device)

            regenerated_px_vals = render_rays(
                                nerf_model, ray_oris, ray_dirs,
                                hn=hn, hf=hf, n_bins=n_bins)
            loss = ((ground_truth_px_vals - regenerated_px_vals) ** 2).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())
        scheduler.step()

    return training_loss


@torch.no_grad()
def test(hn, hf, dataset, out_dir, device='cpu', chunk_size=10, img_index=0, n_bins=192, H=400, W=400):
    """
    Parameters:
        hn: distance from near plane
        hf: distance from far plane
        dataset: ray origins and directions for generating new views
        out_dir: directory for saving files
        device: device to be used for testing (gpu or cpu)
        chunk_size: separate image into chunks for memory efficiency
        img_index: image index to render
        n_bins: number of bins for density estimation
        H: image height
        W: image width
    """
    ray_oris = dataset[img_index*H*W: (img_index+1)*H*W,  :3]
    ray_dirs = dataset[img_index*H*W: (img_index+1)*H*W, 3:6]

    data = []
    for i in range(int(np.ceil(H/chunk_size))):
        #iterate over chunks
        ray_oris_ = ray_oris[i*W*chunk_size: (i+1)*W*chunk_size].to(device)
        ray_dirs_ = ray_dirs[i*W*chunk_size: (i+1)*W*chunk_size].to(device)
        regenerated_px_vals = render_rays(model, ray_oris_, ray_dirs_, 
                                          hn=hn, hf=hf, n_bins=n_bins)
        data.append(regenerated_px_vals)

    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)
    plt.figure()
    plt.imshow(img)
    plt.savefig(f"novel_views/img_{img_index}_N{hn}_F{hf}.png", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.getcwd(), "novel_views")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    #parameters
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    HIDDEN_DIM = 32 #256 #1st
    HEIGHT = 400
    WIDTH = 400
    BATCH_SIZE = 1024
    NUM_BINS = 48 #192 #2nd
    EPOCHS = 1 #16 #3rd
    NEAR = 2
    FAR = 6

    #load data
    print("Loading datasets ...")
    train_dataset = torch.from_numpy(
                        np.load("training_data.pkl",
                            allow_pickle=True))
    #test_dataset = torch.from_numpy(
    #                np.load("testing_data.pkl",
    #                    allow_pickle=True)[:HEIGHT*WIDTH*20])
    data_loader = DataLoader(train_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=True)

    #set up NN model
    print("Loading neural network ...")
    model = NerfModel(hidden_dim=HIDDEN_DIM).to(DEVICE)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)

    #train model
    print("Commencing training ...")
    loss = train(model, model_optimizer, scheduler, data_loader,
                 epochs=EPOCHS, device=DEVICE, hn=NEAR, hf=FAR,
                 n_bins=NUM_BINS, H=HEIGHT, W=WIDTH)

    plt.plot(loss)
    plt.show()

    #test model
    print("Testing ...")
    for img_index in tqdm(range(5)):
        test(hn=NEAR, hf=FAR, dataset=test_dataset, out_dir=output_dir,
                device=DEVICE, img_index=img_index, n_bins=NUM_BINS,
                H=HEIGHT, W=WIDTH)
