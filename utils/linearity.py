from models import attention_unet, diffusion
from datasets_.train_dataset import ImageDataset

import torch
from torch.utils.data import DataLoader


def LinearityScore(a, b, x1, x2, t, noise):
    lin_comb, _ = diff.noise_images(a * x1 + b * x2, t, noise)
    x1_, _ = diff.noise_images(x1, t, noise)
    x2_, _ = diff.noise_images(x2, t, noise)

    y1 = model(lin_comb, t)
    y2 = a * model(x1_, t) + b * model(x2_, t)

    y1_norm = y1 / (torch.linalg.vector_norm(y1, dim=(1, 2, 3), keepdim=True) + 1e-8)
    y2_norm = y2 / (torch.linalg.vector_norm(y2, dim=(1, 2, 3), keepdim=True) + 1e-8)

    cosine_sim = torch.abs(torch.sum(y1_norm * y2_norm, dim=(1, 2, 3)))
    ls = cosine_sim.mean()
    return ls.item()


torch.manual_seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

IMG_DIR = '../../ImageNetDiffusion/datasets_/archive'
IMG_SIZE = 64
TIME_DIM = 256
DIFFUSION_STEPS = 1000
BATCH_SIZE = 100

noise = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)

dataset = ImageDataset(IMG_SIZE, 100, IMG_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=True, drop_last=True)

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

for epoch in torch.arange(1, 100, 2):
    MODEL_PATH = f'../trained_models/ddpm_epoch{epoch}.pth'
    print(MODEL_PATH + ':', end=' ')

    model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])

    t = torch.randint(250, 750, (BATCH_SIZE,))
    a = torch.randn(BATCH_SIZE, 1, 1, 1)
    b = torch.randn(BATCH_SIZE, 1, 1, 1)

    lss = []
    for i, (imgs, _) in enumerate(dataloader):
        if i >= 50:
            break
        x1 = imgs[:BATCH_SIZE]
        x2 = imgs[BATCH_SIZE:]

        ls = LinearityScore(a, b, x1, x2, t, noise)
        lss.append(ls)

    print(f'LS = {sum(lss) / len(lss):.4f}')
