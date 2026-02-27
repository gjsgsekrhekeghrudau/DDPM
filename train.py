from models import attention_unet, diffusion
from datasets_.train_dataset import ImageDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOAD_MODEL = False
LAST_EPOCH = 0
MODEL_PATH = 'trained_models'
IMG_DIR = '../ImageNetDiffusion/datasets_/archive'

IMG_SIZE = 64
TIME_DIM = 256
SAVE_EVERY = 100

EPOCHS = 100
LEARNING_RATE = 3e-5
BATCH_SIZE = 8

DIFFUSION_STEPS = 1000

dataset = ImageDataset(IMG_SIZE, 100, IMG_DIR)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
if LOAD_MODEL:
    checkpoint = torch.load(f'{MODEL_PATH}/ddpm_epoch{LAST_EPOCH}.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state'])

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
mse = nn.MSELoss()

for epoch in range(LAST_EPOCH + 1, EPOCHS):
    model.train()
    progress = tqdm(dataloader, desc=f'Epoch {epoch}')

    for i, (imgs, _) in enumerate(progress):
        imgs = imgs.to(DEVICE)
        t = torch.randint(0, DIFFUSION_STEPS, (imgs.size(0),), device=DEVICE)

        noisy, noise = diff.noise_images(imgs, t)
        pred = model(noisy, t)

        loss = mse(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.set_postfix({'loss': loss.item()})

        if SAVE_EVERY > 0 and i > 0 and i % SAVE_EVERY == 0:
            torch.save({'model_state': model.state_dict()}, f'{MODEL_PATH}/ddpm_epoch{epoch}.pth')
