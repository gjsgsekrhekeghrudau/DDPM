from models import attention_unet, diffusion

import torch
from torchvision import utils

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'trained_models/ddpm_epoch99.pth'
SAVE_PATH = 'samples/sample.png'
IMG_SIZE = 64
TIME_DIM = 256
DIFFUSION_STEPS = 1000

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

samples = diff.sample(model, 1)
utils.save_image(samples, SAVE_PATH, nrow=1, normalize=True)
