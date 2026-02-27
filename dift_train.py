from models import attention_unet, diffusion, classifier
from datasets_.train_dataset import ImageDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 91
MODEL_PATH = f'trained_models/ddpm_epoch{EPOCH}.pth'

IMG_SIZE = 64
TIME_DIM = 256

EPOCHS = 10
LEARNING_RATE = 1e-3
BATCH_SIZE = 64

DIFFUSION_STEPS = 1000

dataset = ImageDataset(IMG_SIZE, num_classes=10, dataset_path='../ImageNetDiffusion/datasets_/archive')
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])
model.eval()

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

classifier = classifier.Classifier(model, num_classes=len(dataset.classes)).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

for epoch in range(EPOCHS):
    total_loss = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        t = torch.ones(BATCH_SIZE, device=DEVICE).long() * 20

        noisy, _ = diff.noise_images(imgs, t)
        pred = classifier(noisy, t)

        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}')

torch.save({'model_state': classifier.state_dict()}, f'classifiers/on_epoch_{EPOCH}.pth')
