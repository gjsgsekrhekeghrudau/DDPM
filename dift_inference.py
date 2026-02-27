from models import attention_unet, diffusion, classifier
from datasets_.val_dataset import ImageDataset

import torch
from torch.utils.data import DataLoader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPOCH = 83
MODEL_PATH = f'trained_models/ddpm_epoch{EPOCH}.pth'
CLASSIFIER_PATH = f'classifiers/on_epoch_{EPOCH}.pth'

IMG_SIZE = 64
TIME_DIM = 256

DIFFUSION_STEPS = 1000


@torch.no_grad()
def accuracy(classifier, dataloader, diffusion):
    classifier.eval()

    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        t = torch.ones(len(dataset), device=DEVICE).long() * 20
        noisy, _ = diffusion.noise_images(imgs, t)

        logits = classifier(noisy, t)
        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return correct / total


dataset = ImageDataset(IMG_SIZE, num_classes=10)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, drop_last=True)

model = attention_unet.UNet(time_dim=TIME_DIM).to(DEVICE)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state'])

diff = diffusion.Diffusion(DIFFUSION_STEPS, IMG_SIZE, DEVICE)

classifier = classifier.Classifier(model, num_classes=len(dataset.classes)).to(DEVICE)
checkpoint = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
classifier.load_state_dict(checkpoint['model_state'])
classifier.eval()

val_acc = accuracy(classifier, dataloader, diff)
print(f'Validation accuracy: {val_acc:.4f}')
