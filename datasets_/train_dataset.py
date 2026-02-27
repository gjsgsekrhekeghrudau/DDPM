from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ImageDataset(Dataset):
    def __init__(self, img_size, num_classes, dataset_path):
        self.roots = [f'{dataset_path}/train.X{i}' for i in range(1, 5)]
        self.num_classes = num_classes
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        self._build_index()

    def _build_index(self):
        class_names = set()
        for root in self.roots:
            for name in os.listdir(root):
                path = os.path.join(root, name)
                if os.path.isdir(path):
                    class_names.add(name)

        self.classes = sorted(class_names)[:self.num_classes]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        for root in self.roots:
            for cls in os.listdir(root):
                if cls not in self.class_to_idx:
                    continue

                cls_path = os.path.join(root, cls)
                if not os.path.isdir(cls_path):
                    continue

                label = self.class_to_idx[cls]
                for fname in os.listdir(cls_path):
                    self.samples.append(
                        (os.path.join(cls_path, fname), label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
