from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tt
import cv2
import numpy as np
import torch


class ImageTransformer:
    def __init__(self, normalization_value=0.5):
        self.normalize = tt.Compose([tt.ToTensor(),
                                     tt.Normalize((normalization_value, normalization_value, normalization_value),
                                                  (normalization_value, normalization_value, normalization_value))])
        self.denormalize = lambda x: x * normalization_value + normalization_value

    def transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img = self.normalize(img)
        L = img[[0], ...]
        ab = img[[1, 2], ...]
        return L, ab

    def inverse_transform(self, L, ab, rtype="PIL"):
        img = torch.cat([L, ab], dim=0)
        img = self.denormalize(img)
        img = (img.permute(1, 2, 0) * 255).detach().cpu().numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        if rtype == "PIL":
            return Image.fromarray(img)
        else:
            return img

    def inverse_transform_batch(self, L_batch, ab_batch, rtype="np"):
        Lab_batch = self.denormalize(torch.cat([L_batch, ab_batch], dim=1)) * 255
        Lab_batch = Lab_batch.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
        rgb_imgs = []
        for img_lab in Lab_batch:
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            rgb_imgs.append(img_rgb if rtype == 'np' else Image.fromarray(img_rgb))
        return np.stack(rgb_imgs, axis=0) if rtype == 'np' else rgb_imgs


class ColorizationDataset(Dataset):
    def __init__(self, image_paths, split, size, preprocessor=None, return_original=False, rtype_original="PIL"):
        self.paths = image_paths
        self.transform = tt.Compose([
            tt.Resize((size, size), Image.BICUBIC),
            tt.RandomHorizontalFlip()
        ]) if split == "train" else tt.Resize((size, size), Image.BICUBIC)
        self.preprocessor = preprocessor or ImageTransformer()
        self.size = size
        self.return_original = return_original
        self.rtype_original = rtype_original

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = np.array(self.transform(img))
        L, ab = self.preprocessor.transform(img)
        if self.return_original:
            return L, ab, (Image.fromarray(img) if self.rtype_original == "PIL" else img)
        return L, ab

    def __len__(self):
        return len(self.paths)
