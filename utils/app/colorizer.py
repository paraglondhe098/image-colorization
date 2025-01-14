from PIL import Image
from typing import Tuple
import torch
from utils.models import Unet
from torchvision import transforms as tt
from utils.dataset import ImageTransformer
import numpy as np


class ImageColorizer:
    def __init__(self, model_path, device):
        self.model = Unet()
        self.model.load_state_dict(torch.load(model_path, weights_only=False)["gen"]["model"])
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.resize = tt.Resize((256, 256), Image.BICUBIC)
        self.trf = ImageTransformer()

    def colorize(self, original: Image.Image) -> Tuple[Image.Image]:
        img = np.array(self.resize(original.convert("RGB")))
        L, _ = self.trf.transform(img)

        L = L.unsqueeze(0).to(self.device)
        with torch.no_grad():
            fake_ab = self.model(L)
        fake_ab = fake_ab.cpu()
        fake_img = self.trf.inverse_transform(L[0].cpu(), fake_ab[0].cpu())
        fake_img = tt.functional.resize(fake_img, (original.size[1], original.size[0]))
        return original, fake_img
