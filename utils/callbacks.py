from candle.callbacks import Callback
import os
import torch
import matplotlib.pyplot as plt


class ImageSaver(Callback):
    def __init__(self, L, real_ab, trnsformer, save_path, show=False):
        super().__init__()
        self.L = L
        self.real_ab = real_ab
        self.save_path = lambda idx: os.path.join(save_path, f"generated_{idx}.jpg")
        self.trf = trnsformer
        self.show = show

    @torch.no_grad()
    def save_generated(self, current_epoch, L, real_ab, nmax=4):
        L, real_ab = self.to_device(L), self.to_device(real_ab)
        save_path = self.save_path(current_epoch)
        fake_ab = self.model.gen(L)
        nmax = min(nmax, L.size(0))
        real_images = self.trf.inverse_transform_batch(L, real_ab, rtype="np")
        fake_images = self.trf.inverse_transform_batch(L, fake_ab, rtype="np")

        fig, axes = plt.subplots(3, nmax, figsize=(15, 8))
        fig.suptitle(f"Generated Images at Epoch {current_epoch}", fontsize=16)

        for i in range(nmax):
            axes[0, i].imshow(L[i][0].cpu(), cmap='gray')
            axes[0, i].axis("off")
            if i == 0:
                axes[0, i].set_title("Input (L)")
            axes[1, i].imshow(fake_images[i])
            axes[1, i].axis("off")
            if i == 0:
                axes[1, i].set_title("Generated (Fake)")
            axes[2, i].imshow(real_images[i])
            axes[2, i].axis("off")
            if i == 0:
                axes[2, i].set_title("Real (AB)")
        fig.savefig(save_path, bbox_inches='tight')
        if self.show:
            plt.show()
        plt.close(fig)

    def on_epoch_end(self):
        self.save_generated(self.trainer.current_epoch, self.L, self.real_ab)


class CheckpointSaver(Callback):
    def __init__(self, save_path, save_interval):
        super().__init__()
        self.save_path = lambda epoch: os.path.join(save_path, f"epoch_{epoch}.pt")
        self.save_interval = save_interval

    def on_epoch_end(self):
        if self.trainer.current_epoch % self.save_interval == 0:
            self.trainer.save_progress(self.save_path(self.trainer.current_epoch))
