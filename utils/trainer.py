from candle.utils.tracking import Tracker
from candle.trainers.template import TrainerTemplate
from torch.amp import autocast
import torch.nn.functional as F
import torch


class ColorizationTrainer(TrainerTemplate):
    def __init__(self,
                 model,
                 generator_config,
                 discriminator_config,
                 callbacks,
                 gan_loss,
                 l1_lambda,
                 logger = None):
        super().__init__(model=model,
                         callbacks=callbacks,
                         clear_cuda_cache=True,
                         use_amp=True,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         logger=logger)

        self.gen = generator_config
        self.disc = discriminator_config
        self.gan_loss = gan_loss
        self.l1_lambda = l1_lambda

    def init_tracker(self):
        return Tracker(["gen_loss", "disc_loss", "l1_loss", "val_loss"])

    def train_discriminator(self, L, real_ab):
        # self.disc.model.train()
        self.disc.set_requires_grad(True)
        L, real_ab = self.to_device(L), self.to_device(real_ab)

        self.disc.optimizer.zero_grad()
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            real_images = torch.cat([L, real_ab], dim=1)
            real_predicted = self.disc.model(real_images)
            real_loss = self.gan_loss(real_predicted, is_real=True)

            with torch.no_grad():
                fake_ab = self.gen.model(L)
                fake_images = torch.cat([L, fake_ab], dim=1)
            fake_predicted = self.disc.model(fake_images.detach())
            fake_loss = self.gan_loss(fake_predicted, is_real=False)

            disc_loss = (real_loss + fake_loss) * 0.5

        self.disc.scaler.scale(disc_loss).backward()
        self.disc.scaler.step(self.disc.optimizer)
        self.disc.scaler.update()

        return disc_loss.item()

    def train_generator(self, L, real_ab):
        # self.gen.model.train()
        self.disc.set_requires_grad(False)

        self.gen.optimizer.zero_grad()
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            fake_ab = self.gen.model(L)
            fake_images = torch.cat([L, fake_ab], dim=1)
            fake_predicted = self.disc.model(fake_images)
            gen_loss = self.gan_loss(fake_predicted, is_real=True)
            l1_loss = F.l1_loss(fake_ab, real_ab)
            total_loss = gen_loss + (self.l1_lambda * l1_loss)

        self.gen.scaler.scale(total_loss).backward()
        self.gen.scaler.step(self.gen.optimizer)
        self.gen.scaler.update()

        return gen_loss.item(), l1_loss.item()

    @torch.no_grad()
    def validate_generator(self, L, real_ab):
        # self.gen.model.eval()
        with autocast(device_type=self.device.type, enabled=self.use_amp):
            fake_ab = self.gen.model(L)
            val_loss = F.mse_loss(fake_ab.detach(), real_ab)
        return val_loss.item()

    def training_step(self, L, real_ab):
        disc_loss = self.train_discriminator(L, real_ab)
        gen_loss, l1_loss = self.train_generator(L, real_ab)
        self.tracker.update({"gen_loss": gen_loss, "disc_loss": disc_loss, "l1_loss": l1_loss})

    def eval_step(self, L, real_ab):
        val_loss = self.validate_generator(L, real_ab)
        self.tracker.update({"val_loss": val_loss})

    def prediction_step(self, L):
        fake_ab = self.gen.model(L)
        return torch.cat([L, fake_ab], dim=1)

    def save_progress(self, save_path):
        values = {
            "epoch": self.current_epoch,
            "gen": {
                "model": self.gen.model.state_dict(),
                "optimizer": self.gen.optimizer.state_dict(),
                "scaler": self.gen.scaler.state_dict()
            },
            "disc": {
                "model": self.disc.model.state_dict(),
                "optimizer": self.disc.optimizer.state_dict(),
                "scaler": self.disc.scaler.state_dict()
            },
            "tracker": self.tracker
        }
        torch.save(values, save_path)

    def load_progress(self, save_path):
        values = torch.load(save_path)
        self.gen.model.load_state_dict(values["gen"]["model"])
        self.gen.optimizer.load_state_dict(values["gen"]["optimizer"])
        self.gen.scaler.load_state_dict(values["gen"]["scaler"])

        self.disc.model.load_state_dict(values["disc"]["model"])
        self.disc.optimizer.load_state_dict(values["disc"]["optimizer"])
        self.disc.scaler.load_state_dict(values["disc"]["scaler"])

        self.tracker = values["tracker"]
        self._best_state_dict = self.gen.model.state_dict()
        self._final_metrics = self.tracker.get_final_values(self.current_epoch)
        self._current_epoch = self.tracker.current_epoch

    def reset_progress(self):
        self._current_epoch = 0
        self.STOPPER = False
        self._best_state_dict = None
        self._final_metrics = {}
        self.tracker = self.init_tracker()
