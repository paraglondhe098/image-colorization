import logging
from abc import abstractmethod
from candle.callbacks import Callback
from candle.trainers.base import TrainerModule
from candle.utils.tracking import Tracker
# from candle.trainers.template import TrainerTemplate, ModelConfig
from torch.amp import autocast
import torch
from typing import Optional, List
import copy
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(self, model, **kwargs):
        self.model = model
        for key, value in kwargs.items():
            setattr(self, key, value)

    def set_requires_grad(self, requires_grad=True):
        for p in self.model.parameters():
            p.requires_grad = requires_grad


class TrainerTemplate(TrainerModule):
    def __init__(self,
                 model: nn.Module,
                 callbacks: Optional[List[Callback]] = None,
                 clear_cuda_cache: bool = True,
                 use_amp: bool = True,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):

        super().__init__(model=model, name="SimpleTrainer", device=(device or torch.device('cpu')), logger=logger)

        self.num_batches = None
        self.batch_size = None
        self._current_epoch = 0

        self.clear_cuda_cache = clear_cuda_cache
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.tracker = self.init_tracker()

        self.STOPPER = False
        self.external_events = set()
        self._best_state_dict = None
        self._final_metrics = {}

        self.std_pos = {'on_train_batch_begin', 'on_train_batch_end', 'on_epoch_begin', 'on_epoch_end',
                        'on_test_batch_begin', 'on_test_batch_end', 'on_predict_batch_begin', 'on_predict_batch_end',
                        'on_train_begin', 'on_train_end', 'on_test_begin', 'on_test_end', 'on_predict_begin',
                        'on_predict_end', 'before_training_starts', 'after_training_ends', 'before_backward_pass'}
        self.callbacks = self.set_callbacks(callbacks or [])

    @abstractmethod
    def init_tracker(self):
        pass

    @abstractmethod
    def training_step(self, inputs, labels):
        pass

    @abstractmethod
    @torch.no_grad()
    def eval_step(self, inputs, labels):
        pass

    @abstractmethod
    @torch.no_grad()
    def prediction_step(self, data):
        return self.model(data)

    @abstractmethod
    def save_progress(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_progress(self, *args, **kwargs):
        pass

    @abstractmethod
    def reset_progress(self):
        self._current_epoch = 0
        self.STOPPER = False
        self._best_state_dict = None
        self._final_metrics = {}
        self.tracker = self.init_tracker()

    def _run_callbacks(self, pos: str) -> List[Optional[str]]:
        return self.callbacks.run_all(pos)

    def train(self, train_loader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        self._run_callbacks(pos="on_train_begin")
        for inputs, labels in self.progress_bar(position='training',
                                                iterable=train_loader,
                                                desc=self.epoch_headline):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self._run_callbacks(pos="on_train_batch_begin")

            self.training_step(inputs, labels)

            self._run_callbacks(pos="on_train_batch_end")
        self._run_callbacks(pos="on_train_end")

    @torch.no_grad()
    def validate(self, val_loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()
        self._run_callbacks(pos="on_test_begin")
        for inputs, labels in self.progress_bar(position='validation', iterable=val_loader, desc="Validation: "):
            self._run_callbacks(pos="on_test_batch_begin")
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.eval_step(inputs, labels)

            self._run_callbacks(pos="on_test_batch_end")
        self._run_callbacks(pos="on_test_end")

    @property
    def current_epoch(self):
        return self._current_epoch

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader,
            epochs: int = 1, epoch_start: int = 0):
        """
        Trains the model for the specified number of epochs.

        Args:
            train_loader (torch.utils.data.DataLoader): DataLoader for training datasets.
            val_loader (torch.utils.data.DataLoader): DataLoader for validation datasets.
            epoch_start (int): from what epoch number we should start
            epochs (int): No. of epochs to run for

        Returns:
            None
        """
        self.reset_progress()
        self.epochs = epochs
        self.num_batches = len(train_loader)
        self.batch_size = train_loader.batch_size
        on_gpu = True if self.device.type == 'cuda' else False

        self._run_callbacks(pos="before_training_starts")
        for self._current_epoch in range(epoch_start, epoch_start + self.epochs):
            self._run_callbacks(pos="on_epoch_begin")

            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.train(train_loader)
            self.validate(val_loader)
            self.tracker.snap_and_reset_all()

            self._run_callbacks(pos="on_epoch_end")

            if self.STOPPER:
                break

        self._run_callbacks(pos="after_training_ends")
        return self.tracker.get_history()

    @property
    def final_metrics_(self):
        return self._final_metrics or self.tracker.get_final_values(self.current_epoch)

    @property
    def best_state_dict_(self):
        return self._best_state_dict or copy.deepcopy(self.model.state_dict())

    def predict(self, data_loader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Predicts outputs for the given DataLoader.

        Args:
            data_loader (torch.utils.data.DataLoader): DataLoader providing input datasets for prediction.

        Returns:
            torch.Tensor: Concatenated model predictions for all input batches.
        """
        self.model.eval()
        self._run_callbacks(pos="on_predict_begin")

        all_predictions = []
        for batch_idx, data in self.progress_bar(position="prediction",
                                                 iterable=enumerate(data_loader),
                                                 desc="Processing"):
            self._run_callbacks(pos="on_predict_batch_begin")
            data = data.to(self.device)
            predictions = self.prediction_step(data)
            all_predictions.append(predictions)
            self._run_callbacks(pos="on_predict_batch_end")

        all_predictions = torch.cat(all_predictions, dim=0)
        self._run_callbacks(pos="on_predict_end")
        return all_predictions


class ColorizationTrainer(TrainerTemplate):
    def __init__(self,
                 model,
                 generator_config,
                 discriminator_config,
                 callbacks,
                 gan_loss,
                 l1_lambda):
        super().__init__(model=model,
                         callbacks=callbacks,
                         clear_cuda_cache=True,
                         use_amp=True,
                         device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                         logger=None)

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
