import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self, loss=nn.BCEWithLogitsLoss(), real_label=1.0, fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = loss

    def __call__(self, preds, is_real):
        labels = self.real_label.expand_as(preds) if is_real else self.fake_label.expand_as(preds)
        return self.loss(preds, labels)

