import torch
from torch import nn
from torch.nn.functional import l1_loss
from src.transforms import MelSpectrogram


class GANLoss(nn.Module):
    def __init__(self, is_discriminator=True):
        super().__init__()
        self.is_discriminator = is_discriminator

    def forward(self, preds, targets):
        if self.is_discriminator:
            real_loss = torch.mean((targets - 1) ** 2)
            fake_loss = torch.mean(preds ** 2)
            return real_loss + fake_loss
        else:
            return torch.mean((preds - 1) ** 2)


class MelSpectrogramLoss(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.mel_spectrogram = MelSpectrogram(config).to(device)

    def forward(self, output_audio, target_spectrogram):
        generated_mel = self.mel_spectrogram(output_audio)
        return l1_loss(generated_mel, target_spectrogram)


class FeatureMatchingLoss(nn.Module):
    def forward(self, preds, targets):
        return sum(
            l1_loss(p, t) for pred, target in zip(preds, targets) for p, t in zip(pred, target)
        )


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.gan_loss = GANLoss(is_discriminator=True)

    def forward(self, scale_disc_pred, scale_disc_gt, period_disc_pred, period_disc_gt, **batch):
        msd_loss = sum(self.gan_loss(pred, target) for pred, target in zip(scale_disc_pred[-1], scale_disc_gt[-1]))
        mpd_loss = sum(self.gan_loss(pred, target) for pred, target in zip(period_disc_pred[-1], period_disc_gt[-1]))
        return {"disc_loss": msd_loss + mpd_loss}


class GeneratorLoss(nn.Module):
    def __init__(self, config, device="cuda", lambda_mel: float = 45.0, lambda_fm: float = 2.0):
        super().__init__()
        self.device = device
        self.gan_loss = GANLoss(is_discriminator=False)
        self.mel_loss_fn = MelSpectrogramLoss(config=config, device=device)
        self.feature_matching_loss = FeatureMatchingLoss()
        self.lambda_mel = lambda_mel
        self.lambda_fm = lambda_fm

    def forward(self, scale_disc_pred, scale_disc_gt, period_disc_pred, period_disc_gt, output_audio, spectrogram, **batch):
        mel_loss = self.mel_loss_fn(output_audio, spectrogram.to(self.device))
        feature_loss = (
            self.feature_matching_loss(scale_disc_pred[:-1], scale_disc_gt[:-1]) +
            self.feature_matching_loss(period_disc_pred[:-1], period_disc_gt[:-1])
        )
        adversarial_loss = (
            sum(self.gan_loss(pred, torch.ones_like(pred)) for pred in scale_disc_pred[-1]) +
            sum(self.gan_loss(pred, torch.ones_like(pred)) for pred in period_disc_pred[-1])
        )

        total_loss = adversarial_loss + self.lambda_fm * feature_loss + self.lambda_mel * mel_loss
        return {"gen_loss": total_loss}
