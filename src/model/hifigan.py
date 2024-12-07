import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F

from src.model.hifigan_modules import MSD, MPD, MRF


class Generator(nn.Module):
    def __init__(self, channels=512, upsample_rates=[16, 16, 4, 4]):
        super().__init__()
        self.input_conv = weight_norm(nn.Conv1d(80, channels, kernel_size=7, padding=3))
        self.upsampling_blocks = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                weight_norm(nn.ConvTranspose1d(
                    channels // (2 ** i), channels // (2 ** (i + 1)),
                    kernel_size=r, stride=r // 2, padding=r // 4
                )),
                MRF(channels // (2 ** (i + 1)))
            ) for i, r in enumerate(upsample_rates)
        ])
        self.output_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(channels // (2 ** len(upsample_rates)), 1, kernel_size=7, padding=3)),
            nn.Tanh()
        )

    def forward(self, spectrogram):
        x = self.input_conv(spectrogram)
        for block in self.upsampling_blocks:
            x = block(x)
        return x.flatten(1)
    
    def __str__(self):
        """
        Return model details including parameter counts.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters: {all_params}"
        result_str += f"\nTrainable parameters: {trainable_params}"
        return result_str


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.preiod_disc = MPD()
        self.scale_dics = MSD()

    def forward(self, output_audio, audio, detach_generated=False, **batch) -> dict:
        if detach_generated:
            output_audio = output_audio.detach()
        return {
            "scale_disc_gt": self.scale_dics(audio),
            "period_disc_gt": self.preiod_disc(audio),
            "scale_disc_pred": self.scale_dics(output_audio),
            "period_disc_pred": self.preiod_disc(output_audio),
        }
    
    def __str__(self):
        """
        Return model details including parameter counts.
        """
        all_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        result_str = super().__str__()
        result_str += f"\nAll parameters: {all_params}"
        result_str += f"\nTrainable parameters: {trainable_params}"
        return result_str
