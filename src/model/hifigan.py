import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F

from src.model.hifigan_modules import MSD, MPD, MRF


class Generator(nn.Module):
    def __init__(self, h_u: int = 512, k_u: list =[16, 16, 4, 4]):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(80, h_u, kernel_size=7, padding=3))
        
        layers = []
        for i, k in enumerate(k_u):
            in_channels = h_u // (2 ** i)
            out_channels = h_u // (2 ** (i + 1))
            layers += [
                nn.LeakyReLU(0.1),
                weight_norm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=k, stride=k//2, padding=k//4)),
                MRF(out_channels)
            ]
        self.layers = nn.Sequential(*layers)
        
        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.1),
            weight_norm(nn.Conv1d(h_u // (2 ** len(k_u)), 1, kernel_size=7, padding=3))
        )
        self.tanh = nn.Tanh()

    def forward(self, spectrogram: torch.Tensor, **batch) -> dict:
        x = self.conv1(spectrogram)
        x = self.layers(x)
        x = self.conv2(x)
        x = self.tanh(x)

        return {"output_audio": torch.flatten(x, start_dim=1)}
    
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

    def forward(self, output_audio, audio, detach=False, **batch) -> dict:
        if detach:
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
