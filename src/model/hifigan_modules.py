import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                spectral_norm(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=1, padding=(2, 0))),
                nn.LeakyReLU(0.1),
                nn.Conv2d(512, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
            ) for _ in periods
        ])

    def forward(self, audio):
        features = []
        for period, disc in zip(self.periods, self.discriminators):
            pad_size = (period - audio.size(1) % period) % period
            padded_audio = F.pad(audio, (0, pad_size), mode="reflect")
            reshaped_audio = padded_audio.view(audio.size(0), 1, -1, period)
            features.append([layer(reshaped_audio) for layer in disc])
        return features


class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            nn.Sequential(
                weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                nn.LeakyReLU(0.1),
                weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                nn.LeakyReLU(0.1),
                nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=1)
            ) for _ in range(3)
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(2)
        ])

    def forward(self, audio):
        features = []
        x = audio.unsqueeze(1)
        for i, disc in enumerate(self.discriminators):
            if i > 0:
                x = self.pools[i - 1](x)
            features.append([layer(x) for layer in disc])
        return features


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 7, 11], dilations=[[1, 3, 5]] * 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.LeakyReLU(0.1),
                        weight_norm(nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding=d))
                    ) for d in ds
                ]
            ) for k, ds in zip(kernel_sizes, dilations)
        ])

    def forward(self, x):
        outputs = [block(x) for block in self.blocks]
        return sum(outputs) / len(outputs)

