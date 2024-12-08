import torch
from torch import nn
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super().__init__()
        self.periods = periods
        self.discriminators = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    spectral_norm(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1), padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1), stride=1, padding=(2, 0))),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    nn.Conv2d(1024, 1, kernel_size=(3, 1), stride=1, padding=(1, 0))
                )
            ]) for _ in periods
        ])

    def forward(self, input_audio):
        features = []
        for period, layer in zip(self.periods, self.discriminators):
            pad_size = (period - input_audio.size(1) % period) % period
            if pad_size > 0:
                padded_input = F.pad(input_audio, (0, pad_size), mode="reflect")
            else:
                padded_input = input_audio

            x = padded_input.view(input_audio.size(0), 1, -1, period)

            res = []
            for conv in layer:
                x = conv(x)
                res.append(x)

            features.append(res)

        return features



class MSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    weight_norm(nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(16, 64, kernel_size=41, stride=4, padding=20, groups=4)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(64, 256, kernel_size=41, stride=4, padding=20, groups=16)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(256, 1024, kernel_size=41, stride=4, padding=20, groups=64)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=41, stride=4, padding=20, groups=256)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1024, kernel_size=5, stride=1, padding=2)),
                    nn.LeakyReLU(0.1)
                ),
                nn.Sequential(
                    weight_norm(nn.Conv1d(1024, 1, kernel_size=3, stride=1, padding=2))
                )
            ]) for _ in range(3)
        ])
        self.pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(2)
        ])

    def forward(self, audio):
        # res = []
        # x = audio.unsqueeze(1)  # 1D -> 2D: [B, C, T]
        # for i, disc in enumerate(self.discriminators):
        #     if i > 0:
        #         x = self.pools[i - 1](x)
        #     ones = []
        #     for layer in disc:
        #         x = layer(x)
        #         ones.append(x)
        #     res.append(ones)
        # return res
    
        res = []
        for i, discriminator in enumerate(self.discriminators):
            if i == 0:
                x = audio
            else:
                x = self.pools[i - 1](audio)

            x = x.unsqueeze(1)
            ones = []
            for layer in discriminator:
                x = layer(x)
                ones.append(x)
            
            res.append(ones)
        
        return res


class MRF(nn.Module):
    def __init__(self, channels, kernel_sizes=[3, 7, 11], dilations=[[1, 3, 5]] * 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                *[
                    nn.Sequential(
                        nn.LeakyReLU(negative_slope=0.1),
                        weight_norm(nn.Conv1d(channels, channels, kernel_size=k, dilation=d, padding="same"))
                    ) for d in ds
                ]
            ) for k, ds in zip(kernel_sizes, dilations)
        ])

    def forward(self, x):
        res = 0
        for block in self.blocks:
            res = res + x + block(x)

        return res / len(self.blocks)

