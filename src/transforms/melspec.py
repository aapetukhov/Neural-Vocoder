import torch
import torchaudio
import torch.nn as nn
import librosa


PAD_CONST = -11.51292


class MelSpectrogram(nn.Module):
    def __init__(self, config: dict, normalize: bool = False):
        super().__init__()
        self.normalize = normalize
        self.pad_value = config.get("pad_value", PAD_CONST)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(**config)

        mel_basis = librosa.filters.mel(
            sr=config["sample_rate"],
            n_fft=config["n_fft"],
            n_mels=config["n_mels"],
            fmin=config["f_min"],
            fmax=config["f_max"],
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis, dtype=torch.float32))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            audio = audio / torch.abs(audio).max(dim=1, keepdim=True)[0]
        audio = torch.nn.functional.pad(
            audio,
            (
                (self.mel_spectrogram.win_length - self.mel_spectrogram.hop_length) // 2,
                (self.mel_spectrogram.win_length - self.mel_spectrogram.hop_length) // 2,
            ),
            mode="reflect",
        )
        melspec = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
        return melspec
