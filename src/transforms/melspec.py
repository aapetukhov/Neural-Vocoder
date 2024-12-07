import torch
import torchaudio
import torch.nn as nn

PAD_CONST = -11.51292


class MelSpectrogram(nn.Module):
    """
    Wrapper for torchaudio.transforms.MelSpectrogram that allows passing
    all parameters dynamically from a configuration file.
    """
    def __init__(self, config: dict, normalize: bool = False):
        """
        Args:
            config (dict): Dictionary containing all MelSpectrogram parameters.
            normalize (bool): Whether to normalize audio input.
        """
        super().__init__()
        self.normalize = normalize
        self.pad_value = config.get("pad_value", PAD_CONST)
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(**config)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Converts audio waveform to mel-spectrogram.
        """
        if self.normalize:
            audio = audio / torch.abs(audio).max(dim=1, keepdim=True).values
        melspec = self.mel_spectrogram(audio).clamp_(min=1e-5).log_()
        return melspec