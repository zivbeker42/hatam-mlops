import librosa
import numpy as np
import torch
from torch import nn


class LogmelFilterBank(nn.Module):
    def __init__(self, sr=8000, n_fft=256, n_mels=64, fmin=50, fmax=4000, ref=1, amin=1e-10,
                 top_db=None, freeze_parameters=True):
        """
        Calculate logmel spectrogram. The mel filter bank is using librosa.filters.mel
        """
        super(LogmelFilterBank, self).__init__()

        self.ref = ref
        self.amin = amin
        self.top_db = top_db
        if fmax is None:
            fmax = sr // 2

        # (n_fft // 2 + 1, mel_bins)
        self.melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.melW = nn.Parameter(torch.Tensor(self.melW))

    def forward(self, spectrogram, is_log: bool = True):
        r"""Calculate (log) mel spectrogram from spectrogram.

        Args:
            :param spectrogram: (*, n_fft)
            :param is_log: (bool)

        Returns:
            output: (*, mel_bins), (log) mel spectrogram
        """

        # Mel spectrogram (*, mel_bins)
        mel_spectrogram = torch.matmul(spectrogram, self.melW)

        # Logmel spectrogram
        if is_log:
            return self.power_to_db(mel_spectrogram)

        return mel_spectrogram

    def power_to_db(self, input):
        """
        Power to db, this function is the pytorch implementation of librosa.power_to_lb
        """
        ref_value = self.ref
        log_spec = 10.0 * torch.log10(torch.clamp(input, min=self.amin, max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(self.amin, ref_value))

        if self.top_db is not None:
            if self.top_db < 0:
                raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            if torch.any(log_spec.max() - log_spec > self.top_db):
                cut_off = log_spec.max() - self.top_db
                log_spec[log_spec < cut_off] = cut_off

        return log_spec
