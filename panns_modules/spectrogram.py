from torch import nn

from .stft import STFT


class Spectrogram(nn.Module):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        """Calculate spectrogram. The STFT is implemented with Conv1d.
        The function has the same output of librosa.stft
        """
        super(Spectrogram, self).__init__()

        self.stft = STFT(n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=window, center=center,
                         pad_mode=pad_mode, freeze_parameters=True)

    def forward(self, signal, power: float = 2.0):
        r"""Calculate spectrogram of input signals.
        Args:
            :param signal: signal to calculate spectogram of (batch_size, data_length)
            :param power: raise the spectrogram to the power

        Returns:
            spectrogram: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        (real, imag) = self.stft.forward(signal)
        # (batch_size, n_fft // 2 + 1, time_steps)

        spectrogram = real ** 2 + imag ** 2

        if power != 2.0:
            spectrogram = spectrogram ** (power / 2.0)

        return spectrogram
