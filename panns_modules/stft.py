import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFTBase(nn.Module):
    def __init__(self):
        r"""Base class for DFT and IDFT matrix.
        """
        super(DFTBase, self).__init__()

    def dft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(-2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W

    def idft_matrix(self, n):
        (x, y) = np.meshgrid(np.arange(n), np.arange(n))
        omega = np.exp(2 * np.pi * 1j / n)
        W = np.power(omega, x * y)  # shape: (n, n)
        return W


class STFT(DFTBase):
    def __init__(self, n_fft=2048, hop_length=None, win_length=None,
                 window='hann', center=True, pad_mode='reflect', freeze_parameters=True):
        r"""PyTorch implementation of STFT with Conv1d. The function has the
        same output as librosa.stft.

        Args:
            n_fft: int, fft window size, e.g., 2048
            hop_length: int, hop length samples, e.g., 441
            win_length: int, window length e.g., 2048
            window: str, window function name, e.g., 'hann'
            center: bool
            pad_mode: str, e.g., 'reflect'
            freeze_parameters: bool, set to True to freeze all parameters. Set
                to False to finetune all parameters.
        """
        super(STFT, self).__init__()

        assert pad_mode in ['constant', 'reflect']

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.center = center
        self.pad_mode = pad_mode

        # By default, use the entire frame.
        if self.win_length is None:
            self.win_length = n_fft

        # Set the default hop, if it's not already specified.
        if self.hop_length is None:
            self.hop_length = int(self.win_length // 4)

        self.fft_window = librosa.filters.get_window(window, self.win_length, fftbins=True)

        # DFT & IDFT matrix.
        self.W = self.dft_matrix(n_fft)

        self.out_channels = n_fft // 2 + 1

        self.conv_real = nn.Conv1d(in_channels=1, out_channels=self.out_channels,
                                   kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
                                   groups=1, bias=False)

        self.conv_imag = nn.Conv1d(in_channels=1, out_channels=self.out_channels,
                                   kernel_size=n_fft, stride=self.hop_length, padding=0, dilation=1,
                                   groups=1, bias=False)

        if freeze_parameters:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, input):
        """
        Calculate STFT of batch of signals.

        Args:
            input: (batch_size, data_length), input signals.

        Returns:
            real: (batch_size, 1, time_steps, n_fft // 2 + 1)
            imag: (batch_size, 1, time_steps, n_fft // 2 + 1)
        """

        x = input[:, None, :].numpy()  # (batch_size, channels_num, data_length)

        if self.center:
            x = F.pad(torch.Tensor(x), pad=(self.n_fft // 2, self.n_fft // 2), mode=self.pad_mode)

        # Initialize Conv1d weights.
        conv_weights = np.vstack([self.W[i, :self.out_channels] * self.fft_window[i]
                                  for i in range(self.n_fft)]).T
        real_conv_weights, imag_conv_weights = np.real(conv_weights), np.imag(conv_weights)

        self.conv_real.weight.data = torch.Tensor(real_conv_weights)[:, None, :]
        self.conv_imag.weight.data = torch.Tensor(imag_conv_weights)[:, None, :]
        # (n_fft // 2 + 1, 1, n_fft)

        real = self.conv_real(x)
        imag = self.conv_imag(x)
        # (batch_size, n_fft // 2 + 1, time_steps)

        real = real[:, None, :, :].transpose(2, 3)
        imag = imag[:, None, :, :].transpose(2, 3)
        # (batch_size, 1, time_steps, n_fft // 2 + 1)

        return real, imag
