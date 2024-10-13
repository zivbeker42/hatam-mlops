import csv

import torch
import torch.nn.functional as F
from torch import nn
from torchaudio.transforms import Resample

from .conv_block import ConvBlock, init_layer, init_bn
from .logmel import LogmelFilterBank
from .panns_input import PannsInput
from .spectrogram import Spectrogram


class Cnn14_8k(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num,
                 class_label_indices_file_path: str):
        super(Cnn14_8k, self).__init__()

        assert sample_rate == 8000
        assert window_size == 256
        assert hop_size == 80
        assert mel_bins == 64
        assert fmin == 50
        assert fmax == 4000

        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = 80.0

        self.sample_rate = sample_rate

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window=window,
                                                 center=center, pad_mode=pad_mode)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size,
                                                 n_mels=mel_bins, fmin=fmin, fmax=fmax,
                                                 ref=ref, amin=amin, top_db=top_db)

        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)

        self.init_weight()

        self.class_label_indices = {}
        with open(class_label_indices_file_path, 'r') as file:
            for row in csv.DictReader(file):
                class_index = row.get('index')
                class_label = row.get('display_name')
                self.class_label_indices[int(class_index)] = class_label

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc_audioset)

    def forward(self, p_input: PannsInput):
        resampled_audio = []

        for audio, sample_rate in zip(p_input.audios, p_input.sample_rates):
            resampler = Resample(sample_rate, self.sample_rate)
            resampled_audio.append(resampler(audio))

        x = torch.concat(resampled_audio, dim=0)

        x = self.spectrogram_extractor.forward(x)  # (batch_size, 1, time_steps, freq_bins)
        x = self.logmel_extractor.forward(x)  # (batch_size, 1, time_steps, mel_bins)

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))

        embedding = F.dropout(x, p=0.5, training=self.training)
        panns_output = torch.sigmoid(self.fc_audioset(x))

        class_labels = []
        for i, top_k in enumerate(p_input.top_ks):
            panns_values, label_indices = torch.sort(panns_output[i], descending=True)
            class_labels_dict = {}
            for i in range(top_k):
                label_index, panns_value = int(label_indices[i]), float(panns_values[i])
                class_label = self.class_label_indices[int(label_index)]
                class_labels_dict[class_label] = float(panns_value)
            class_labels.append(class_labels_dict)

        output_dict = {'panns_output': panns_output, 'embedding': embedding,
                       'class_labels': class_labels}
        return output_dict
