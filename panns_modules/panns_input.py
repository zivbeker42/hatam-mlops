from typing import List

import torch
import torchaudio


class PannsInput:
    def __init__(self, audios: List[torch.Tensor] = None, sample_rates: List[int] = None,
                 top_ks: List[int] = None):
        self.audios = audios or []
        self.sample_rates = sample_rates or []
        self.top_ks = top_ks or []

    def add_audio(self, audio: torch.Tensor, sample_rate: int = 8000,
                  top_k: int = 3) -> 'PannsInput':
        self.audios.append(audio)
        self.sample_rates.append(sample_rate)
        self.top_ks.append(top_k)
        return self

    def add_from_file(self, audio_file_path: str, device: str = 'cuda',
                      top_k: int = 3) -> 'PannsInput':
        audio, sample_rate = torchaudio.load(audio_file_path)
        audio = audio[:1, :].to(device)  # take a single channel
        return self.add_audio(audio, sample_rate, top_k)
