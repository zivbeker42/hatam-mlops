from typing import List

import torch
import torchaudio


class PannsInput:
    def __init__(self, audios: torch.Tensor = None, sample_rates: List[int] = None,
                 top_ks: List[int] = None):
        self.audios = audios or torch.tensor([])
        self.sample_rates = sample_rates or []
        self.top_ks = top_ks or []

    def add_audio(self, audio: torch.Tensor, sample_rate: int = 8000,
                  top_k: int = 10) -> 'PannsInput':
        self.audios = torch.concat([self.audios, audio], dim=0)
        self.sample_rates.append(sample_rate)
        self.sample_rates.append(top_k)
        return self

    def from_file(self, audio_file_path: str, device: str = 'cuda',
                  top_k: int = 10) -> 'PannsInput':
        audio, sample_rate = torchaudio.load(audio_file_path)
        return self.add_audio(audio.to(device), sample_rate, top_k)
