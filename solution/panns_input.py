import torch
import torchaudio


class PannsInput:
    def __init__(self, audio: torch.Tensor, sample_rate: int, top_k: int = 10):
        self.audio = audio
        self.sample_rate = sample_rate
        self.top_k = top_k

    @staticmethod
    def from_file(audio_file_path: str, device: str = 'cuda',
                  top_k: int = 10) -> 'PannsInput':
        audio, sample_rate = torchaudio.load(audio_file_path)
        return PannsInput(audio.to(device), sample_rate, top_k)
