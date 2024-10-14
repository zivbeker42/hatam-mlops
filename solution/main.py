import csv
import time

import torch
import torchaudio
from torchaudio.transforms import Resample

from solution.panns_input import PannsInput
from solution.cnn_model import Cnn14_8k

device = "cuda" if torch.cuda.is_available() else "cpu"
model_config = {
    "sample_rate": 8000,
    "window_size": 256,
    "hop_size": 80,
    "mel_bins": 64,
    "fmin": 50,
    "fmax": 4000,
    "classes_num": 527,
}

model = Cnn14_8k(**model_config)

class_label_indices = {}
with open("../panns_utils/class_labels_indices.csv", 'r') as file:
    for row in csv.DictReader(file):
        class_index = row.get('index')
        class_label = row.get('display_name')
        class_label_indices[int(class_index)] = class_label

weights = torch.load("../panns_utils/Cnn14_8k_mAP=0.416.pth", weights_only=False, map_location=device)
model.load_state_dict(weights["model"])
model = model.eval().to(device)

p_input = PannsInput.from_file(r'C:\Users\User\PycharmProjects\mlops\audios2/0125-learn_to_whistle_loud.wav', device=device)

resampler = Resample(p_input.sample_rate, model_config["sample_rate"])
x = resampler.forward(p_input.audio)


with torch.no_grad():
    torch_script_model = torch.jit.trace(model, x)
    torch.jit.save(torch_script_model, 'model.pt')

    torch.onnx.export(model, args=x, f='model.onnx',
                      input_names=['input'],
                      output_names=['panns_output', 'embedding'])

# panns_output, embedding = model(x)
#
# print(panns_output, embedding)
# start_time = time.time()
#
# class_labels = {}
# panns_values, label_indices = torch.sort(panns_output.squeeze(), descending=True)
#
# for i in range(p_input.top_k):
#     label_index, panns_value = int(label_indices[i]), float(panns_values[i])
#     class_label = class_label_indices[int(label_index)]
#     class_labels[class_label] = float(panns_value)
#
# print(f"{time.time() - start_time:.3f}s")
# print(class_labels)


# model.half()
# print("\nHALF parameters dtypes")
# for k, v in model.named_parameters():
#     print(k, v.dtype)
#
# audio_resampled_16 = audio_resampled.half()
# audio_resampled_16 = torch.vstack([audio_resampled_16, audio_resampled_16])

# out_model_16, _ = model(audio_resampled_16)
# print(f"{out_model_16.isnan().any() = }")
# print(f"{out_model_16.isinf().any() = }")

# print(f"{torchscript_out_model.isnan().any() = }")
# print(f"{torchscript_out_model.isinf().any() = }")
#
# diff = torch.abs(torchscript_model(audio_resampled)[1] - model(audio_resampled)[1])
# print(f"{diff.max() = }")

# torch.onnx.export(model, (audio_resampled,), 'model.onnx',
#                   input_names=['input'],
#                   output_names=['output1', 'output2'])

