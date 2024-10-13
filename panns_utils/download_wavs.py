import os
import random

import tqdm
from pydub import AudioSegment
from pytubefix import YouTube

current_dir = os.path.dirname(os.path.abspath(__file__))
audios_dir = os.path.join(current_dir, "audios")


def is_english(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


times = [
    2, 2,
    3, 3,
    4, 4, 4, 4, 4,
    5, 5, 5, 5, 5, 5,
    6, 6,
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    8, 8, 8, 8, 8, 8, 8, 8,
    9, 9, 9, 9, 9, 9,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
]

# Read csv
with open(os.path.join(current_dir, "eval_segments.csv"), 'r') as f:
    lines = f.readlines()

lines = list(lines[3:1004])  # Remove csv head info

# Download
for (n, line) in enumerate(tqdm.tqdm(lines)):
    if n >= 1000:
        break

    items = line.split(', ')
    audio_id = items[0]
    start_time = int(float(items[1]) * 1000)
    end_time = int(float(items[2]) * 1000)

    duration = random.choice(times) - random.random()
    start_time += (10 - duration) * 500
    end_time -= (10 - duration) * 500

    try:

        # Download full video of whatever format
        yt = YouTube(f"https://www.youtube.com/watch?v={audio_id}")
        video = yt.streams.filter(only_audio=True).first()
        out_file = video.download(output_path=audios_dir)

        if not is_english(out_file):
            os.remove(out_file)
            continue

        base, ext = os.path.splitext(out_file)
        file_name = os.path.basename(base)
        new_file_name = f"{n:04}-{file_name.replace(' ', '_').lower()}.wav"
        new_file = os.path.join(audios_dir, new_file_name)

        if os.path.isfile(new_file):
            os.remove(out_file)
        else:
            os.rename(out_file, new_file)

        audio_seg = AudioSegment.from_file(new_file)
        cut_audio_seg = audio_seg[start_time:end_time]
        cut_audio_seg.export(new_file, format="wav")

    except Exception as e:
        pass
