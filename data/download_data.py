from collections import defaultdict
import os
import torchaudio
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
import torch


def get_top_speakers(min_sample=200, max_speakers=30, scan_limit=50000):
    dataset = load_dataset(
        "mythicinfinity/libritts", "clean", split="train.clean.360", streaming=True
    )

    speaker_counts = defaultdict(int)

    print("\n Counting speakers...")

    for i, item in enumerate(tqdm(dataset)):
        speaker_id = str(item["speaker_id"])
        speaker_counts[speaker_id] += 1

        if i >= scan_limit:
            break

    filtered = [
        (spk, count) for spk, count in speaker_counts.items() if count >= min_sample
    ]

    filtered.sort(key=lambda x: x[1], reverse=True)

    selected = [spk for spk, _ in filtered[:max_speakers]]

    print("\n Selected speakers:")
    for spk, cnt in filtered[:max_speakers]:
        print(f"Speaker {spk}: {cnt} samples")

    return selected


def build_balanced_tts_speech(
    selected_speakers,
    per_speaker_limit=300,
    raw_dir=Path("data/raw_audio"),
    subset="clean",
    min_duration=3,
):

    os.makedirs(raw_dir, exist_ok=True)

    speaker_sample_count = {spk: 0 for spk in selected_speakers}

    total_target = len(selected_speakers) * per_speaker_limit

    print(
        f"\n Target: {len(selected_speakers)} speakers x {per_speaker_limit} samples = {total_target}"
    )

    dataset = load_dataset(
        "mythicinfinity/libritts", subset, split="train.clean.360", streaming=True
    )

    pbar = tqdm(total=total_target, desc="Downloading samples")

    for item in dataset:
        speaker_id = str(item["speaker_id"])

        if speaker_id not in speaker_sample_count:
            continue

        if speaker_sample_count[speaker_id] >= per_speaker_limit:
            continue

        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]

        if len(audio_array) < min_duration * sampling_rate:
            continue

        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

        audio_id = item["id"]
        text = item["text"]

        spk_path = raw_dir / f"speaker_{speaker_id}"
        spk_path.mkdir(parents=True, exist_ok=True)

        wav_path = spk_path / f"{audio_id}.wav"
        txt_path = spk_path / f"{audio_id}.txt"

        torchaudio.save(wav_path, waveform, sampling_rate)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        speaker_sample_count[speaker_id] += 1
        pbar.update(1)

        if all(count >= per_speaker_limit for count in speaker_sample_count.values()):
            print("\n Download complete!")
            break

    pbar.close()

    print("\n Final counts:")
    for spk, count in speaker_sample_count.items():
        print(f"Speaker {spk}: {count} samples")


if __name__ == "__main__":
    selected_speakers = get_top_speakers(min_sample=300, max_speakers=30)
    build_balanced_tts_speech(selected_speakers)
