import os
import torchaudio
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path
import torch


def build_balanced_tts_speech(
    selected_speakers,
    per_speaker_limit=3000,
    raw_dir=Path("data/raw_audio"),
    subset="train-clean-100",
    min_duration=2.0,
):
    os.makedirs(raw_dir, exist_ok=True)
    speaker_sample_count = {spk: 0 for spk in selected_speakers}
    total_target = len(selected_speakers) * per_speaker_limit
    print(
        f"\n Target: {len(selected_speakers)} speakers, {per_speaker_limit} samples each = {total_target} total samples"
    )
    print(f"Total samples :{total_target}\n")
    dataset = load_dataset(
        "mythicinfinity/libritts",
        subset,
        split="train",
        streamin=True,
        trust_remote_code=True,
    )
    pbar = tqdm(total=total_target, desc="Downloading Samples")

    for item in dataset:
        speaker_id = str(item["speaker_id"])
        if speaker_id not in speaker_sample_count:
            continue
        if speaker_sample_count[speaker_id] >= per_speaker_limit:
            continue

        audio_id = item["id"]
        text = item["text"]
        audio_array = item["audio"]["array"]
        sampling_rate = item["audio"]["sampling_rate"]

        if len(audio_array) < min_duration * sampling_rate:
            continue

        waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

        # max_val = waveform.abs().max()
        # if max_val > 0:
        #     waveform = waveform / max_val

        # waveform = waveform.unsqueeze(0)

        spk_path = os.path.join(raw_dir, speaker_id)
        os.makedirs(spk_path, exist_ok=True)

        wav_path = os.path.join(spk_path, f"{audio_id}.wav")
        txt_path = os.path.join(spk_path, f"{audio_id}.txt")

        torchaudio.save(wav_path, waveform, sampling_rate)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        speaker_sample_count[speaker_id] += 1
        pbar.update(1)

        if all(count >= per_speaker_limit for count in speaker_sample_count.values()):
            print("\n All speaker targets reached")
            break

    pbar.close()
    print("\n Download complete")
    for spk, count in speaker_sample_count.items():
        print(f"Speaker {spk}: {count} samples")


if __name__ == "__main__":
    selected_speakers = ["1320", "1284", "3575", "1221"]
    build_balanced_tts_speech(selected_speakers)
