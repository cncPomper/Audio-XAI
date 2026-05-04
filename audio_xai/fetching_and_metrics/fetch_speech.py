import os

import soundfile as sf
from datasets import Audio, load_dataset
from datasets import Dataset as HFDataset

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

N_SAMPLES = 5
VERSION = "v0.02"  # "v0.01" or "v0.02"
SPLIT = "train"  # "train", "validation", "test"
LABEL_FILTER = None  # e.g. "yes", "no", "stop" — None keeps all
SKIP_SILENCE = True
OUTPUT_DIR = "speech_commands_samples"

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading google/speech_commands {VERSION} [{SPLIT}] ...")
    ds = load_dataset(
        "google/speech_commands",
        VERSION,
        split=SPLIT,
        streaming=True,
        trust_remote_code=True,
    ).cast_column("audio", Audio(sampling_rate=16000))

    # Resolve label names from a non-streaming load (streaming datasets lack .features).
    _info_ds = load_dataset("google/speech_commands", VERSION, split=SPLIT, trust_remote_code=True)
    assert isinstance(_info_ds, HFDataset)
    label_names = _info_ds.features["label"].names  # list[str], index == label int

    silence_idx = label_names.index("_silence_") if "_silence_" in label_names else -1

    filter_idx = None
    if LABEL_FILTER is not None:
        name = LABEL_FILTER.lower()
        if name not in label_names:
            raise ValueError(f"Unknown label '{name}'. Available: {label_names}")
        filter_idx = label_names.index(name)

    saved = 0
    for sample in ds:
        if saved >= N_SAMPLES:
            break

        idx = sample["label"]

        if SKIP_SILENCE and idx == silence_idx:
            continue
        if filter_idx is not None and idx != filter_idx:
            continue

        label = label_names[idx]
        speaker_id = sample["speaker_id"] or "none"
        utt_id = sample["utterance_id"]
        filename = f"{label}_{speaker_id}_{utt_id}.wav"

        sf.write(
            os.path.join(OUTPUT_DIR, filename),
            sample["audio"]["array"],
            sample["audio"]["sampling_rate"],
        )
        saved += 1
        print(f"[{saved:>4}/{N_SAMPLES}] {filename}")

    print(f"\nDone. {saved} files saved to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
