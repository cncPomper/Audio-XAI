# Renamed to wav2vec2_binary.py — kept for backward compatibility.
from audio_xai.models.wav2vec2_binary import (  # noqa: F401
    Wav2Vec2Binary as SpecTraBinary,
    WAV2VEC2_CHECKPOINT as SPECTRA_CHECKPOINT,
    WAV2VEC2_SAMPLE_RATE as SPECTRA_SAMPLE_RATE,
    WAV2VEC2_HIDDEN_SIZE as SPECTRA_HIDDEN_SIZE,
)