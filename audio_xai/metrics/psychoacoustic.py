"""Psychoacoustic masking threshold for adversarial attacks.

Implements the simplified mask from Qin et al. 2019 ("Imperceptible,
Robust, and Targeted Adversarial Examples for Automatic Speech Recognition").
The full ISO 11172-3 (MPEG psychoacoustic model 1) involves tonal/non-tonal
maskers, downsampling the masking grid, and several thresholding steps that
are non-differentiable. This version keeps the core idea — louder frequency
components mask quieter nearby ones — while staying gradient-friendly.

Used in the attack loop: any spectral perturbation energy *above* this
threshold is audible and gets penalized; energy below it is, by the
psychoacoustic model, inaudible and is free.
"""

from __future__ import annotations

import torch


def hz_to_bark(freq_hz: torch.Tensor) -> torch.Tensor:
    """Traunmüller 1990 Hz->Bark conversion."""
    return 26.81 * freq_hz / (1960 + freq_hz) - 0.53


def absolute_threshold_of_hearing(freq_hz: torch.Tensor) -> torch.Tensor:
    """Terhardt's ATH model in dB SPL. Sets the floor for masking."""
    f_khz = freq_hz / 1000.0
    f_khz = torch.clamp(f_khz, min=0.02)  # avoid log singularity at 0
    ath = 3.64 * f_khz.pow(-0.8) - 6.5 * torch.exp(-0.6 * (f_khz - 3.3).pow(2)) + 1e-3 * f_khz.pow(4)
    return ath


def spreading_function(bark_diff: torch.Tensor) -> torch.Tensor:
    """Schroeder's spreading function in dB.

    bark_diff: distance in Bark between masker and maskee. Output is the
    drop in masking effect (dB) at that distance — i.e. how much weaker the
    maskee can be while still being masked.
    """
    return 15.81 + 7.5 * (bark_diff + 0.474) - 17.5 * torch.sqrt(1 + (bark_diff + 0.474).pow(2))


def masking_threshold(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 128,
    db_offset: float = 90.0,
) -> torch.Tensor:
    """Compute the per-frame masking threshold of a waveform.

    Parameters
    ----------
    waveform : [B, T] tensor.
    db_offset : SPL calibration. We don't know absolute playback level, so
        we pick an offset that makes the threshold roughly correspond to a
        normal listening level. 90 dB SPL is the standard MPEG choice.

    Returns
    -------
    threshold_db : [B, n_freq, n_frames] — dB level above which spectral
        energy at each (freq, time) bin becomes audible.
    """
    device = waveform.device
    window = torch.hann_window(n_fft, device=device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        return_complex=True,
    )
    psd_amp = spec.abs()  # [B, n_freq, n_frames]
    psd_db = db_offset + 20 * torch.log10(psd_amp + 1e-10)

    n_freq = psd_db.shape[1]
    freqs = torch.linspace(0, sample_rate / 2, n_freq, device=device)
    barks = hz_to_bark(freqs)  # [n_freq]

    # Spreading: pairwise Bark distances between every (masker, maskee) pair.
    # bark_dist[i, j] = barks[j] - barks[i]   (maskee j relative to masker i)
    bark_dist = barks.unsqueeze(0) - barks.unsqueeze(1)  # [n_freq, n_freq]
    spread_db = spreading_function(bark_dist)  # [n_freq, n_freq]

    # Each masker contributes (its level + spread - constant) at every other
    # frequency. The total threshold is the dB-domain sum of contributions
    # (approximated as max for differentiability and standard practice).
    # contrib[B, masker, maskee, frame] = psd_db[B, masker, frame] + spread[masker, maskee] - 6.025
    contrib = (
        psd_db.unsqueeze(2)  # [B, n_freq(masker), 1, n_frames]
        + spread_db.unsqueeze(0).unsqueeze(-1)  # [1, n_freq(masker), n_freq(maskee), 1]
        - 6.025  # offset from the standard model
    )
    # Combine maskers via softmax-max (smooth max) to keep gradients alive.
    masker_threshold_db, _ = contrib.max(dim=1)  # [B, n_freq(maskee), n_frames]

    # Combine with absolute threshold of hearing (also a floor).
    ath_db = absolute_threshold_of_hearing(freqs).to(device)  # [n_freq]
    ath_db = ath_db.view(1, -1, 1)  # broadcast

    threshold_db = torch.maximum(masker_threshold_db, ath_db)
    return threshold_db


def perturbation_audibility_loss(
    delta: torch.Tensor,
    threshold_db: torch.Tensor,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 128,
    db_offset: float = 90.0,
) -> torch.Tensor:
    """Penalty for perturbation energy above the masking threshold.

    Returns a scalar loss. Zero if the perturbation is fully masked (inaudible
    by the psychoacoustic model), positive otherwise.
    """
    window = torch.hann_window(n_fft, device=delta.device)
    delta_spec = torch.stft(delta, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    delta_db = db_offset + 20 * torch.log10(delta_spec.abs() + 1e-10)

    excess = torch.relu(delta_db - threshold_db)  # only above-threshold counts
    return excess.pow(2).mean()
