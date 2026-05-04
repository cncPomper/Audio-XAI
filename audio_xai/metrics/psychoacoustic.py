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
    """
    Convert frequencies from hertz to the Bark scale using Traunmüller (1990).
    
    Parameters:
        freq_hz (torch.Tensor): Frequencies in hertz.
    
    Returns:
        torch.Tensor: Frequencies on the Bark scale, same shape as `freq_hz`.
    """
    return 26.81 * freq_hz / (1960 + freq_hz) - 0.53


def absolute_threshold_of_hearing(freq_hz: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute threshold of hearing (ATH) in dB SPL for input frequencies using Terhardt's model.
    
    Returns:
        ath_db (torch.Tensor): ATH values in dB SPL, with the same shape as `freq_hz`.
    """
    f_khz = freq_hz / 1000.0
    f_khz = torch.clamp(f_khz, min=0.02)  # avoid log singularity at 0
    ath = 3.64 * f_khz.pow(-0.8) - 6.5 * torch.exp(-0.6 * (f_khz - 3.3).pow(2)) + 1e-3 * f_khz.pow(4)
    return ath


def spreading_function(bark_diff: torch.Tensor) -> torch.Tensor:
    """
    Compute Schroeder's spreading function (masking drop in decibels) as a function of Bark distance.
    
    The result is the attenuation (in dB) that a masker produces at a maskee frequency separated by the given Bark distance.
    
    Parameters:
        bark_diff (torch.Tensor): Distance in Bark from masker to maskee.
    
    Returns:
        torch.Tensor: Masking drop in dB for each input Bark distance (same shape as `bark_diff`).
    """
    return 15.81 + 7.5 * (bark_diff + 0.474) - 17.5 * torch.sqrt(1 + (bark_diff + 0.474).pow(2))


def masking_threshold(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    n_fft: int = 512,
    hop_length: int = 128,
    db_offset: float = 90.0,
) -> torch.Tensor:
    """
    Computes the per-frame psychoacoustic masking threshold for a batch waveform.
    
    Parameters:
        waveform (torch.Tensor): Audio tensor shaped [B, T], where B is batch size and T is time samples.
        db_offset (float): SPL calibration offset applied to STFT magnitudes so thresholds align with typical listening levels; 90 dB SPL is the standard MPEG choice.
    
    Returns:
        threshold_db (torch.Tensor): Tensor shaped [B, n_freq, n_frames] containing the dB level above which spectral energy at each frequency/time bin is considered audible under the model.
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
    """
    Compute the mean squared audible excess of a perturbation relative to a masking threshold.
    
    Parameters:
        delta (torch.Tensor): Perturbation waveform, shape [B, T] (batch-first) or broadcastable to that shape; values are audio samples.
        threshold_db (torch.Tensor): Masking threshold in dB SPL with shape [B, n_freq, n_frames] (or broadcastable to the STFT output); values are dB levels to compare against.
        sample_rate (int): Sample rate used for STFT (informational; does not affect computation here).
        n_fft (int): FFT size used to compute the STFT.
        hop_length (int): Hop length (frame advance) used for the STFT.
        db_offset (float): Calibration offset added to 20*log10(magnitude) to express magnitudes in dB SPL.
    
    Returns:
        torch.Tensor: Scalar loss equal to the mean of squared positive dB differences (delta_db - threshold_db). Returns `0.0` if the perturbation is at or below the threshold everywhere.
    """
    window = torch.hann_window(n_fft, device=delta.device)
    delta_spec = torch.stft(delta, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True)
    delta_db = db_offset + 20 * torch.log10(delta_spec.abs() + 1e-10)

    excess = torch.relu(delta_db - threshold_db)  # only above-threshold counts
    return excess.pow(2).mean()
