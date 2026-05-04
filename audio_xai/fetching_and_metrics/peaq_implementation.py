from math import gcd

import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import get_window, resample_poly

# =========================
# Pomocnicze
# =========================


def hz_to_bark(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)


# =========================
# I/O
# =========================


def read_wav(path, target_sr=48000, verbose=True):
    """
    Read an audio file and return a 2-D float64 waveform and its sample rate, optionally resampling to a target rate.
    
    Reads `path` with channels preserved as a 2-D array (shape: samples x channels). If the file's sample rate differs from `target_sr`, the audio is resampled to `target_sr` before being returned. The returned waveform is cast to `np.float64`.
    
    Parameters:
        path (str): Path to the audio file to read.
        target_sr (int, optional): Desired output sample rate; if different from the file's rate, resampling is performed. Default is 48000.
        verbose (bool, optional): If True, print file info and resampling details. Default is True.
    
    Returns:
        tuple[np.ndarray, int]: A tuple (x, sr) where `x` is a 2-D numpy array of dtype `np.float64` (samples x channels) and `sr` is the sample rate (Hz) after any resampling.
    """
    x, sr = sf.read(path, always_2d=True)

    if verbose:
        print(f"[read_wav] {path}")
        print(f"  sr={sr}, shape={x.shape}, dtype={x.dtype}")
        print(f"  min={np.min(x):.6f}, max={np.max(x):.6f}, rms={np.sqrt(np.mean(x * x)):.6f}")

    if sr != target_sr:
        g = gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        if verbose:
            print(f"  resampling: {sr} -> {target_sr}, up={up}, down={down}")
        x = resample_poly(x, up, down, axis=0)
        sr = target_sr

    return x.astype(np.float64), sr


def align_and_trim(ref, test, verbose=True):
    n = min(len(ref), len(test))
    if verbose:
        print(f"[align_and_trim] ref_len={len(ref)}, test_len={len(test)}, used={n}")
    return ref[:n], test[:n]


# =========================
# Analiza widmowa
# =========================


def frame_signal(x, frame_size=2048, hop=1024):
    if len(x) < frame_size:
        x = np.pad(x, (0, frame_size - len(x)))

    n_frames = 1 + (len(x) - frame_size) // hop

    frames = np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_size),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )

    return frames.copy()


def bark_bands(sr, n_fft, n_bands=109):
    freqs = rfftfreq(n_fft, 1 / sr)
    bark = hz_to_bark(freqs)
    bark_max = hz_to_bark(sr / 2)

    edges = np.linspace(0, bark_max, n_bands + 1)

    bands = []
    for i in range(n_bands):
        idx = np.where((bark >= edges[i]) & (bark < edges[i + 1]))[0]
        if len(idx) > 0:
            bands.append(idx)

    return bands


def band_power_spectrogram(x, sr, frame_size=2048, hop=1024, verbose=True):
    frames = frame_signal(x, frame_size, hop)
    win = get_window("hann", frame_size)

    if verbose:
        print("[band_power_spectrogram]")
        print(f"  frames={frames.shape[0]}")

    spec = np.abs(rfft(frames * win[None, :], axis=1)) ** 2
    bands = bark_bands(sr, frame_size)

    band_power = np.zeros((spec.shape[0], len(bands)))

    for b, idx in enumerate(bands):
        band_power[:, b] = np.sum(spec[:, idx], axis=1)

    band_power += 1e-20

    return band_power


# =========================
# MOVs
# =========================


def spectral_centroid_from_bands(bp):
    weights = np.arange(bp.shape[1]) + 1
    return np.sum(bp * weights[None, :], axis=1) / np.sum(bp, axis=1)


def compute_movs_channel(ref, test, sr, verbose=True):
    print("\n[compute_movs_channel]")

    # --- sanity SNR ---
    err = test - ref
    snr_est = 10 * np.log10(np.mean(ref**2) / (np.mean(err**2) + 1e-30))
    print(f"  estimated SNR: {snr_est:.2f} dB")

    ref_bp = band_power_spectrogram(ref, sr, verbose=verbose)
    test_bp = band_power_spectrogram(test, sr, verbose=verbose)

    n = min(len(ref_bp), len(test_bp))
    ref_bp = ref_bp[:n]
    test_bp = test_bp[:n]

    ref_db = 10 * np.log10(ref_bp)
    test_db = 10 * np.log10(test_bp)
    diff_db = test_db - ref_db

    noise_bp = np.maximum(test_bp - ref_bp, 0.0)
    missing_bp = np.maximum(ref_bp - test_bp, 0.0)

    # ===== KLUCZOWA POPRAWKA =====
    frame_max = np.max(ref_bp, axis=1, keepdims=True)
    mask = ref_bp > (frame_max * 1e-5)

    print("[masking]")
    print(f"  active ratio: {np.mean(mask):.4f}")

    safe_ref = np.maximum(ref_bp, frame_max * 1e-5)

    rel_noise = noise_bp / safe_ref
    rel_missing = missing_bp / safe_ref
    rel_diff = np.abs(test_bp - ref_bp) / safe_ref

    avg_noise_loudness = np.mean(np.log1p(rel_noise[mask]))
    max_nmr = np.percentile(np.maximum(diff_db[mask], 0), 95)
    avg_lin_dist = np.mean(rel_diff[mask])
    missing_components = np.mean(np.log1p(rel_missing[mask]))

    c_ref = spectral_centroid_from_bands(ref_bp)
    c_test = spectral_centroid_from_bands(test_bp)
    centroid_shift = np.mean(np.abs(c_test - c_ref)) / ref_bp.shape[1]

    ref_energy_band = np.mean(ref_bp, axis=0)
    test_energy_band = np.mean(test_bp, axis=0)

    ref_bw = np.max(np.where(ref_energy_band > 1e-6)[0])
    test_bw = np.max(np.where(test_energy_band > 1e-6)[0])

    bandwidth_loss = max(0, ref_bw - test_bw) / max(ref_bw, 1)

    ref_env = np.sqrt(np.sum(ref_bp, axis=1))
    test_env = np.sqrt(np.sum(test_bp, axis=1))
    modulation_diff = np.mean(np.abs(test_env - ref_env)) / (np.mean(ref_env) + 1e-12)

    movs = {
        "avg_noise_loudness": float(avg_noise_loudness),
        "max_nmr_db_p95": float(max_nmr),
        "avg_lin_dist": float(avg_lin_dist),
        "missing_components": float(missing_components),
        "centroid_shift": float(centroid_shift),
        "bandwidth_loss": float(bandwidth_loss),
        "modulation_diff": float(modulation_diff),
    }

    print("[MOVs]")
    for k, v in movs.items():
        print(f"  {k}: {v:.8f}")

    return movs


# =========================
# Mapowanie do ODG
# =========================


def movs_to_odg(movs):
    raw = (
        0.55 * np.log1p(movs["avg_noise_loudness"])
        + 0.030 * movs["max_nmr_db_p95"]
        + 0.35 * np.log1p(movs["avg_lin_dist"])
        + 0.45 * np.log1p(movs["missing_components"])
        + 1.20 * movs["centroid_shift"]
        + 1.80 * movs["bandwidth_loss"]
        + 0.55 * np.log1p(movs["modulation_diff"])
    )

    odg = -4.0 * (1.0 - np.exp(-raw))
    odg = float(np.clip(odg, -4.0, 0.0))

    print("[movs_to_odg]")
    print(f"  raw={raw:.6f}")
    print(f"  ODG={odg:.4f}")

    return odg


# =========================
# MAIN METRYKA
# =========================


def peaq_like(ref_wav, test_wav):
    print("\n========== PEAQ DEBUG ==========")

    ref, sr = read_wav(ref_wav)
    test, _ = read_wav(test_wav)

    ref, test = align_and_trim(ref, test)

    odgs = []

    for ch in range(min(ref.shape[1], test.shape[1])):
        print(f"\n--- CHANNEL {ch} ---")
        movs = compute_movs_channel(ref[:, ch], test[:, ch], sr)
        odg = movs_to_odg(movs)
        odgs.append(odg)

    final = np.mean(odgs)

    print("\nFINAL ODG:", final)
    return final


# =========================
# TEST + NOISE
# =========================


def add_gaussian_noise(input_wav, output_wav, snr_db=20):
    x, sr = read_wav(input_wav)

    signal_power = np.mean(x**2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.normal(0, np.sqrt(noise_power), x.shape)
    noisy = np.clip(x + noise, -1, 1)

    sf.write(output_wav, noisy, sr, subtype="FLOAT")

    print(f"[noise] target SNR={snr_db} dB")
    return noisy


# =========================
# MAIN (BEZ CLI)
# =========================

if __name__ == "__main__":
    ref_wav = "reference.wav"
    test_wav = "noisy.wav"

    # opcjonalnie generuj szum
    add_gaussian_noise(ref_wav, test_wav, snr_db=2000)

    peaq_like(ref_wav, test_wav)
