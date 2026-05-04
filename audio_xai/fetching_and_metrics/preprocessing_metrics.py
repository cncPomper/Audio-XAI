"""preprocessing.py ---------------- Audio preprocessing utilities and perceptual metric computation.

Usage:
    python preprocessing.py
"""

import os
import re
import time

import cdpam
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
from peaq_implementation import peaq_like
from pymcd.mcd import Calculate_MCD
from pystoi import stoi
from scipy.stats import entropy
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from zimtohrli import mos_from_signals

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_REAL = os.path.join(BASE_DIR, "real_songs")
FOLDER_FAKE = os.path.join(BASE_DIR, "fake_songs")
FOLDER_PERTURBED = os.path.join(BASE_DIR, "perturbed_songs")

NOISE_FACTOR = 0.005
CDPAM_SR = 22050
PESQ_SR = 16000
ZIMTOHRLI_SR = 48000
VIZ_SR = 16000

CDPAM_CHUNK_SIZE = int(CDPAM_SR * 2)  # 2 s
PESQ_CHUNK_SIZE = PESQ_SR * 8  # 8 s

# ---------------------------------------------------------------------------
# AUDIO I/O
# ---------------------------------------------------------------------------


def load_audio(path: str):
    """Load mono audio at CDPAM_SR and return arrays/tensors for all metrics.

    Returns:
    -------
    y_np         : np.ndarray [T]      at CDPAM_SR
    tensor_cdpam : Tensor [1, 1, T]
    tensor_pesq  : Tensor [T']         at PESQ_SR
    y_pesq_np    : np.ndarray [T']     at PESQ_SR
    """
    y, _ = librosa.load(path, sr=CDPAM_SR, mono=True)
    y_pesq = librosa.resample(y, orig_sr=CDPAM_SR, target_sr=PESQ_SR)
    tensor_cdpam = torch.tensor(y).unsqueeze(0).unsqueeze(0).float()
    tensor_pesq = torch.tensor(y_pesq).float()
    return y, tensor_cdpam, tensor_pesq, y_pesq


def load_for_visualization(path: str, sr: int = VIZ_SR, duration: float | None = None):
    """Load mono audio for waveform / mel-spectrogram visualisation."""
    audio, _ = librosa.load(path, sr=sr, mono=True, duration=duration)
    return audio, sr


# ---------------------------------------------------------------------------
# PREPROCESSING / AUGMENTATION
# ---------------------------------------------------------------------------


def add_noise(y_np: np.ndarray, noise_factor: float = NOISE_FACTOR) -> np.ndarray:
    """Add Gaussian noise and clip to [-1, 1]."""
    noise = np.random.randn(*y_np.shape) * noise_factor
    return np.clip(y_np + noise, -1.0, 1.0)


def visualize_audio(audio: np.ndarray, sr: int, title: str = "") -> None:
    """Display waveform and mel-spectrogram side-by-side."""
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(audio, sr=sr)
    plt.title(f"Waveform: {title}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    librosa.display.specshow(mel_spec_db, x_axis="time", y_axis="mel", sr=sr, fmax=8000)
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel-spectrogram (Log scale)")
    plt.tight_layout()
    plt.show()


def distort_signal(
    ref_waveform: torch.Tensor,
    sample_rate: int,
    noise_std: float = 0.01,
    gain_db: float = -2.0,
    lowpass_hz: float = 7000.0,
) -> torch.Tensor:
    """Apply additive Gaussian noise, apply a gain in decibels, and apply a low-pass biquad filter to an input waveform.

    Parameters:
        ref_waveform (torch.Tensor): Input audio of shape [T] or [C, T]. If 2-D, channels are averaged to produce mono before processing.
        sample_rate (int): Sample rate of the input waveform in Hz (used by the low-pass filter).
        noise_std (float): Standard deviation of the additive Gaussian noise applied to the signal.
        gain_db (float): Gain to apply in decibels before filtering (negative values attenuate).
        lowpass_hz (float): Cutoff frequency for the low-pass biquad filter in Hz.

    Returns:
        torch.Tensor: Processed mono waveform of shape [1, T] with samples clipped to the range [-1.0, 1.0].

    Raises:
        ValueError: If `ref_waveform` does not have shape [T] or [C, T].
    """
    if ref_waveform.dim() == 1:
        ref = ref_waveform.unsqueeze(0)
    elif ref_waveform.dim() == 2:
        ref = ref_waveform.mean(dim=0, keepdim=True)
    else:
        raise ValueError("ref_waveform must have shape [T] or [C, T]")

    ref = ref.float().clone()
    noisy = ref + noise_std * torch.randn_like(ref)
    gain_lin = 10 ** (gain_db / 20.0)
    distorted = torchaudio.functional.lowpass_biquad(noisy * gain_lin, sample_rate=sample_rate, cutoff_freq=lowpass_hz)
    return torch.clamp(distorted, -1.0, 1.0)


# ---------------------------------------------------------------------------
# METRIC COMPUTATION
# ---------------------------------------------------------------------------


def _iter_chunks(a, b, chunk_size: int):
    """Yield aligned chunk pairs along the last axis (tensors or arrays)."""
    T = min(a.shape[-1], b.shape[-1])
    for start in range(0, T - chunk_size + 1, chunk_size):
        yield a[..., start : start + chunk_size], b[..., start : start + chunk_size]


def compute_cdpam(metric, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> float:
    """CDPAM averaged over 0.5-s chunks.

    Expects tensors [1, 1, T].
    """
    a = tensor_a.squeeze(1)  # [1, T]
    b = tensor_b.squeeze(1)
    scores = []
    for ca, cb in _iter_chunks(a, b, CDPAM_CHUNK_SIZE):
        with torch.no_grad():
            scores.append(metric.forward(ca, cb).item())
    return float(np.mean(scores)) if scores else 0.0


def compute_pesq(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    sr: int = PESQ_SR,
    mode: str = "wb",
) -> float:
    """PESQ averaged over 8-s chunks.

    Expects 1-D tensors [T].
    """
    pesq_fn = PerceptualEvaluationSpeechQuality(sr, mode)
    a = tensor_a.squeeze()
    b = tensor_b.squeeze()
    scores = []
    for ca, cb in _iter_chunks(a, b, PESQ_CHUNK_SIZE):
        with torch.no_grad():
            scores.append(float(pesq_fn(ca, cb).item()))
    return float(np.mean(scores)) if scores else 0.0


def compute_stoi(
    y_a: np.ndarray,
    y_b: np.ndarray,
    sr: int = PESQ_SR,
    extended: bool = False,
) -> float:
    """STOI averaged over 8-s chunks.

    Expects 1-D arrays at PESQ_SR.
    """
    T = min(len(y_a), len(y_b))
    a, b = y_a[:T], y_b[:T]
    scores = []
    for start in range(0, T - PESQ_CHUNK_SIZE + 1, PESQ_CHUNK_SIZE):
        scores.append(
            stoi(
                a[start : start + PESQ_CHUNK_SIZE],
                b[start : start + PESQ_CHUNK_SIZE],
                sr,
                extended=extended,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def compute_zimtohrli(path_a: str, path_b: str, sr: int = ZIMTOHRLI_SR) -> float:
    """Zimtohrli MOS between two audio files."""
    sig_a, _ = librosa.load(path_a, sr=sr, mono=True)
    sig_b, _ = librosa.load(path_b, sr=sr, mono=True)
    return float(mos_from_signals(sig_a, sig_b))


def compute_snr(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """SNR: Stosunek sygnału do szumu (y_a = referencja, y_b = wygenerowany)."""
    min_len = min(len(y_a), len(y_b))
    ref, gen = y_a[:min_len], y_b[:min_len]

    noise = gen - ref
    sig_power = np.mean(ref**2)
    noise_power = np.mean(noise**2)

    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(sig_power / noise_power))


def compute_psnr(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """PSNR: Szczytowy stosunek sygnału do szumu."""
    min_len = min(len(y_a), len(y_b))
    ref, gen = y_a[:min_len], y_b[:min_len]

    mse = np.mean((ref - gen) ** 2)
    if mse == 0:
        return float("inf")

    max_val = np.max(np.abs(ref))
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def compute_lsd(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """LSD: Odległość Log-Spektralna między dwoma sygnałami."""
    # Zabezpieczenie przed nieskończonością przy logarytmie
    eps = 1e-8

    S_ref = np.abs(librosa.stft(y_a))
    S_gen = np.abs(librosa.stft(y_b))

    log_S_ref = np.log10(S_ref**2 + eps)
    log_S_gen = np.log10(S_gen**2 + eps)

    min_frames = min(log_S_ref.shape[1], log_S_gen.shape[1])
    log_S_ref = log_S_ref[:, :min_frames]
    log_S_gen = log_S_gen[:, :min_frames]

    lsd = np.mean(np.sqrt(np.mean((log_S_ref - log_S_gen) ** 2, axis=0)))
    return float(lsd)


def compute_dtw(y_a: np.ndarray, y_b: np.ndarray, sr: int = 22050) -> float:
    """DTW: Dynamic Time Warping na cechach MFCC."""
    mfcc_a = librosa.feature.mfcc(y=y_a, sr=sr)
    mfcc_b = librosa.feature.mfcc(y=y_b, sr=sr)

    D, _ = librosa.sequence.dtw(X=mfcc_a, Y=mfcc_b, metric="cosine")
    return float(D[-1, -1])


def compute_kl_divergence(y_a: np.ndarray, y_b: np.ndarray) -> float:
    """KL Divergence między uśrednionymi widmami amplitudowymi."""
    eps = 1e-10

    S_ref = np.mean(np.abs(librosa.stft(y_a)), axis=1)
    S_gen = np.mean(np.abs(librosa.stft(y_b)), axis=1)

    # Normalizacja do rozkładu prawdopodobieństwa (suma = 1)
    P_ref = (S_ref + eps) / np.sum(S_ref + eps)
    P_gen = (S_gen + eps) / np.sum(S_gen + eps)

    return float(entropy(P_ref, P_gen))


def compute_mcd(mcd_calc, path_a: str, path_b: str) -> float:
    """MCD: Mel-Cepstral Distortion (wymaga ścieżek do plików)."""
    return float(mcd_calc.calculate_mcd(path_a, path_b))


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _init_cdpam():
    """Load CDPAM model, bypassing the weights_only restriction."""
    original_load = torch.load

    def _load_unsafe(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = _load_unsafe  # type: ignore - intentional shadowing for unsafe loads
    model = cdpam.CDPAM(dev="cpu")
    torch.load = original_load  # type: ignore
    return model


# ---------------------------------------------------------------------------
# MAIN — full evaluation loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Inicjalizacja kalkulatora MCD przed pętlą
    mcd_calc = Calculate_MCD(MCD_mode="plain")

    # Dodane nowe metryki: snr, psnr, lsd, dtw, kl_div, mcd
    # (cdpam celowo usunięte z tej krotki, by nie zwalniać eksperymentu)
    metric_keys = (
        "zimtohrli",
        "pesq",
        "stoi",
        "pesq_pert",
        "stoi_pert",
        # "cdpam",
        "peaq*",
        "snr",
        "psnr",
        "lsd",
        "dtw",
        "kl_div",
        "mcd",
    )
    times = {k: [] for k in metric_keys}
    results = {k: [] for k in metric_keys}

    os.makedirs(FOLDER_PERTURBED, exist_ok=True)

    print("Initialising CDPAM model (this may take a moment)...")
    cdpam_model = _init_cdpam()

    real_files = sorted(f for f in os.listdir(FOLDER_REAL) if f.endswith(".wav"))
    print(f"Found {len(real_files)} real files. Starting analysis...\n")

    for real_file in real_files:
        match = re.search(r"real_(\d{5})", real_file)
        if not match:
            continue

        num = match.group(1)
        path_real = os.path.join(FOLDER_REAL, real_file)
        path_fake = os.path.join(FOLDER_FAKE, f"fake_{num}.wav")
        path_perturbed = os.path.join(FOLDER_PERTURBED, f"perturbed_fake_{num}.wav")

        if not os.path.exists(path_fake):
            print(f"Missing fake_{num}.wav — skipping.")
            continue

        print(f"--- Pair: {num} ---")

        y_real, t_real_cdpam, t_real_pesq, y_real_pesq = load_audio(path_real)
        y_fake, t_fake_cdpam, t_fake_pesq, y_fake_pesq = load_audio(path_fake)
        print(f"Real: {len(y_real)} samples @ {CDPAM_SR} Hz")

        # --- DOTYCHCZASOWE METRYKI ---
        t0 = time.time()
        mos = compute_zimtohrli(path_real, path_fake)
        times["zimtohrli"].append(time.time() - t0)
        results["zimtohrli"].append(mos)
        print(f" -> Zimtohrli MOS:           {mos:.4f}")

        t0 = time.time()
        pesq_score = compute_pesq(t_real_pesq, t_fake_pesq)
        times["pesq"].append(time.time() - t0)
        results["pesq"].append(pesq_score)
        print(f" -> PESQ WB:                 {pesq_score:.4f}")

        t0 = time.time()
        stoi_score = compute_stoi(y_real_pesq, y_fake_pesq)
        times["stoi"].append(time.time() - t0)
        results["stoi"].append(stoi_score)
        print(f" -> STOI:                    {stoi_score:.4f}")

        if "cdpam" in metric_keys:
            t0 = time.time()
            cdpam_score = compute_cdpam(cdpam_model, t_real_cdpam, t_fake_cdpam)
            times["cdpam"].append(time.time() - t0)
            results["cdpam"].append(cdpam_score)
            print(f" -> CDPAM:                   {cdpam_score:.4f}")

        t0 = time.time()
        snr_score = compute_snr(y_real, y_fake)
        times["snr"].append(time.time() - t0)
        results["snr"].append(snr_score)
        print(f" -> SNR:                     {snr_score:.4f}")

        t0 = time.time()
        psnr_score = compute_psnr(y_real, y_fake)
        times["psnr"].append(time.time() - t0)
        results["psnr"].append(psnr_score)
        print(f" -> PSNR:                    {psnr_score:.4f}")

        t0 = time.time()
        lsd_score = compute_lsd(y_real, y_fake)
        times["lsd"].append(time.time() - t0)
        results["lsd"].append(lsd_score)
        print(f" -> LSD:                     {lsd_score:.4f}")

        t0 = time.time()
        dtw_score = compute_dtw(y_real, y_fake, sr=CDPAM_SR)
        times["dtw"].append(time.time() - t0)
        results["dtw"].append(dtw_score)
        print(f" -> DTW Cost:                {dtw_score:.4f}")

        t0 = time.time()
        kl_div = compute_kl_divergence(y_real, y_fake)
        times["kl_div"].append(time.time() - t0)
        results["kl_div"].append(kl_div)
        print(f" -> KL Divergence:           {kl_div:.4f}")

        t0 = time.time()
        mcd_score = compute_mcd(mcd_calc, path_real, path_fake)
        times["mcd"].append(time.time() - t0)
        results["mcd"].append(mcd_score)
        print(f" -> MCD:                     {mcd_score:.4f}")

        # --- PERTURBACJE (Oryginalna część) ---
        y_perturbed = add_noise(y_fake)
        sf.write(path_perturbed, y_perturbed, CDPAM_SR)
        y_perturbed_pesq = librosa.resample(y_perturbed, orig_sr=CDPAM_SR, target_sr=PESQ_SR)
        t_pert_pesq = torch.tensor(y_perturbed_pesq).float()

        t0 = time.time()
        pesq_pert = compute_pesq(t_fake_pesq, t_pert_pesq)
        times["pesq_pert"].append(time.time() - t0)
        results["pesq_pert"].append(pesq_pert)
        print(f" -> PESQ WB (fake vs pert):  {pesq_pert:.4f}")

        t0 = time.time()
        peaq_score = peaq_like(path_fake, path_perturbed)
        times["peaq*"].append(time.time() - t0)
        results["peaq*"].append(peaq_score)
        print(f" -> PEAQ-like ODG (fake vs pert): {peaq_score:.4f}")

        t0 = time.time()
        stoi_pert = compute_stoi(y_fake_pesq, y_perturbed_pesq)
        times["stoi_pert"].append(time.time() - t0)
        results["stoi_pert"].append(stoi_pert)
        print(f" -> STOI (fake vs pert): {stoi_pert:.4f}")

        print("-" * 40)

    print("\n=== Average results ===")
    for key, vals in results.items():
        # Obsługa nieskończoności (gdy SNR/PSNR wyjdzie inf, ignorujemy to w średniej by nie zepsuć wydruku)
        valid_vals = [v for v in vals if v != float("inf")]
        avg = np.mean(valid_vals) if valid_vals else float("inf")
        print(f"  {key:25s}: {avg:.4f}")

    print("\n=== Average times [s] ===")
    for key, vals in times.items():
        print(f"  {key:25s}: {np.mean(vals):.4f}")
