import librosa
import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


class ViSQOL:
    def __init__(self, sr=16000, patch_size_frames=30):
        self.sr = sr
        self.patch_size = patch_size_frames  # 30 ramek = 480 ms [cite: 274]
        # Współczynniki do mapowania NSIM -> MOS [cite: 445]
        self.a = 158.7
        self.b = -373.6
        self.c = 295.5
        self.d = -75.3

    def _level_align(self, x, y):
        """
        Wyrównuje poziom mocy sygnału zniekształconego (y)
        do sygnału referencyjnego (x)[cite: 267].
        """
        power_x = np.sum(x**2)
        power_y = np.sum(y**2)
        if power_y == 0:
            return y
        return y * np.sqrt(power_x / power_y)

    def _get_spectrogram(self, signal):
        """
        Tworzy spektrogram za pomocą STFT (Short-Term Fourier Transform) [cite: 268-270].
        Używa okna Hamminga 512 próbek z 50% nałożeniem dla 16 kHz[cite: 270].
        """
        n_fft = 512 if self.sr == 16000 else 256
        hop_length = n_fft // 2

        # Obliczenie STFT i przejście na amplitudę
        stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, window="hamming")
        spectrogram = np.abs(stft_matrix)

        # Filtrowanie do pasm krytycznych (uproszczone mapowanie na skalę Mel dla kompatybilności)
        # Dokument wymienia 16 do 21 pasm[cite: 274], użyjemy 21 pasm dla wideband
        spectrogram = librosa.feature.melspectrogram(S=spectrogram, sr=self.sr, n_mels=21, fmin=50, fmax=8000)
        return spectrogram

    def _voice_activity_detection(self, r_patch, threshold=0.01):
        """
        Prosty detektor aktywności głosu (VAD) oparty na progowaniu energii[cite: 275].
        """
        return np.mean(r_patch) > threshold

    def _calculate_nsim(self, r_patch, d_patch, L):
        """
        Neurogram Similarity Index Measure (NSIM) [cite: 430-434].
        """
        # Zabezpieczenia matematyczne na brzegach [cite: 434]
        C1 = 0.01 * L
        C3 = (0.03 * L) ** 2

        mu_r = np.mean(r_patch)
        mu_d = np.mean(d_patch)
        sigma_r_sq = np.var(r_patch)
        sigma_d_sq = np.var(d_patch)

        # Obliczanie kowariancji
        sigma_rd = np.mean((r_patch - mu_r) * (d_patch - mu_d))

        # Równanie 4
        term1 = (2 * mu_r * mu_d + C1) / (mu_r**2 + mu_d**2 + C1)
        term2 = (2 * sigma_rd + C3) / (sigma_r_sq + sigma_d_sq + C3)

        return term1 * term2

    def _warp_patch(self, patch, warp_factor):
        """
        Wykonywanie dwuwymiarowej interpolacji sześciennej w celu odkształcenia w czasie[cite: 360].
        Współczynniki to 0.95 (krótszy), 1.0 (bez zmian) i 1.05 (dłuższy)[cite: 284].
        """
        # Zmiana wymiaru czasowego przy zachowaniu wymiaru częstotliwościowego
        return zoom(patch, (1.0, warp_factor), order=3)

    def _map_to_mos(self, Q):
        """
        Mapowanie uśrednionego wyniku NSIM na skalę MOS (1-5)
        przy użyciu dopasowania wielomianowego [cite: 438, 441-445].
        """
        mos = self.a * (Q**3) + self.b * (Q**2) + self.c * Q + self.d
        return np.clip(mos, 1.0, 5.0)

    def evaluate(self, ref_signal, deg_signal):
        """
        Główna funkcja oceniająca jakość mowy.
        """
        # 1. Wyrównanie poziomu mocy [cite: 242, 267]
        deg_signal = self._level_align(ref_signal, deg_signal)

        # 2. Tworzenie spektrogramów [cite: 243, 245]
        r = self._get_spectrogram(ref_signal)
        d = self._get_spectrogram(deg_signal)

        # Ograniczanie do wartości minimalnej ze spektrogramu referencyjnego [cite: 246, 247, 271]
        r_min = np.min(r)
        r = np.clip(r - r_min, 0, None)
        d = np.clip(d - r_min, 0, None)

        # Globalny zakres intensywności L (wymagany do stałych C1, C3) [cite: 434, 465]
        L = np.max(r) - np.min(r)
        if L == 0:
            L = 1e-6

        n_frames = r.shape[1]
        patch_scores = []

        # Zdefiniowane czynniki odkształceń [cite: 284, 457]
        warp_factors = [0.95, 1.0, 1.05]

        # 3. Tworzenie i analiza aktywnych paczek (patchy) [cite: 248-264]
        for t in range(0, n_frames - self.patch_size):
            r_patch = r[:, t : t + self.patch_size]

            # Wymagany Voice Activity Detection (VAD) [cite: 249]
            if not self._voice_activity_detection(r_patch):
                continue

            best_patch_q = -1.0

            # 4. Globalne poszukiwanie dopasowania w zniekształconym sygnale d
            # Uproszczone: sprawdzamy lokalne okno czasowe, aby zapobiec
            # przeszukiwaniu całego sygnału (względy wydajnościowe)
            search_window = 10
            t_start = max(0, t - search_window)
            t_end = min(d.shape[1] - self.patch_size, t + search_window)

            for t_d in range(t_start, t_end):
                d_patch_candidate = d[:, t_d : t_d + self.patch_size]

                # Testowanie odkształceń (warps) dla lepszego dopasowania
                # czasowego (jitter/clock drift) [cite: 283, 361-363]
                for warp in warp_factors:
                    warped_r_patch = self._warp_patch(r_patch, warp)

                    # Wyrównanie wymiarów w przypadku, gdy zoom zaokrągli o 1 ramkę
                    min_frames = min(warped_r_patch.shape[1], d_patch_candidate.shape[1])
                    cur_r = warped_r_patch[:, :min_frames]
                    cur_d = d_patch_candidate[:, :min_frames]

                    # 5. Porównanie podobieństwa [cite: 257, 258]
                    q_val = self._calculate_nsim(cur_r, cur_d, L)

                    if q_val > best_patch_q:
                        best_patch_q = q_val

            if best_patch_q != -1.0:
                patch_scores.append(best_patch_q)

        # 6. Agregacja wyników i mapowanie MOS [cite: 264, 265]
        if not patch_scores:
            return 1.0  # Minimalny MOS w przypadku kompletnego braku sygnału

        Q_mean = np.mean(patch_scores)
        Q_mos = self._map_to_mos(Q_mean)

        return Q_mos


# --- PRZYKŁAD UŻYCIA ---
if __name__ == "__main__":
    # Załaduj dwa sygnały audio (referencyjny i zniekształcony)
    visqol_metric = ViSQOL(sr=16000)
    reference_audio, _ = librosa.load("example_audio.mp3", sr=16000)

    for name in tqdm(
        ["degraded_noisy.mp3", "degraded_clipped.mp3", "degraded_low_res.mp3"],
        desc="Ocena jakości zniekształconych sygnałów",
    ):
        print(f"\nOcena jakości dla: {name}")
        degraded_audio, _ = librosa.load(name, sr=16000)
        mos_score = visqol_metric.evaluate(reference_audio, degraded_audio)
        print(f"Przewidywany wynik MOS: {mos_score:.2f}")
        with open("mos_scores.txt", "a") as f:
            f.write(f"{name}: {mos_score:.2f}\n")
