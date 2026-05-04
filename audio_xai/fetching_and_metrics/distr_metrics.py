import os

import librosa
import numpy as np
import torch

# ---------------------------------------------------------------------------
# KONFIGURACJA
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_REAL = os.path.join(BASE_DIR, "real_songs")
FOLDER_FAKE = os.path.join(BASE_DIR, "fake_songs")
FOLDER_PERTURBED = os.path.join(BASE_DIR, "perturbed_songs")
SAMPLE_RATE = 22050

# ---------------------------------------------------------------------------
# MATEMATYKA METRYK (PYTORCH)
# ---------------------------------------------------------------------------


def calculate_kad(embeddings_ref: torch.Tensor, embeddings_gen: torch.Tensor, sigma: float = 1.0) -> float:
    """
    Compute the Kernel Audio Distance (an unbiased MMD^2 estimate using an RBF kernel) between two sets of embeddings.
    
    Parameters:
        embeddings_ref (torch.Tensor): Reference embeddings with shape [N, D] where N >= 0 is number of samples and D is feature dimension.
        embeddings_gen (torch.Tensor): Generated embeddings with shape [M, D] where M >= 0 is number of samples and D matches reference features.
        sigma (float): RBF kernel bandwidth (standard deviation).
    
    Returns:
        float: The KAD value (sum of intra-set kernel terms minus twice the cross-set term). Returns `nan` if either input has fewer than 2 samples.
    """
    # --- NAPRAWA: Normalizacja L2 ---
    # Skaluje wektory, zapobiegając ucięciu wyniku do zera przez funkcję exp()
    embeddings_ref = torch.nn.functional.normalize(embeddings_ref, p=2, dim=1)
    embeddings_gen = torch.nn.functional.normalize(embeddings_gen, p=2, dim=1)

    def rbf_kernel(X, Y, sig):
        XX = torch.sum(X**2, dim=1).view(-1, 1)
        YY = torch.sum(Y**2, dim=1).view(1, -1)
        # torch.clamp zapobiega ujemnym wartościom bliskim zera z powodu błędów precyzji float
        dist = torch.clamp(XX + YY - 2.0 * torch.matmul(X, Y.t()), min=0.0)
        return torch.exp(-dist / (2 * sig**2))

    n, m = embeddings_ref.shape[0], embeddings_gen.shape[0]

    if n < 2 or m < 2:
        return float("nan")

    K_xx = rbf_kernel(embeddings_ref, embeddings_ref, sigma)
    K_xx.fill_diagonal_(0)
    sum_xx = torch.sum(K_xx) / (n * (n - 1))

    K_yy = rbf_kernel(embeddings_gen, embeddings_gen, sigma)
    K_yy.fill_diagonal_(0)
    sum_yy = torch.sum(K_yy) / (m * (m - 1))

    K_xy = rbf_kernel(embeddings_ref, embeddings_gen, sigma)
    sum_xy = torch.sum(K_xy) / (n * m)

    kad_score = sum_xx + sum_yy - 2 * sum_xy
    return float(kad_score.item())


def calculate_inception_score(probs_gen: torch.Tensor) -> float:
    """
    Oblicza Inception Score.
    Oczekuje tensora o kształcie [N, C] z prawdopodobieństwami (np. wyjście softmax).
    """
    if probs_gen.shape[0] == 0:
        return float("nan")

    p_y = torch.mean(probs_gen, dim=0)  # Marginesowy rozkład prawdopodobieństwa

    # KL(p(y|x) || p(y))
    kl_div = probs_gen * (torch.log(probs_gen + 1e-10) - torch.log(p_y + 1e-10))
    kl_sum = torch.sum(kl_div, dim=1)

    is_score = torch.exp(torch.mean(kl_sum))
    return float(is_score.item())


# ---------------------------------------------------------------------------
# EKSTRAKCJA CECH (SYMULATOR MODELU)
# ---------------------------------------------------------------------------


def extract_features(folder_path: str):
    """
    Skanuje folder i wyciąga z plików wektory cech.
    UWAGA: Do testów używamy tu uśrednionego Mel-spektrogramu.
    W docelowym systemie użyj np. torchaudio.models.wav2vec2 lub VGGish.
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    files.sort()

    embeddings = []
    probs = []

    print(f"Przetwarzanie {len(files)} plików w: {folder_path}")
    for f in files:
        path = os.path.join(folder_path, f)
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

        # 1. Tworzymy pseudo-embedding (średnia z 128 pasm Mel) [Wymiar: 128]
        mel = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        embedding = np.mean(mel_db, axis=1)  # Średnia po czasie
        embeddings.append(embedding)

        # 2. Tworzymy pseudo-prawdopodobieństwa klas dla Inception Score
        # (Wymaga wyjścia w postaci prawdopodobieństw sumujących się do 1, np. 50 klas)
        # Symulujemy, przepuszczając embedding przez prosty softmax
        emb_tensor = torch.tensor(embedding, dtype=torch.float32)
        simulated_logits = emb_tensor[:50]  # Bierzemy pierwsze 50 wartości jako logity
        prob = torch.softmax(simulated_logits, dim=0).numpy()
        probs.append(prob)

    tensor_embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32)
    tensor_probs = torch.tensor(np.array(probs), dtype=torch.float32)

    return tensor_embeddings, tensor_probs


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not os.path.exists(FOLDER_REAL) or not os.path.exists(FOLDER_FAKE):
        print("Błąd: Upewnij się, że foldery FOLDER_REAL i FOLDER_FAKE istnieją.")
        exit(1)

    print("=== EKSTRAKCJA CECH (Prawdziwe Audio) ===")
    real_embeddings, _ = extract_features(FOLDER_REAL)

    print("\n=== EKSTRAKCJA CECH (Wygenerowane Audio) ===")
    fake_embeddings, fake_probs = extract_features(FOLDER_FAKE)

    print("\n=== WYNIKI METRYK POPULACYJNYCH ===")
    if real_embeddings.shape[0] < 2 or fake_embeddings.shape[0] < 2:
        print("Potrzebujesz co najmniej 2 plików w każdym folderze, aby policzyć KAD.")
    else:
        kad = calculate_kad(real_embeddings, fake_embeddings)
        # Wynik KAD > 0. Bliżej 0 = mniejsza różnica między rozkładami = LEPIEJ.
        print(f"KAD (Kernel Audio Distance)  : {kad:.6f}")

        inc_score = calculate_inception_score(fake_probs)
        # Wyższy IS = lepsza jakość i większa różnorodność.
        print(f"IS (Inception Score)         : {inc_score:.6f}")
