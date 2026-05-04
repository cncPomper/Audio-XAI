import os
import re

import pandas as pd
import yt_dlp

# --- KONFIGURACJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER_Z_FAKE = os.path.join(BASE_DIR, "fake_songs")
FOLDER_NA_REAL = os.path.join(BASE_DIR, "real_songs")
ILOSC_PROBEK = 5

os.makedirs(FOLDER_NA_REAL, exist_ok=True)

print("Wczytywanie metadanych...")
# low_memory=False załatwia sprawę z DtypeWarning
df_fake = pd.read_csv("fake_songs.csv", low_memory=False)
df_real = pd.read_csv("real_songs.csv", low_memory=False)

# Pobieramy nazwy plików audio z naszego wypakowanego folderu
wypakowane_pliki = [f for f in os.listdir(FOLDER_Z_FAKE) if f.endswith((".wav", ".mp3", ".flac"))][:ILOSC_PROBEK]

print(f"Znaleziono {len(wypakowane_pliki)} plików fake. Szukam odpowiedników na YT...\n")

for plik in wypakowane_pliki:
    print(f"\n--- Przetwarzam fałszywy plik: {plik} ---")

    # 1. Wyciągamy 5-cyfrowy numer z nazwy pliku za pomocą wyrażenia regularnego
    # Szuka wzorca "fake_XXXXX" i wyciąga same cyfry
    match = re.search(r"fake_(\d{5})", plik)

    if not match:
        print(f" -> UWAGA: Nie udało się wyciągnąć numeru z pliku {plik}")
        continue

    numer = match.group(1)  # np. "00001"
    nazwa_real = f"real_{numer}"
    print(f" -> Szukam prawdziwego odpowiednika: {nazwa_real}")

    # 2. Szukamy tej nazwy w prawdziwych piosenkach
    # Zakładam, że w df_real kolumna nazywa się 'filename'
    wiersz_real = df_real[df_real["filename"].astype(str).str.contains(nazwa_real, na=False)]

    if wiersz_real.empty:
        print(f" -> UWAGA: Brak wpisu dla {nazwa_real} w real_songs.csv")
        continue

    y_id = wiersz_real.iloc[0]["youtube_id"]

    # 3. Pobieramy z YouTube
    youtube_url = f"https://www.youtube.com/watch?v={y_id}"
    print(f" -> Znaleziono w metadanych (YouTube ID: {y_id}). Pobieranie...")

    # --- ZMIANA: Konfiguracja przeniesiona do pętli, aby ustawiać outtmpl dynamicznie ---
    ydl_opts = {
        "format": "bestaudio/best",
        # Nazwa pliku to teraz np. real_00001.%(ext)s
        "outtmpl": f"{FOLDER_NA_REAL}/{nazwa_real}.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }
        ],
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([youtube_url])
            print(f" -> Sukces! Zapisano jako: {nazwa_real}.wav")
        except Exception as e:
            print(f" -> Błąd pobierania z YT: {e}")
