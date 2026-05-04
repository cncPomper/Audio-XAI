"""Utility helpers: YouTube audio download and project path status."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, cast

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, PostProcessingError

from audio_xai.config import DATA_DIR, FIGURES_DIR, MODELS_DIR, PROJ_ROOT, REPORTS_DIR


def download_audio(filename: str, yt_id: str) -> Path | None:
    """Download audio from a YouTube video.

    Args:
        filename (str): The base path/name to save audio as (without extension).
        yt_id (str): The YouTube video ID.

    Returns:
        Path | None: Downloaded file path when successful, otherwise None.
    """
    url = f"https://www.youtube.com/watch?v={yt_id}"
    ffmpeg_path = shutil.which("ffmpeg")

    base_options: dict[str, Any] = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "web"],
            }
        },
        "outtmpl": f"{filename}.%(ext)s",
        "noplaylist": True,
        "quiet": False,
        "retries": 3,
    }

    if ffmpeg_path:
        base_options["ffmpeg_location"] = str(Path(ffmpeg_path).parent)

    try:
        with YoutubeDL(
            cast(
                Any,
                {
                    **base_options,
                    "postprocessors": [
                        {
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "mp3",
                            "preferredquality": "192",
                        }
                    ],
                },
            )
        ) as video:
            info = video.extract_info(url, download=True)
            if info is None:
                return None

            downloaded_path = Path(video.prepare_filename(info))
            return downloaded_path.with_suffix(".mp3")
    except PostProcessingError as error:
        error_message = str(error).lower()
        ffmpeg_missing = "ffmpeg not found" in error_message or "ffprobe and ffmpeg not found" in error_message

        if ffmpeg_missing:
            print("ffmpeg/ffprobe not found, saving original audio format instead of mp3.")
            try:
                with YoutubeDL(cast(Any, base_options)) as video:
                    info = video.extract_info(url, download=True)
                    if info is None:
                        return None
                    return Path(video.prepare_filename(info))
            except DownloadError:
                print(f"{yt_id} not available")
                return None

        print(f"postprocessing failed for {yt_id}: {error}")
        return None
    except DownloadError:
        print(f"Download failed for {yt_id}; YouTube may be blocking this request or the video may be unavailable.")
        return None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the YouTube audio download helper.

    Returns:
        argparse.Namespace: Parsed arguments with ``yt_id`` and ``output`` attributes.
    """
    parser = argparse.ArgumentParser(description="Download YouTube audio via yt-dlp.")
    parser.add_argument("--yt-id", required=True, help="YouTube video ID, e.g. dQw4w9WgXcQ")
    parser.add_argument(
        "--output",
        required=True,
        help="Output base name or path without extension (e.g. audio-xai-data/song)",
    )
    return parser.parse_args()


def main() -> int:
    """Download YouTube audio from the ID and output path provided on the command line.

    Returns:
        int: Exit code — 0 on success, 1 on download failure.
    """
    args = parse_args()
    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    downloaded = download_audio(str(out_base), args.yt_id)
    if downloaded is None:
        print("Download failed.")
        return 1

    print(f"Saved: {downloaded}")
    return 0


def get_project_paths_status() -> list[tuple[str, Path, bool]]:
    """Return key project paths together with their existence status."""
    project_paths = {
        "project_root": PROJ_ROOT,
        "data": DATA_DIR,
        "models": MODELS_DIR,
        "reports": REPORTS_DIR,
        "figures": FIGURES_DIR,
    }
    return [(name, path, path.exists()) for name, path in project_paths.items()]


if __name__ == "__main__":
    for name, path, exists in get_project_paths_status():
        print(f"{name}: {path} - {'Exists' if exists else 'Missing'}")

    saved_path = download_audio("example_audio", "dQw4w9WgXcQ")
    if saved_path is not None:
        print(f"Downloaded: {saved_path}")
