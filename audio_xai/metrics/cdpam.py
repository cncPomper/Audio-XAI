import sys
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.notebook import tqdm
import cdpam


def calculate_cdpam_score(
    ref_audio: str | torch.Tensor, 
    cand_audio: str | torch.Tensor, 
    model_type: str = "acoustic",
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
) -> float | str:
    """Computes CDPAM (Conditional Deep Perceptual Audio Metric) for an audio pair.
    
    Accepts file paths or 2D tensors. Model type options: 'acoustic' or 'cross'.
    """
    try:
        # Initialize the updated contrastive conditional model architecture
        loss_fn = cdpam.DPAM(model_type=model_type).to(device)
        
        if isinstance(ref_audio, (str, bytes)):
            wav_ref = cdpam.load_audio(ref_audio).to(device)
            wav_out = cdpam.load_audio(cand_audio).to(device)
        else:
            wav_ref = ref_audio.to(device)
            wav_out = cand_audio.to(device)

        dist = loss_fn.forward(wav_ref, wav_out)
        return dist.item() if hasattr(dist, "item") else float(dist)

    except Exception as e:
        return f"CDPAM Error: {str(e)}"
