import librosa
import numpy as np

def spectral_rolloff_function(y, sr):
    """
    Trích xuất Spectral Rolloff trung bình.
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return np.mean(rolloff)
