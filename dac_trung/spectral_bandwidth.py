import librosa
import numpy as np

def spectral_bandwidth_function(y, sr):
    """
    Trích xuất Spectral Bandwidth (độ rộng phổ) trung bình.
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(bandwidth)
