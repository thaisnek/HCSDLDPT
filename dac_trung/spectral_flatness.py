import librosa
import numpy as np

def spectral_flatness_function(y):
    """
    Trích xuất Spectral Flatness (độ phẳng phổ / độ thở) trung bình.
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    return np.mean(flatness)
