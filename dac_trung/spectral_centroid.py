import librosa
import numpy as np

def spectral_centroid_function(y, sr):
    """
    Trích xuất Spectral Centroid (độ sáng phổ) trung bình.
    Acoustic property: tần số trọng tâm năng lượng phổ — đo độ sáng của âm thanh.
    Vai trò: chủ yếu là DISCRIMINABILITY — piccolo và flute có centroid cao do âm khu cao
    và sáng; clarinet và saxophone có centroid thấp hơn do năng lượng tập trung ở hoà âm thấp.
    Hạn chế: phụ thuộc mạnh vào cao độ của nốt được ghi âm, có thể gây nhầm lẫn với bản
    sắc nhạc cụ.
    """
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return np.mean(centroid)
