import librosa
import numpy as np

def spectral_rolloff_function(y, sr):
    """
    Trích xuất Spectral Rolloff trung bình.
    Acoustic property: tần số mà dưới đó tập trung 85% tổng năng lượng phổ.
    Vai trò: chủ yếu là DISCRIMINABILITY — piccolo và saxophone (giàu overtone) có rolloff cao;
    clarinet ở âm khu thấp có rolloff thấp do năng lượng tập trung ở ít hoà âm hơn.
    Hạn chế: phụ thuộc vào cao độ nốt nhạc, có thể làm lẫn lộn bản sắc nhạc cụ với âm khu.
    """
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    return np.mean(rolloff)
