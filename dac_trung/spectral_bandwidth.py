import librosa
import numpy as np

def spectral_bandwidth_function(y, sr):
    """
    Trích xuất Spectral Bandwidth (độ rộng phổ) trung bình.
    Acoustic property: độ phân tán năng lượng phổ xung quanh centroid.
    Vai trò: chủ yếu là DISCRIMINABILITY — nhạc cụ lưỡi gà (clarinet, saxophone) có chuỗi
    hoà âm phong phú nên bandwidth rộng hơn; flute và piccolo với âm thuần hơn có bandwidth
    hẹp hơn.
    Hạn chế: tiếng thở và nhiễu thu âm có thể làm tăng bandwidth bất kể nhạc cụ là gì.
    """
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    return np.mean(bandwidth)
