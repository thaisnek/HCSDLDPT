import librosa
import numpy as np

def spectral_flatness_function(y):
    """
    Trích xuất Spectral Flatness (độ phẳng phổ / độ thở) trung bình.
    Acoustic property: tỉ lệ trung bình nhân / trung bình cộng của phổ — đo mức độ âm thanh
    giống noise hay giống tone thuần.
    Vai trò: chủ yếu là DISCRIMINABILITY — flute và piccolo có flatness cao hơn do thành phần
    hơi thở; clarinet và oboe có flatness thấp hơn nhờ cấu trúc hoà âm rõ ràng từ lưỡi gà.
    Hạn chế: rất nhạy cảm với tiếng thở và nhiễu ở đầu/cuối đoạn ghi âm.
    """
    flatness = librosa.feature.spectral_flatness(y=y)
    return np.mean(flatness)
