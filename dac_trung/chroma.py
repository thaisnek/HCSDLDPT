import librosa
import numpy as np

def chroma_function(y, sr):
    """
    Trích xuất 12 giá trị chroma trung bình.
    Acoustic property: phân bố năng lượng theo 12 lớp cao độ của thang âm bình quân.
    Vai trò: chủ yếu là SIMILARITY — nhạc cụ khác nhau cùng chơi một cao độ tạo ra vector
    chroma gần giống nhau, hỗ trợ khớp nội dung giai điệu nhưng yếu trong việc phân biệt
    nhạc cụ.
    Hạn chế: không đặc trưng cho nhạc cụ — flute và clarinet cùng nốt cho chroma gần như
    đồng nhất.
    """
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return np.mean(chroma, axis=1)
