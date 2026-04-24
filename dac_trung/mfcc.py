import librosa
import numpy as np

def mfcc_function(y, sr):
    """
    Trích xuất 13 hệ số MFCC trung bình.
    Acoustic property: mô hình hóa đường bao phổ / âm sắc của nhạc cụ qua hệ số cepstral.
    Vai trò: BOTH similarity và discriminability — nhạc cụ đơn lưỡi gà (clarinet, saxophone)
    có độ giàu hoà âm khác biệt so với nhạc cụ thổi miệng (flute, piccolo), giúp phân biệt
    hai nhóm; đồng thời các nhạc cụ cùng cao độ có đường bao phổ tương tự, hỗ trợ tìm kiếm
    sự tương đồng.
    Hạn chế: lấy trung bình theo thời gian làm mất thông tin về diễn tấu, rung và attack.
    """
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs, axis=1)
