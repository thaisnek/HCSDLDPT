import librosa
import numpy as np

from dac_trung.mfcc import mfcc_function
from dac_trung.chroma import chroma_function
from dac_trung.spectral_centroid import spectral_centroid_function
from dac_trung.spectral_bandwidth import spectral_bandwidth_function
from dac_trung.spectral_flatness import spectral_flatness_function
from dac_trung.spectral_rolloff import spectral_rolloff_function

def extract_features(file_path):
    """
    Load file .wav và trích xuất vector 29 chiều.
    """
    # Load file âm thanh, sr=22050 để chuẩn hóa, chuyển về mono
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    
    # Tính toán đặc trưng
    mfcc_feat = mfcc_function(y, sr)                  # 13
    chroma_feat = chroma_function(y, sr)              # 12
    centroid_feat = spectral_centroid_function(y, sr) # 1
    bandwidth_feat = spectral_bandwidth_function(y, sr)# 1
    flatness_feat = spectral_flatness_function(y)     # 1
    rolloff_feat = spectral_rolloff_function(y, sr)   # 1
    
    # Ghép nối thành vector 29 chiều
    feature_vector = np.concatenate([
        mfcc_feat,
        chroma_feat,
        [centroid_feat],
        [bandwidth_feat],
        [flatness_feat],
        [rolloff_feat]
    ])
    
    return feature_vector
