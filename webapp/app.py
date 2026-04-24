import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tempfile
import pandas as pd
from search.search_similar import search_similar_top5
from processing.extract_features import extract_features

st.set_page_config(page_title="Tìm kiếm nhạc cụ bộ hơi", layout="wide")
st.title("🎷 Hệ Thống Tìm Kiếm Tiếng Nhạc Cụ Bộ Hơi")
st.markdown("Hỗ trợ 5 nhạc cụ: **Piccolo, Flute, Oboe, Clarinet, Saxophone**")

uploaded = st.file_uploader("📂 Tải file âm thanh (.wav)", type=["wav"])

if uploaded:
    # 1. Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(uploaded.read())
        temp_path = f.name

    st.success("✅ Đã tải file lên thành công!")
    st.audio(uploaded, format='audio/wav')

    # 2. Vẽ waveform
    st.subheader("1. Kết Quả Trung Gian: Biểu đồ Sóng Âm (Waveform)")
    y, sr = librosa.load(temp_path, sr=22050)
    fig, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform của file đầu vào")
    ax.set_xlabel("Thời gian (s)")
    ax.set_ylabel("Biên độ")
    st.pyplot(fig)

    # 3. Trích xuất đặc trưng (Kết quả trung gian)
    st.subheader("2. Kết Quả Trung Gian: Vector Đặc Trưng (29 chiều)")
    with st.spinner("Đang phân tích đặc trưng..."):
        features = extract_features(temp_path)
        
        # Format lại để hiển thị đẹp hơn
        feat_dict = {
            "MFCC (13)": [round(x, 4) for x in features[0:13]],
            "Chroma (12)": [round(x, 4) for x in features[13:25]],
            "Spectral Centroid (1)": round(features[25], 4),
            "Spectral Bandwidth (1)": round(features[26], 4),
            "Spectral Flatness (1)": round(features[27], 4),
            "Spectral Rolloff (1)": round(features[28], 4)
        }
        st.json(feat_dict)

    # 4. Tìm kiếm
    st.subheader("3. Kết Quả Tìm Kiếm: Top 5 file giống nhất")
    with st.spinner("Đang so sánh Cosine Similarity với CSDL..."):
        results, all_sims = search_similar_top5(temp_path)
        
        if not results:
            st.warning("CSDL trống! Hãy chạy `python run_extract.py` trước để tạo CSDL.")
        else:
            # Hiển thị biểu đồ phân bố similarity
            st.markdown("**Biểu đồ phân bố độ tương đồng (Similarity Distribution)**")
            fig_sim, ax_sim = plt.subplots(figsize=(10, 2))
            ax_sim.hist(all_sims, bins=30, color='skyblue', edgecolor='black')
            ax_sim.set_title("Phân bố Cosine Similarity của file đầu vào với toàn bộ CSDL")
            ax_sim.set_xlabel("Độ tương đồng (Cosine Similarity)")
            ax_sim.set_ylabel("Số lượng file")
            st.pyplot(fig_sim)
            
            st.markdown("---")
            st.markdown("### 🏆 TOP 5 KẾT QUẢ:")
            
            for i, res in enumerate(results, 1):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown(f"### #{i}")
                    st.metric(label="Độ tương đồng", value=f"{res['similarity']:.4f}")
                    st.markdown(f"**Nhãn:** `{res['label'].upper()}`")
                    
                with col2:
                    st.markdown(f"**File:** `{res['filename']}`")
                    
                    file_path = res['path']
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format='audio/wav')
                    else:
                        st.warning(f"⚠️ Không tìm thấy file gốc tại `{file_path}`")
                
                st.markdown("---")

