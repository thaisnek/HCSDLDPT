import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from processing.extract_features import extract_features
from database.db_manager import get_all_records

def search_similar_top5(file_path):
    """
    Tìm kiếm 5 file giống nhất từ database.
    """
    # 1. Trích xuất đặc trưng file input
    input_feat = extract_features(file_path).reshape(1, -1)
    
    # 2. Lấy dữ liệu từ DB
    file_info, db_features = get_all_records()
    if len(file_info) == 0:
        return [], []
    if not hasattr(db_features, 'shape') or db_features.ndim != 2 or db_features.shape[1] != 29:
        print("WARNING: db_features shape mismatch, expected (n, 29)", file=sys.stderr)
        return [], []
    if not hasattr(input_feat, 'shape') or input_feat.shape != (1, 29):
        print("WARNING: input_feat shape mismatch, expected (1, 29)", file=sys.stderr)
        return [], []

    # 3. Chuẩn hóa đặc trưng và tính Cosine Similarity
    scaler = StandardScaler()
    db_features_scaled = scaler.fit_transform(db_features)
    input_feat_scaled = scaler.transform(input_feat)
    sims = cosine_similarity(input_feat_scaled, db_features_scaled)[0]

    # 4. Loại self-match bằng mask index (không dựa vào giá trị score)
    self_match_mask = np.zeros(len(sims), dtype=bool)
    for i, info in enumerate(file_info):
        if info['path'] == file_path:
            self_match_mask[i] = True

    eligible_idxs = np.where(~self_match_mask)[0]
    top_idxs = eligible_idxs[sims[eligible_idxs].argsort()[::-1]][:5]

    results = []
    for i in top_idxs:
        results.append({
            'filename': file_info[i]['filename'],
            'label': file_info[i]['label'],
            'path': file_info[i]['path'],
            'similarity': sims[i]
        })

    # Trả về top 5 và toàn bộ mảng similarities để vẽ biểu đồ
    return results, sims
