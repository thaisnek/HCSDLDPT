import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from processing.extract_features import extract_features
from database.db_manager import get_all_records

def search_similar_top5(file_path, input_feat=None):
    """
    Tìm kiếm 5 file giống nhất từ database.
    """
    # 1. Trích xuất đặc trưng file input (dùng lại nếu đã có sẵn)
    if input_feat is None:
        input_feat = extract_features(file_path).reshape(1, -1)
    else:
        input_feat = input_feat.reshape(1, -1)
    
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

    # Precompute scalar feature range across all DB rows (scaled)
    scalar_ranges = {}
    for col, name in [(25, 'Spectral Centroid'), (26, 'Spectral Bandwidth'),
                      (27, 'Spectral Flatness'), (28, 'Spectral Rolloff')]:
        scalar_ranges[name] = (db_features_scaled[:, col].min(),
                               db_features_scaled[:, col].max())

    def _scalar_sim(query_val, result_val, min_val, max_val):
        if max_val == min_val:
            return 1.0 if query_val == result_val else 0.0
        return max(0.0, min(1.0, 1.0 - abs(query_val - result_val) / (max_val - min_val)))

    results = []
    for i in top_idxs:
        q = input_feat_scaled[0]
        r = db_features_scaled[i]

        mfcc_raw = cosine_similarity(q[0:13].reshape(1, -1), r[0:13].reshape(1, -1))[0][0]
        chroma_raw = cosine_similarity(q[13:25].reshape(1, -1), r[13:25].reshape(1, -1))[0][0]

        breakdown = {
            'MFCC': max(0.0, min(1.0, float(mfcc_raw))),
            'Chroma': max(0.0, min(1.0, float(chroma_raw))),
            'Spectral Centroid': _scalar_sim(q[25], r[25], *scalar_ranges['Spectral Centroid']),
            'Spectral Bandwidth': _scalar_sim(q[26], r[26], *scalar_ranges['Spectral Bandwidth']),
            'Spectral Flatness': _scalar_sim(q[27], r[27], *scalar_ranges['Spectral Flatness']),
            'Spectral Rolloff': _scalar_sim(q[28], r[28], *scalar_ranges['Spectral Rolloff']),
        }

        results.append({
            'filename': file_info[i]['filename'],
            'label': file_info[i]['label'],
            'path': file_info[i]['path'],
            'similarity': sims[i],
            'feature_breakdown': breakdown
        })

    # Trả về top 5 và toàn bộ mảng similarities để vẽ biểu đồ
    return results, sims
