import sys
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, normalize
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

    # 3. Load Index & Scaler & Scalar Ranges
    if not os.path.exists('database/scaler.pkl') or not os.path.exists('database/index_tree.pkl'):
        print("WARNING: Missing index files. Please run `python run_extract.py` first.", file=sys.stderr)
        return [], None
        
    with open('database/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('database/index_tree.pkl', 'rb') as f:
        tree = pickle.load(f)
    with open('database/scalar_ranges.pkl', 'rb') as f:
        scalar_ranges = pickle.load(f)

    # 4. Chuẩn hóa đặc trưng query
    input_feat_scaled = scaler.transform(input_feat)
    input_feat_l2 = normalize(input_feat_scaled, norm='l2')

    # 5. Query KDTree
    # K=10 để dự phòng self-match
    k_neighbors = min(10, len(file_info))
    dists, indices = tree.query(input_feat_l2, k=k_neighbors)
    dists = dists[0]
    indices = indices[0]

    # Chuyển đổi Euclidean sang Cosine Similarity
    sims_k = 1 - (dists ** 2) / 2

    # Lọc self-match
    eligible_mask = []
    for idx in indices:
        if file_info[idx]['path'] == file_path:
            eligible_mask.append(False)
        else:
            eligible_mask.append(True)
            
    top_idxs = indices[eligible_mask][:5]
    top_sims = sims_k[eligible_mask][:5]

    def _scalar_sim(query_val, result_val, min_val, max_val):
        if max_val == min_val:
            return 1.0 if query_val == result_val else 0.0
        return max(0.0, min(1.0, 1.0 - abs(query_val - result_val) / (max_val - min_val)))

    results = []
    for rank_i, i in enumerate(top_idxs):
        q = input_feat_scaled[0]
        # Lấy raw feature từ DB và scale nó để tính breakdown
        r_raw = db_features[i].reshape(1, -1)
        r = scaler.transform(r_raw)[0]

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
            'similarity': top_sims[rank_i],
            'feature_breakdown': breakdown
        })

    # Trả về top 5. Không tính sims cho toàn bộ mảng nữa vì ta đang dùng Indexed KDTree O(log N)
    return results, None
