import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
        
    # 3. Tính Cosine Similarity
    sims = cosine_similarity(input_feat, db_features)[0]
    
    # 4. Sắp xếp kết quả giảm dần và lấy top 5
    top_idxs = sims.argsort()[-5:][::-1]
    
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
