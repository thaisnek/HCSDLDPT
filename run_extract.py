import os
import pickle
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import KDTree
from processing.extract_features import extract_features
from database.db_manager import create_table, insert_record, clear_db, get_all_records

def run_extraction():
    print("Bat dau xu ly du lieu...")
    
    # Reset DB
    clear_db()
    create_table()
    
    dataset_dir = 'dataset'
    count = 0
    
    for root, _, files in os.walk(dataset_dir):
        for f in files:
            if f.endswith('.wav'):
                file_path = os.path.join(root, f)
                # Ten thu muc cha lam nhan
                label = os.path.basename(root)
                
                try:
                    # Trich xuat dac trung
                    features = extract_features(file_path)
                    
                    # Luu vao CSDL
                    insert_record(f, label, file_path, features)
                    count += 1
                    print(f"Da xu ly [{count}]: {f} ({label})")
                except Exception as e:
                    print(f"Loi file {f}: {e}")
                    
    print(f"\nHoan tat! Da trich xuat va luu {count} file vao co so du lieu SQLite.")
    
    print("\nBat dau xay dung chi muc (Index) KD-Tree...")
    build_index()
    print("Hoan tat xay dung chi muc!")

def build_index():
    # Lấy toàn bộ dữ liệu từ DB
    file_info, db_features = get_all_records()
    if len(file_info) == 0:
        print("Khong co du lieu de xay dung chi muc.")
        return

    # 1. Standard Scaler
    scaler = StandardScaler()
    db_features_scaled = scaler.fit_transform(db_features)
    
    # 2. L2 Normalization cho KDTree (để Euclidean tương đương Cosine)
    db_features_l2 = normalize(db_features_scaled, norm='l2')
    
    # 3. Xây dựng KDTree
    tree = KDTree(db_features_l2, metric='euclidean')
    
    # 4. Lưu mô hình ra file
    with open('database/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    with open('database/index_tree.pkl', 'wb') as f:
        pickle.dump(tree, f)
        
    # 5. Lưu min/max của các scalar features để dùng cho breakdown
    scalar_ranges = {}
    for col, name in [(25, 'Spectral Centroid'), (26, 'Spectral Bandwidth'),
                      (27, 'Spectral Flatness'), (28, 'Spectral Rolloff')]:
        scalar_ranges[name] = (db_features_scaled[:, col].min(),
                               db_features_scaled[:, col].max())
    with open('database/scalar_ranges.pkl', 'wb') as f:
        pickle.dump(scalar_ranges, f)

if __name__ == '__main__':
    # Tạo thư mục con nếu chưa có
    if not os.path.exists('dataset/piccolo'): os.makedirs('dataset/piccolo')
    if not os.path.exists('dataset/flute'): os.makedirs('dataset/flute')
    if not os.path.exists('dataset/oboe'): os.makedirs('dataset/oboe')
    if not os.path.exists('dataset/clarinet'): os.makedirs('dataset/clarinet')
    if not os.path.exists('dataset/saxophone'): os.makedirs('dataset/saxophone')
    if not os.path.exists('database'): os.makedirs('database')
    
    run_extraction()
