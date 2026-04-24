import os
from processing.extract_features import extract_features
from database.db_manager import create_table, insert_record, clear_db

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

if __name__ == '__main__':
    # Tạo thư mục con nếu chưa có
    if not os.path.exists('dataset/piccolo'): os.makedirs('dataset/piccolo')
    if not os.path.exists('dataset/flute'): os.makedirs('dataset/flute')
    if not os.path.exists('dataset/oboe'): os.makedirs('dataset/oboe')
    if not os.path.exists('dataset/clarinet'): os.makedirs('dataset/clarinet')
    if not os.path.exists('dataset/saxophone'): os.makedirs('dataset/saxophone')
    if not os.path.exists('database'): os.makedirs('database')
    
    run_extraction()
