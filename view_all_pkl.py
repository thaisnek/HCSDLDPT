import pickle
import os

def load_and_print_pkl(file_name):
    file_path = os.path.join('database', file_name)
    print(f"\n{'='*50}")
    print(f"DANG DOC FILE: {file_name}")
    print(f"{'='*50}")
    
    if not os.path.exists(file_path):
        print(f"[X] Loi: Khong tim thay file {file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"> Kieu du lieu: {type(data)}\n")
        
        print("> Noi dung / Thong so:")
        print(data)
        
        if isinstance(data, dict):
            print("\n> Chi tiet cac key-value trong Dictionary:")
            for k, v in data.items():
                print(f"  - {k}: {v}")
                
    except Exception as e:
        print(f"[X] Co loi khi doc file {file_name}: {e}")

if __name__ == '__main__':
    pkl_files = [
        'scaler.pkl',
        'index_tree.pkl',
        'scalar_ranges.pkl'
    ]
    
    for f in pkl_files:
        load_and_print_pkl(f)
    
    print(f"\n{'='*50}")
    print("HOAN TAT DOC 3 FILE PKL")
