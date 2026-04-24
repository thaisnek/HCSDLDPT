import sqlite3
import numpy as np

DB_PATH = 'database/audio_metadata.db'

def create_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Tạo bảng metadata với 29 cột đặc trưng
    # 13 MFCC, 12 Chroma, 1 Centroid, 1 Bandwidth, 1 Flatness, 1 Rolloff = 29
    columns = []
    for i in range(13): columns.append(f'mfcc_{i} REAL')
    for i in range(12): columns.append(f'chroma_{i} REAL')
    columns.extend([
        'spectral_centroid REAL',
        'spectral_bandwidth REAL',
        'spectral_flatness REAL',
        'spectral_rolloff REAL'
    ])
    
    columns_sql = ',\n        '.join(columns)
    
    cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS audio_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        label TEXT NOT NULL,
        file_path TEXT NOT NULL,
        
        {columns_sql}
    )
    ''')
    
    conn.commit()
    conn.close()

def insert_record(filename, label, file_path, features):
    """
    Chèn 1 bản ghi vào CSDL. Features là array numpy 29 chiều.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    placeholders = ', '.join(['?'] * 29)
    query = f'''
    INSERT INTO audio_metadata (
        filename, label, file_path,
        mfcc_0, mfcc_1, mfcc_2, mfcc_3, mfcc_4, mfcc_5, mfcc_6, mfcc_7, mfcc_8, mfcc_9, mfcc_10, mfcc_11, mfcc_12,
        chroma_0, chroma_1, chroma_2, chroma_3, chroma_4, chroma_5, chroma_6, chroma_7, chroma_8, chroma_9, chroma_10, chroma_11,
        spectral_centroid, spectral_bandwidth, spectral_flatness, spectral_rolloff
    ) VALUES (?, ?, ?, {placeholders})
    '''
    
    data = [filename, label, file_path] + features.tolist()
    cursor.execute(query, data)
    
    conn.commit()
    conn.close()

def get_all_records():
    """
    Lấy toàn bộ dữ liệu từ DB để tìm kiếm.
    Trả về: (thông_tin_file, features_matrix)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM audio_metadata")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return [], np.array([])
        
    file_info = []
    features_list = []
    
    for row in rows:
        # row: (id, filename, label, path, mfcc_0...rolloff)
        file_info.append({
            'filename': row[1],
            'label': row[2],
            'path': row[3]
        })
        features_list.append(row[4:]) # Từ cột 4 trở đi là 29 đặc trưng
        
    return file_info, np.array(features_list)

def clear_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS audio_metadata")
    conn.commit()
    conn.close()
