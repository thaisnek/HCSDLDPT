import os
import sqlite3

import numpy as np


METADATA_DB_PATH = 'database/audio_metadata.db'
FEATURES_DB_PATH = 'database/audio_features.db'
LEGACY_DB_PATH = 'database/audio_database.db'

# Keep this alias for older imports/debug scripts.
DB_PATH = METADATA_DB_PATH

MFCC_COLUMNS = [f'mfcc_{i}' for i in range(13)]
CHROMA_COLUMNS = [f'chroma_{i}' for i in range(12)]
SPECTRAL_COLUMNS = [
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_flatness',
    'spectral_rolloff',
]
FEATURE_COLUMNS = MFCC_COLUMNS + CHROMA_COLUMNS + SPECTRAL_COLUMNS


def _connect(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return sqlite3.connect(db_path)


def _table_exists(cursor, table_name):
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def _metadata_has_legacy_features(cursor):
    cursor.execute("PRAGMA table_info(audio_metadata)")
    columns = {row[1] for row in cursor.fetchall()}
    return all(column in columns for column in FEATURE_COLUMNS)


def create_table():
    metadata_conn = _connect(METADATA_DB_PATH)
    metadata_cursor = metadata_conn.cursor()
    metadata_cursor.execute('''
    CREATE TABLE IF NOT EXISTS audio_metadata (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT NOT NULL,
        label TEXT NOT NULL,
        file_path TEXT NOT NULL
    )
    ''')
    metadata_conn.commit()
    metadata_conn.close()

    features_conn = _connect(FEATURES_DB_PATH)
    features_cursor = features_conn.cursor()
    feature_columns_sql = ',\n        '.join(
        f'{column} REAL NOT NULL' for column in FEATURE_COLUMNS
    )
    features_cursor.execute(f'''
    CREATE TABLE IF NOT EXISTS audio_features (
        metadata_id INTEGER PRIMARY KEY,
        {feature_columns_sql}
    )
    ''')
    features_conn.commit()
    features_conn.close()


def insert_record(filename, label, file_path, features):
    """
    Insert one row into metadata DB and its 29-dimensional vector into features DB.
    """
    features = np.asarray(features, dtype=float).reshape(-1)
    if features.shape[0] != len(FEATURE_COLUMNS):
        raise ValueError(f"Expected {len(FEATURE_COLUMNS)} features, got {features.shape[0]}")

    metadata_conn = _connect(METADATA_DB_PATH)
    features_conn = _connect(FEATURES_DB_PATH)

    try:
        metadata_cursor = metadata_conn.cursor()
        features_cursor = features_conn.cursor()

        metadata_cursor.execute(
            '''
            INSERT INTO audio_metadata (filename, label, file_path)
            VALUES (?, ?, ?)
            ''',
            (filename, label, file_path)
        )
        metadata_id = metadata_cursor.lastrowid

        columns_sql = ', '.join(FEATURE_COLUMNS)
        placeholders = ', '.join(['?'] * (1 + len(FEATURE_COLUMNS)))
        features_cursor.execute(
            f'''
            INSERT INTO audio_features (metadata_id, {columns_sql})
            VALUES ({placeholders})
            ''',
            [metadata_id] + features.tolist()
        )

        metadata_conn.commit()
        features_conn.commit()
    except Exception:
        metadata_conn.rollback()
        features_conn.rollback()
        raise
    finally:
        metadata_conn.close()
        features_conn.close()


def _read_split_records():
    if not os.path.exists(METADATA_DB_PATH) or not os.path.exists(FEATURES_DB_PATH):
        return [], np.array([])

    metadata_conn = _connect(METADATA_DB_PATH)
    features_conn = _connect(FEATURES_DB_PATH)

    try:
        metadata_cursor = metadata_conn.cursor()
        features_cursor = features_conn.cursor()

        if not _table_exists(metadata_cursor, 'audio_metadata'):
            return [], np.array([])
        if not _table_exists(features_cursor, 'audio_features'):
            return [], np.array([])

        metadata_cursor.execute('''
            SELECT id, filename, label, file_path
            FROM audio_metadata
            ORDER BY id
        ''')
        metadata_rows = metadata_cursor.fetchall()

        feature_select = ', '.join(FEATURE_COLUMNS)
        features_cursor.execute(f'''
            SELECT metadata_id, {feature_select}
            FROM audio_features
            ORDER BY metadata_id
        ''')
        feature_rows = features_cursor.fetchall()
    finally:
        metadata_conn.close()
        features_conn.close()

    feature_by_id = {
        row[0]: row[1:]
        for row in feature_rows
    }

    file_info = []
    features_list = []
    for metadata_id, filename, label, file_path in metadata_rows:
        if metadata_id not in feature_by_id:
            continue
        file_info.append({
            'filename': filename,
            'label': label,
            'path': file_path
        })
        features_list.append(feature_by_id[metadata_id])

    if not file_info:
        return [], np.array([])
    return file_info, np.array(features_list, dtype=float)


def _read_combined_db(db_path):
    if not os.path.exists(db_path):
        return [], np.array([])

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if not _table_exists(cursor, 'audio_metadata'):
        conn.close()
        return [], np.array([])

    rows = []
    if _table_exists(cursor, 'audio_features'):
        feature_select = ', '.join(f'f.{column}' for column in FEATURE_COLUMNS)
        cursor.execute(f'''
            SELECT m.id, m.filename, m.label, m.file_path, {feature_select}
            FROM audio_metadata m
            JOIN audio_features f ON f.metadata_id = m.id
            ORDER BY m.id
        ''')
        rows = cursor.fetchall()
    elif _metadata_has_legacy_features(cursor):
        cursor.execute("SELECT * FROM audio_metadata ORDER BY id")
        rows = cursor.fetchall()

    conn.close()

    if not rows:
        return [], np.array([])

    file_info = []
    features_list = []
    for row in rows:
        file_info.append({
            'filename': row[1],
            'label': row[2],
            'path': row[3]
        })
        features_list.append(row[4:])

    return file_info, np.array(features_list, dtype=float)


def get_all_records():
    """
    Return all audio metadata and their feature matrix for search.
    """
    file_info, features = _read_split_records()
    if file_info:
        return file_info, features

    # Fallback for databases created before the two-file split.
    return _read_combined_db(LEGACY_DB_PATH)


def clear_db():
    metadata_conn = _connect(METADATA_DB_PATH)
    metadata_cursor = metadata_conn.cursor()
    metadata_cursor.execute("DROP TABLE IF EXISTS audio_metadata")
    metadata_conn.commit()
    metadata_conn.close()

    features_conn = _connect(FEATURES_DB_PATH)
    features_cursor = features_conn.cursor()
    features_cursor.execute("DROP TABLE IF EXISTS audio_features")
    features_conn.commit()
    features_conn.close()
