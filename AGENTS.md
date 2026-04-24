# AGENTS.md

Hướng dẫn hành vi cho AI agent làm việc trên dự án **Hệ Thống Tìm Kiếm Tiếng Nhạc Cụ Bộ Hơi**.

---

## 1. Hiểu Dự Án Trước Khi Làm

Đây là hệ thống nhận dạng và tìm kiếm âm thanh nhạc cụ bộ hơi (Piccolo, Flute, Oboe, Clarinet, Saxophone) bằng cách:
1. Trích xuất vector đặc trưng 29 chiều từ file `.wav` (MFCC × 13, Chroma × 12, Spectral Centroid, Bandwidth, Flatness, Rolloff)
2. Lưu vector vào SQLite (`database/audio_metadata.db`)
3. Tìm kiếm Top 5 file giống nhất bằng Cosine Similarity
4. Hiển thị kết quả qua giao diện Streamlit (`webapp/app.py`)

**Cấu trúc thư mục thực tế:**
```
HCSDLDPT/
├── dac_trung/                  # Các hàm trích xuất đặc trưng riêng lẻ
│   ├── __init__.py
│   ├── mfcc.py                 # 13 hệ số MFCC trung bình
│   ├── chroma.py               # 12 giá trị Chroma trung bình
│   ├── spectral_centroid.py    # Spectral Centroid trung bình
│   ├── spectral_bandwidth.py   # Spectral Bandwidth trung bình
│   ├── spectral_flatness.py    # Spectral Flatness trung bình
│   └── spectral_rolloff.py     # Spectral Rolloff trung bình
│
├── database/
│   └── db_manager.py           # Quản lý SQLite (create, insert, get, clear)
│
├── dataset/                    # ~500+ file .wav — KHÔNG chỉnh sửa thủ công
│   ├── clarinet/               # ~100+ file .wav
│   ├── flute/                  # ~100+ file .wav
│   ├── oboe/                   # ~100+ file .wav
│   ├── piccolo/                # ~100+ file .wav
│   └── saxophone/              # ~100+ file .wav
│
├── processing/
│   └── extract_features.py     # Gộp tất cả đặc trưng → vector 29 chiều
│
├── search/
│   └── search_similar.py       # Tính Cosine Similarity, trả về Top 5
│
├── webapp/
│   └── app.py                  # Giao diện Streamlit
│
├── requirements.txt
└── run_extract.py              # Script trích xuất toàn bộ dataset → DB
```

---

## 2. Lệnh Chạy Chuẩn

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Trích xuất đặc trưng toàn bộ dataset và lưu vào DB
python run_extract.py

# Chạy giao diện tìm kiếm
streamlit run webapp/app.py

# Kiểm tra nhanh DB có dữ liệu chưa
python -c "from database.db_manager import get_all_records; info, feats = get_all_records(); print(f'DB có {len(info)} bản ghi, vector shape: {feats.shape}')"
```

Nếu lệnh kiểm tra nhanh DB lỗi do tên hàm thay đổi, phải đọc `database/db_manager.py` để dùng đúng hàm hiện có. Không tự tạo thêm hàm mới chỉ để làm lệnh verify nếu không cần thiết.

> **Test/Lint:** Hiện tại dự án chưa có test/lint command chính thức. Nếu thêm test/lint sau này, phải cập nhật Mục 2 này và Definition of Done ở Mục 9.

---

## 3. Suy Nghĩ Trước Khi Code

**Không giả định. Không ẩn sự mơ hồ. Nêu rõ đánh đổi.**

Trước khi thực hiện bất kỳ thay đổi nào:
- Nêu rõ giả định. Nếu không chắc, hỏi lại.
- Nếu có nhiều cách hiểu, liệt kê tất cả — không tự chọn im lặng.
- Nếu tồn tại cách đơn giản hơn, hãy nói ra.

**Khi nào phải dừng và hỏi:**
Chỉ dừng và hỏi người dùng khi vấn đề có thể dẫn đến một trong các hệ quả sau:
- Sai hoặc mất dữ liệu trong `dataset/` hoặc DB
- Thay đổi schema DB không tương thích
- Xóa hoặc reset DB
- Thay đổi thuật toán tìm kiếm
- Mở rộng phạm vi bài toán ngoài yêu cầu

Nếu chỉ là chi tiết implementation nhỏ, hãy chọn phương án đơn giản nhất, nêu giả định ngắn gọn và tiếp tục.

---

## 4. Ưu Tiên Đơn Giản

**Code tối thiểu giải quyết đúng vấn đề. Không suy đoán thêm.**

- Không thêm tính năng ngoài yêu cầu.
- Không tạo abstraction cho code dùng một lần.
- Không thêm "linh hoạt" hay "cấu hình" nếu không được yêu cầu.
- Không over-engineer error handling. Tuy nhiên phải xử lý các lỗi thực tế có khả năng xảy ra:
  - File audio không đọc được hoặc định dạng lạ
  - Feature extraction trả về `NaN` / `None`
  - Insert DB lỗi
  - Dataset rỗng hoặc thiếu thư mục nhạc cụ
- Nếu viết 200 dòng mà có thể viết 50 dòng, hãy viết lại.

---

## 5. Thay Đổi Phẫu Thuật

**Chỉ chạm vào những gì cần thiết. Dọn sạch đúng phần mình tạo ra.**

Khi sửa code hiện có:
- Không "cải thiện" code, comment, hay format xung quanh nếu không liên quan.
- Không refactor những gì đang hoạt động tốt.
- Giữ nguyên phong cách code hiện tại, kể cả khi bạn làm khác.
- Nếu phát hiện dead code không liên quan, **đề cập** — không tự xóa.

Khi thay đổi tạo ra code mồ côi:
- Xóa import/biến/hàm mà **chính thay đổi của bạn** khiến chúng không dùng đến.
- Không xóa dead code có sẵn trừ khi được yêu cầu.

**Kiểm tra:** Mỗi dòng thay đổi phải truy ngược trực tiếp về yêu cầu của người dùng.

---

## 6. Thực Thi Theo Mục Tiêu

**Định nghĩa tiêu chí thành công. Lặp cho đến khi xác minh được.**

Chuyển đổi nhiệm vụ thành mục tiêu kiểm chứng được:
- "Thêm đặc trưng mới" → "Cập nhật `dac_trung/`, `processing/extract_features.py`, schema DB, rồi chạy `run_extract.py` để xác minh"
- "Sửa lỗi tìm kiếm" → "Tái hiện lỗi, sửa, chạy lại để xác nhận"
- "Cải thiện UI" → "Xác nhận `streamlit run webapp/app.py` chạy được trước và sau"

Với nhiệm vụ nhiều bước, nêu kế hoạch ngắn gọn:
```
1. [Bước] → xác minh: [kiểm tra]
2. [Bước] → xác minh: [kiểm tra]
3. [Bước] → xác minh: [kiểm tra]
```

---

## 7. Quy Tắc Riêng Cho Dự Án Này

### Dataset (`dataset/`)
- Chứa hơn 500 file `.wav` chia vào 5 thư mục con theo nhãn loại nhạc cụ.
- **Không chỉnh sửa, xóa, hay thêm file** trong `dataset/` trừ khi được yêu cầu rõ ràng.
- Nhãn (label) được xác định bởi tên thư mục con — không hard-code nhãn ở nơi khác.

### Thêm hoặc sửa đặc trưng (`dac_trung/`)
- Mỗi đặc trưng nằm trong file riêng tại `dac_trung/`.
Sau khi thêm đặc trưng mới, PHẢI cập nhật đồng thời:
1. `dac_trung/` — thêm file/hàm trích xuất đặc trưng mới
2. `processing/extract_features.py` — thêm import và ghép vào vector
3. `database/db_manager.py` — thêm cột mới vào schema và câu lệnh INSERT/SELECT
4. `search/search_similar.py` — đảm bảo vector query và vector DB có cùng số chiều
5. `webapp/app.py` — nếu UI hiển thị vector hoặc tên đặc trưng

Sau khi sửa, PHẢI xác minh vector shape mới khớp giữa query và DB.

### Database (`database/`)
- Không tự xóa hoặc reset DB (`clear_db()`) trừ khi người dùng yêu cầu rõ ràng.
- Thay đổi schema bảng phải đi kèm tái tạo DB bằng `run_extract.py`.
- File DB nằm tại `database/audio_metadata.db`.

### Streamlit UI (`webapp/app.py`)
- File duy nhất cho UI. Không tách thêm file UI trừ khi được yêu cầu.
- Không thêm widget hoặc bước hiển thị mới trừ khi được yêu cầu.

### Tìm kiếm (`search/search_similar.py`)
- Thuật toán mặc định là **Cosine Similarity**. Không thay đổi thuật toán trừ khi được yêu cầu.
- Top 5 kết quả phải luôn được sắp xếp **giảm dần** theo Cosine Similarity.

---

## 8. Ràng Buộc Kỹ Thuật Bắt Buộc

### Instrument labels — LOCKED

Các nhãn hợp lệ duy nhất:

- `clarinet`
- `flute`
- `oboe`
- `piccolo`
- `saxophone`

Tên nhãn PHẢI lấy từ tên thư mục con trong `dataset/`.
Không tự tạo nhãn mới, không đổi chữ hoa/thường, không Việt hóa tên nhãn.

### Vector đặc trưng — LOCKED

Vector đặc trưng hiện tại có đúng **29 chiều**, theo thứ tự cố định sau:

1. `mfcc_1` đến `mfcc_13`
2. `chroma_1` đến `chroma_12`
3. `spectral_centroid`
4. `spectral_bandwidth`
5. `spectral_flatness`
6. `spectral_rolloff`

Tuyệt đối không tự ý đổi thứ tự vector.
Vector của file query và vector trong DB phải có cùng thứ tự, cùng số chiều, cùng cách tính.

### Normalization policy

Hiện tại hệ thống tính Cosine Similarity trực tiếp trên vector đặc trưng thô 29 chiều.
Không thêm StandardScaler, MinMaxScaler, L2 normalization hoặc bất kỳ bước chuẩn hóa nào nếu người dùng không yêu cầu rõ.

Nếu sau này thêm normalization, PHẢI áp dụng đồng nhất cho cả:
- Vector lưu trong DB
- Vector của file query upload vào Streamlit

### Schema DB hiện tại

Database file: `database/audio_metadata.db`

Bảng chính lưu metadata âm thanh và vector đặc trưng. Schema logic tối thiểu phải gồm:

- `id` hoặc khóa chính tương đương
- `file_path`
- `instrument`
- 29 cột đặc trưng theo đúng thứ tự vector:
  - `mfcc_1` ... `mfcc_13`
  - `chroma_1` ... `chroma_12`
  - `spectral_centroid`
  - `spectral_bandwidth`
  - `spectral_flatness`
  - `spectral_rolloff`

Tên bảng hiện tại: `audio_metadata`.
Không tự tạo bảng mới với tên khác nếu bảng hiện tại đã tồn tại và đang được `search/search_similar.py` hoặc `webapp/app.py` sử dụng.
Không đổi tên bảng/cột nếu code hiện tại đang phụ thuộc vào tên đó.
Nếu bắt buộc đổi schema, PHẢI cập nhật đồng thời `database/db_manager.py`, `processing/extract_features.py`, `search/search_similar.py`, và `webapp/app.py` nếu có liên quan.

### Self-match policy

Nếu file query trùng đường dẫn với một file trong DB, mặc định loại chính file đó khỏi Top 5 để kết quả demo có ý nghĩa hơn.
Nếu query là file upload tạm thời không có trong DB, tìm kiếm bình thường.
Self-match chỉ bắt buộc loại khi xác định được trùng file_path. Không cần thêm hash/audio fingerprinting nếu người dùng không yêu cầu.

### Dependencies

- Không thêm dependency mới nếu có thể giải quyết bằng thư viện hiện có.
- Không thêm PyTorch, TensorFlow, Keras, FAISS hoặc framework nặng nếu người dùng không yêu cầu.
- Nếu bắt buộc thêm dependency, PHẢI:
  1. Giải thích lý do
  2. Cập nhật `requirements.txt`
  3. Đảm bảo lệnh `pip install -r requirements.txt` vẫn chạy được

### Generated files

Các file sinh ra trong quá trình chạy:
- `database/audio_metadata.db`
- cache Python `__pycache__/`
- file tạm của Streamlit

Không chỉnh sửa thủ công file DB bằng tay.
Không commit cache hoặc file tạm nếu có Git.

---

## 9. Definition of Done

Một thay đổi chỉ được coi là **hoàn thành** khi:

- [ ] Code chạy được bằng lệnh liên quan ở Mục 2.
- [ ] Không chỉnh sửa `dataset/` nếu không được yêu cầu.
- [ ] Không reset database nếu không được yêu cầu.
- [ ] Không thêm dependency mới nếu không cần thiết.
- [ ] Nếu thay đổi đặc trưng, đã cập nhật đủ:
  - `dac_trung/`
  - `processing/extract_features.py`
  - `database/db_manager.py`
  - `search/search_similar.py` nếu vector search bị ảnh hưởng
  - `webapp/app.py` nếu UI hiển thị vector/feature name
- [ ] Nếu thay đổi tìm kiếm, Top 5 vẫn được sắp xếp giảm dần theo Cosine Similarity.
- [ ] Nếu thay đổi UI, `streamlit run webapp/app.py` chạy được không lỗi.
- [ ] Sau khi chạy `python run_extract.py`, DB có khoảng 500+ bản ghi hợp lệ.
- [ ] Vector feature có đúng shape `(n, 29)` nếu chưa thay đổi đặc trưng.
- [ ] File query upload vào Streamlit tạo ra vector đúng 29 chiều.
- [ ] Không có `NaN`, `None`, hoặc vector rỗng trong DB.
- [ ] Nếu query trùng file trong DB, self-match được xử lý đúng theo Self-match policy ở Mục 8.

---

## 10. Luồng Làm Việc Chuẩn

```
dataset/<label>/*.wav   (500+ file, đã có sẵn)
        ↓
python run_extract.py   → trích xuất đặc trưng & lưu vào DB
        ↓
database/audio_metadata.db  (SQLite, 29 cột đặc trưng)
        ↓
streamlit run webapp/app.py
        ↓
Upload file .wav → Xem waveform → Xem vector đặc trưng → Top 5 kết quả
```

---

**Những hướng dẫn này hiệu quả khi:** diff gọn hơn, ít rewrite do overcomplicate, câu hỏi làm rõ xuất hiện trước khi implement chứ không phải sau khi mắc lỗi.
