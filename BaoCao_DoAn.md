# BÁO CÁO ĐỒ ÁN MÔN HỌC: HỆ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN
**Đề tài:** Xây dựng hệ CSDL lưu trữ và tìm kiếm tiếng nhạc cụ thuộc bộ hơi

---

## PHẦN 1. GIỚI THIỆU CHUNG
Cùng với sự phát triển của lượng dữ liệu đa phương tiện khổng lồ, việc tìm kiếm dựa trên nội dung (Content-Based Retrieval) trở thành bài toán nòng cốt. Trong giới hạn của đồ án môn học "Hệ cơ sở dữ liệu đa phương tiện", nhóm chúng em thực hiện đề tài **"Xây dựng hệ CSDL lưu trữ và tìm kiếm tiếng nhạc cụ thuộc bộ hơi"**.

**Mục tiêu đồ án:**
- Xây dựng kho dữ liệu âm thanh đặc thù cho 5 loại nhạc cụ bộ hơi.
- Trích xuất siêu dữ liệu (Metadata) dưới dạng các vector đặc trưng âm học thay vì dùng tín hiệu thô.
- Thiết kế hệ quản trị CSDL kết hợp Cấu trúc chỉ số hóa đa chiều (KD-Tree) để tối ưu hóa thời gian và độ chính xác của thao tác tìm kiếm bằng thuật toán Cosine Similarity.

**Công nghệ sử dụng:**
- **Ngôn ngữ:** Python
- **Xử lý tín hiệu số:** `librosa`
- **Cơ sở dữ liệu:** `SQLite` (Lưu trữ thô), `pickle` (Lưu trữ Index Tree)
- **Học máy & Toán học:** `scikit-learn`, `numpy`
- **Giao diện:** `Streamlit`

---

## PHẦN 2. XÂY DỰNG VÀ THU THẬP DỮ LIỆU
Bộ dữ liệu được xây dựng tập trung vào các nhạc cụ đơn tấu thuộc bộ hơi. 

**1. Thông tin bộ dữ liệu:**
- **Số lượng:** 634 files âm thanh định dạng chuẩn `.wav`.
- **Phân lớp (Labels):** 5 loại nhạc cụ bộ hơi bao gồm `Clarinet`, `Flute`, `Oboe`, `Piccolo`, `Saxophone`.
- **Tần số lấy mẫu (Sample Rate):** Chuẩn hóa về $22050$ Hz khi nạp vào hệ thống để đảm bảo sự đồng nhất trong khâu trích xuất vector.

**2. Tổ chức lưu trữ trên ổ đĩa:**
Bộ dữ liệu được tổ chức theo cấu trúc thư mục dạng hình cây, trong đó tên thư mục con (ví dụ: `dataset/flute/`) đóng vai trò là nhãn (label) tự động khi tiến hành crawl dữ liệu vào Database.

---

## PHẦN 3. XÂY DỰNG BỘ THUỘC TÍNH ĐỂ NHẬN DIỆN TIẾNG NHẠC CỤ
Dựa trên kiến thức của **Bài giảng 10: Chỉ số hóa và truy vấn dữ liệu âm thanh**, hệ thống không lưu trữ tín hiệu âm thanh thô để so sánh (vì kích thước quá lớn và không hiệu quả). Thay vào đó, nhóm thiết kế một vector siêu dữ liệu (metadata) **29 chiều** cho mỗi file âm thanh để đại diện cho nội dung.

**Cấu trúc vector 29 chiều bao gồm:**
1. **13 Hệ số MFCC (Mel-frequency cepstral coefficients):**
   - *Giá trị thông tin:* Mô phỏng lại cách tai người cảm nhận âm thanh và đặc biệt phản ánh **hình dáng vật lý của ống ca (ống kèn)**. 
   - *Lý do chọn:* Đây là thuộc tính tối quan trọng để tìm ra **sự khác nhau** giữa các nhạc cụ. Tiếng Flute và Saxophone thổi cùng một nốt nhạc sẽ có MFCC hoàn toàn khác nhau do cấu tạo chất liệu và hình dáng kèn khác nhau.
2. **12 Hệ số Chroma:**
   - *Giá trị thông tin:* Đại diện cho mức năng lượng của 12 cung bậc nốt nhạc (C, C#, D, D#,...).
   - *Lý do chọn:* Giúp hệ thống tìm kiếm được **sự tương đồng** về mặt giai điệu và cao độ giữa file truy vấn và các file trong DB.
3. **4 Đặc trưng phổ (Spectral Features):**
   - *Spectral Centroid (Tâm phổ):* Đại diện cho "độ sáng" của âm thanh.
   - *Spectral Bandwidth (Độ rộng phổ):* Sự phân bố năng lượng quanh tâm phổ.
   - *Spectral Flatness (Độ phẳng phổ):* Phân biệt giữa âm thanh có tính chu kỳ (tiếng kèn) và âm thanh nhiễu (tiếng lấy hơi).
   - *Spectral Rolloff:* Đặc trưng cho dải tần số cao.

---

## PHẦN 4. XÂY DỰNG HỆ THỐNG TÌM KIẾM TIẾNG NHẠC CỤ THUỘC BỘ HƠI
Hệ thống truy vấn được thiết kế kiến trúc chuẩn mực dựa trên lý thuyết của **Bài 4 (Kiến trúc CSDL ĐPT)** và **Bài 8 (Truy vấn không gian vector)**.

### 4.1. Kiến trúc CSDL và Xây dựng Chỉ mục (Index)
Theo nguyên tắc toàn vẹn dữ liệu, hệ thống tách biệt quá trình lưu trữ và quá trình đánh chỉ mục:
- **Tầng lưu trữ thô (SQLite):** Lưu trữ 29 thuộc tính âm thanh ở dạng giá trị nguyên bản nhằm mục đích kiểm toán và bảo toàn dữ liệu.
- **Tầng Chỉ số hóa (Index Building):** Vì 29 thuộc tính có thang đo rất lệch nhau (Chroma chạy từ 0-1, Rolloff lên tới hàng ngàn), hệ thống áp dụng `StandardScaler` để đưa dữ liệu về cùng thang đo chuẩn. Sau đó tiến hành chuẩn hóa `L2-Normalization`. Cuối cùng, nạp các vector này vào cấu trúc cây đa chiều **KD-Tree** và lưu thành file nhị phân `index_tree.pkl`.

### 4.2. Cơ chế tìm kiếm và Sơ đồ khối
Sơ đồ thuật toán truy vấn được diễn ra như sau:
`File Upload` $\rightarrow$ `Trích xuất 29 chiều (Raw)` $\rightarrow$ `Áp dụng StandardScaler` $\rightarrow$ `Chuẩn hóa L2` $\rightarrow$ `Truy vấn KD-Tree` $\rightarrow$ `Áp dụng Self-match policy` $\rightarrow$ `Top 5 kết quả`.

**Mẹo toán học (Mathematical Trick):** Hệ thống được yêu cầu sử dụng độ đo **Cosine Similarity**. Tuy nhiên, các thuật toán không gian cây như KD-Tree thường chỉ hỗ trợ **Euclidean Distance**. Nhóm đã khắc phục điều này bằng cách chuẩn hóa các vector có độ dài bằng $1$ (L2 Normalize), sau đó dùng cây KD-Tree để tìm khoảng cách Euclidean nhỏ nhất, và chuyển đổi ngược ra giá trị Cosine bằng phương trình đại số:
$Cosine = 1 - \frac{Euclidean^2}{2}$

---

## PHẦN 5. DEMO HỆ THỐNG VÀ ĐÁNH GIÁ KẾT QUẢ

**1. Giao diện Demo:**
Hệ thống sử dụng Streamlit cung cấp giao diện Web trực quan. Khi người dùng tải file `.wav` lên, hệ thống sẽ thực thi quy trình sau:
- In ra Kết quả trung gian 1: Biểu đồ dạng sóng âm (Waveform) của file input.
- In ra Kết quả trung gian 2: Dữ liệu JSON hiển thị vector 29 chiều vừa được trích xuất (Metadata nội dung).
- Trả về màn hình TOP 5 file âm thanh có Cosine Similarity cao nhất từ CSDL. Cho phép phát trực tiếp âm thanh kết quả trên trình duyệt.
- Vẽ biểu đồ thanh ngang (Feature Breakdown) lý giải việc file kết quả giống với file truy vấn chủ yếu là nhờ đặc trưng nào (MFCC hay Chroma...).

**2. Đánh giá kết quả đạt được:**
- **Về tính chính xác:** Nhờ có bước Chuẩn hóa (Scaler), thuật toán Cosine đánh giá công bằng cả 29 chiều, giải quyết triệt để vấn đề "thuộc tính lớn đè bẹp thuộc tính nhỏ". Kết quả Top 5 thường rơi vào đúng loại kèn của file đầu vào.
- **Về hiệu năng:** Nhờ cấu trúc KD-Tree, thời gian tìm kiếm Top 5 từ 600+ files mất chưa tới $0.05$ giây, giảm độ phức tạp quét tuyến tính $O(N)$ xuống còn độ phức tạp lô-ga-rít $O(\log N)$.
- **Về logic nghiệp vụ:** Ứng dụng tích hợp hệ thống *Self-match policy*, tự động phát hiện và loại bỏ chính file âm thanh đó ra khỏi kết quả Top 5 nếu file đầu vào trùng lặp với CSDL, giúp demo mang tính thực tiễn cao hơn.

---

## TÀI LIỆU THAM KHẢO
1. *Tài liệu bài giảng Hệ cơ sở dữ liệu đa phương tiện* (Chương 4: Kiến trúc hệ CSDL, Chương 8: Truy vấn không gian vector, Chương 10: Chỉ số hóa âm thanh).
2. Thư viện phân tích tín hiệu âm học Python - `librosa` (https://librosa.org/)
3. Hệ sinh thái Máy học - `scikit-learn` documentation (Nearest Neighbors, KDTree, Data Preprocessing).
