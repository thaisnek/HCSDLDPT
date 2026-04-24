@AGENTS.md

## Claude Code specific rules

- Đọc và tuân thủ toàn bộ nội dung trong AGENTS.md trước khi làm bất kỳ việc gì.
- Nếu không đọc được AGENTS.md, DỪNG LẠI và báo người dùng — không tự code.
- Trước khi sửa nhiều file hoặc thay đổi kiến trúc/schema, nêu kế hoạch ngắn gọn. Chỉ chờ xác nhận nếu thay đổi có thể ảnh hưởng `dataset/`, DB, schema, thuật toán tìm kiếm hoặc phạm vi bài toán.
- Không chỉnh sửa thư mục `dataset/` trừ khi người dùng yêu cầu rõ ràng bằng văn bản.
- Không reset database (`clear_db` hoặc xóa file `.db`) trừ khi người dùng chỉ định rõ ràng.
- Khi thêm đặc trưng mới, LUÔN cập nhật đồng thời: `dac_trung/`, `processing/extract_features.py`, `database/db_manager.py`, `search/search_similar.py` nếu vector search bị ảnh hưởng, và `webapp/app.py` nếu UI hiển thị vector hoặc tên đặc trưng.
- Không thêm PyTorch, TensorFlow, Keras, FAISS hoặc bất kỳ framework nặng nào nếu người dùng không yêu cầu.
