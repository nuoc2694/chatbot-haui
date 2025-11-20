import os
import time
import json
import hashlib
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from dotenv import load_dotenv
from functools import wraps
from google import genai
from google.genai import types

# -------------------------------------------------
# 1. Load API key & khởi tạo client
# -------------------------------------------------
load_dotenv()  # đọc file .env nếu có

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Vui lòng đặt biến môi trường GEMINI_API_KEY trước khi chạy.")

client = genai.Client(api_key=API_KEY)

# Mô hình LLM sử dụng
MODEL_NAME = "gemini-2.5-flash"  # :contentReference[oaicite:1]{index=1}

# Tên store dùng cho File Search Tool (một store dùng chung cho toàn bộ tài liệu HaUI)
FILE_SEARCH_STORE = None

app = Flask(__name__)
UPLOAD_DIR = "uploads"              # Thư mục lưu file trên server
META_FILE = "uploaded_docs.json"    # File lưu metadata các file đã upload lên File Search
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change_me_please")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password123")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return f(*args, **kwargs)
    return decorated

# -------------------------------------------------
# 2. Hàm tiện ích: khởi tạo File Search Store
# -------------------------------------------------
STORE_ID_FILE = "store_id.txt"

def init_file_search_store():
    global FILE_SEARCH_STORE
    if FILE_SEARCH_STORE is not None:
        return FILE_SEARCH_STORE

    # 1. Nếu đã có file store_id.txt thì đọc lại
    if os.path.exists(STORE_ID_FILE):
        with open(STORE_ID_FILE, "r", encoding="utf-8") as f:
            store_name = f.read().strip()
            if store_name:
                FILE_SEARCH_STORE = store_name
                print(f"[INFO] Reuse File Search Store: {FILE_SEARCH_STORE}")
                return FILE_SEARCH_STORE

    # 2. Nếu chưa có thì tạo store mới
    store = client.file_search_stores.create(
        config={"display_name": "HaUI Regulations Store"}
    )
    FILE_SEARCH_STORE = store.name

    # 3. Ghi store.name xuống file để lần sau dùng lại
    with open(STORE_ID_FILE, "w", encoding="utf-8") as f:
        f.write(FILE_SEARCH_STORE)

    print(f"[INFO] Created new File Search Store: {FILE_SEARCH_STORE}")
    return FILE_SEARCH_STORE

# Khởi tạo store ngay khi app start
init_file_search_store()

def compute_file_hash(path: str) -> str:
    """Tính SHA-256 của file để nhận diện nội dung."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_meta():
    """Đọc danh sách metadata các file đã từng upload lên File Search."""
    if not os.path.exists(META_FILE):
        return []

    try:
        with open(META_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def save_meta(meta_list):
    """Ghi lại danh sách metadata."""
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)


def find_existing_file(meta_list, file_hash, file_size):
    """Kiểm tra xem đã có file cùng hash + size trong metadata chưa."""
    for item in meta_list:
        if item.get("hash") == file_hash and item.get("size") == file_size:
            return item
    return None

# -------------------------------------------------
# 3. Routes giao diện
# -------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            next_url = request.args.get("next") or url_for("upload_page")
            return redirect(next_url)
        else:
            error = "Sai tài khoản hoặc mật khẩu."

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat")
def chat_page():
    return render_template("chat.html")


@app.route("/upload")
@login_required
def upload_page():
    return render_template("upload.html")


# -------------------------------------------------
# 4. API: Chat với Gemini + File Search (RAG)
# -------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True)
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Tin nhắn rỗng."}), 400

    # Đảm bảo đã có File Search Store
    store_name = init_file_search_store()

    try:
        # Cấu hình tool File Search để LLM có thể truy vấn tài liệu đã upload
        config = types.GenerateContentConfig(
            system_instruction=(
                "Bạn là trợ lý ảo của Trường Đại học Công nghiệp Hà Nội (HaUI). "
                "Ưu tiên trả lời dựa trên các tài liệu quy định đã được cung cấp. "
                "Nếu thông tin không có trong tài liệu, hãy nói rõ và trả lời ở mức tổng quát."
            ),
            tools=[
                types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[store_name]
                    )
                )
            ],
        )

        # Gửi câu hỏi tới Gemini 2.5 Flash + File Search Tool :contentReference[oaicite:3]{index=3}
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_message,
            config=config,
        )

        reply = response.text or "Xin lỗi, mình chưa hiểu câu hỏi."
        return jsonify({"reply": reply})

    except Exception as e:
        print("[ERROR] /api/chat:", e)
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# 5. API: Upload tài liệu vào File Search Store
# -------------------------------------------------
@app.route("/api/upload", methods=["POST"])
@login_required
def api_upload():
    # 1. Lấy file từ form
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file trong request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Bạn chưa chọn file."}), 400

    store_name = init_file_search_store()

    # 2. Lưu file về server (luôn luôn)
    #    Đặt tên file đầy đủ đường dẫn
    server_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(server_path)

    # 3. Tính hash & size để nhận diện nội dung file
    file_hash = compute_file_hash(server_path)
    file_size = os.path.getsize(server_path)

    # 4. Đọc metadata hiện có
    meta_list = load_meta()

    # 5. Kiểm tra xem đã có file giống hệt (hash + size) hay chưa
    existing = find_existing_file(meta_list, file_hash, file_size)

    if existing is not None:
        # ĐÃ TỪNG UPLOAD FILE NÀY LÊN FILE SEARCH RỒI
        # → Xoá bản mới tải về (giữ lại bản cũ trên server nếu bạn thích)
        try:
            os.remove(server_path)
        except OSError:
            pass

        return jsonify(
            {
                "message": (
                    "File này đã từng được tải lên File Search trước đó. "
                    "Đã xoá bản trùng mới tải về, không upload lại lên Gemini."
                ),
                "file_name": file.filename,
                "hash": file_hash,
                "size": file_size,
                "store_name": store_name,
                "first_uploaded_at": existing.get("uploaded_at"),
            }
        )

    # 6. Nếu CHƯA có file giống → Upload lên File Search
    try:
        operation = client.file_search_stores.upload_to_file_search_store(
            file=server_path,
            file_search_store_name=store_name,
            config={"display_name": file.filename},
        )

        # Chờ index xong (đơn giản bằng polling)
        while not operation.done:
            time.sleep(2)
            operation = client.operations.get(operation)

        # 7. Lưu metadata cho file này (để nhận diện lần sau)
        meta_list.append(
            {
                "file_name": file.filename,
                "path": server_path,
                "hash": file_hash,
                "size": file_size,
                "store_name": store_name,
                "uploaded_at": datetime.now().isoformat(),
                # Nếu bạn lấy được document_name từ operation thì lưu thêm ở đây
                # "document_name": "...",
            }
        )
        save_meta(meta_list)

        # LƯU Ý: KHÁC VỚI BẢN CŨ, Ở ĐÂY MÌNH **KHÔNG XOÁ** server_path
        # để bạn luôn có bản lưu cục bộ trong thư mục uploads.

        return jsonify(
            {
                "message": "Tải tài liệu thành công, đã lưu ở server và index vào File Search.",
                "store_name": store_name,
                "file_name": file.filename,
                "hash": file_hash,
                "size": file_size,
            }
        )

    except Exception as e:
        print("[ERROR] /api/upload:", e)
        # Nếu upload lên Gemini lỗi, bạn có thể chọn:
        # - Giữ file trên server (để thử lại sau), hoặc
        # - Xoá file. Ở đây mình GIỮ lại.
        return jsonify({"error": str(e)}), 500


# -------------------------------------------------
# 6. Chạy app
# -------------------------------------------------
if __name__ == "__main__":
    # debug=True chỉ dùng lúc phát triển
    app.run(host="0.0.0.0", port=5000, debug=True)
