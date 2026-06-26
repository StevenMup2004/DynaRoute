from huggingface_hub import HfApi
import os

# 1. Tên repository bạn muốn tạo trên Hugging Face (Thay username của bạn vào)
# Ví dụ: "dang-vh/safe-route-dynaguard"
REPO_ID = "StevenMup2004/SafeRoute-final"

# 2. Đường dẫn tới thư mục chứa file model.pt của chúng ta
LOCAL_MODEL_DIR = "/home/user04/save/dynaguard_1p7b_8b/bnn_small"

api = HfApi()

print(f"Đang tạo repository {REPO_ID} trên Hugging Face (nếu chưa có)...")
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

print("Đang đẩy file model lên Hugging Face. Vui lòng chờ...")
api.upload_folder(
    folder_path=LOCAL_MODEL_DIR,
    repo_id=REPO_ID,
    repo_type="model",
)

print(f"Xong! Model của bạn đã có mặt tại: https://huggingface.co/{REPO_ID}")
