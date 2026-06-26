import os
import shutil
from huggingface_hub import HfApi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# CẤU HÌNH REPOSITORY HUGGING FACE
# =====================================================================
REPO_ID = "StevenMup2004/DynaGuard-Data"  # <-- Dataset Repository chính thức
REPO_TYPE = "dataset"  # <-- Để "dataset" hoặc "model"

api = HfApi()

print(f"[*] Đang kết nối tới Hugging Face Repo: {REPO_ID} ({REPO_TYPE})...")
api.create_repo(repo_id=REPO_ID, repo_type=REPO_TYPE, exist_ok=True)

# Đảm bảo file test_router.json tồn tại ở local (copy từ valid_router.json nếu cần)
valid_path = os.path.join(BASE_DIR, "valid_router.json")
test_path = os.path.join(BASE_DIR, "test_router.json")
if not os.path.exists(test_path) and os.path.exists(valid_path):
    print("[*] Đang sao chép valid_router.json sang test_router.json ở local...")
    shutil.copy(valid_path, test_path)

files_to_push = ["train_router.json", "test_router.json"]

for fname in files_to_push:
    fpath = os.path.join(BASE_DIR, fname)
    if os.path.exists(fpath):
        print(f"[*] Đang upload file: {fname} (dung lượng: {os.path.getsize(fpath)/1024/1024:.2f} MB)...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
        )
        print(f" => Thành công upload {fname}!")
    else:
        print(f"[!] LỖI: Không tìm thấy file tại {fpath}")

# Upload Dataset Card làm README.md trên Hugging Face
card_path = os.path.join(BASE_DIR, "DATASET_CARD.md")
if os.path.exists(card_path):
    print("[*] Đang upload Dataset Card (README.md)...")
    api.upload_file(
        path_or_fileobj=card_path,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )
    print(" => Thành công upload Dataset Card!")

repo_url = f"https://huggingface.co/{'datasets/' if REPO_TYPE == 'dataset' else ''}{REPO_ID}"
print(f"\n[+] HOÀN TẤT! Dữ liệu của bạn đã sẵn sàng tại: {repo_url}")
