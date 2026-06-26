import torch
import random
import os

# Bạn sửa tên thư mục theo đúng version đang chạy nhé
data_dir = "data/dynaguard_1p7b_8b"

print("Đang tải dữ liệu Validation cũ...")
val_features = torch.load(f"{data_dir}/val_features.pt")
val_labels = torch.load(f"{data_dir}/val_labels.pt")

# Tìm ra index (vị trí) của nhãn 0 và nhãn 1
idx_0 = torch.where(val_labels == 0)[0].tolist()
idx_1 = torch.where(val_labels == 1)[0].tolist()

print(f"Tổng số mẫu ban đầu -> Nhãn 0: {len(idx_0)}, Nhãn 1: {len(idx_1)}")

random.seed(42) # Cố định seed để kết quả không bị thay đổi mỗi lần chạy

# Lấy ngẫu nhiên 100 mẫu mỗi class cho Test
N_TEST = 100
test_idx_0 = random.sample(idx_0, N_TEST)
test_idx_1 = random.sample(idx_1, N_TEST)

# Phần còn lại sẽ dùng cho tập Validation mới
new_val_idx_0 = list(set(idx_0) - set(test_idx_0))
new_val_idx_1 = list(set(idx_1) - set(test_idx_1))

# Trộn ngẫu nhiên thứ tự
test_indices = test_idx_0 + test_idx_1
random.shuffle(test_indices)

new_val_indices = new_val_idx_0 + new_val_idx_1
random.shuffle(new_val_indices)

# Tạo tensor dữ liệu mới dựa trên index
test_features = val_features[test_indices]
test_labels = val_labels[test_indices]

new_val_features = val_features[new_val_indices]
new_val_labels = val_labels[new_val_indices]

# Backup file cũ để phòng rủi ro
os.rename(f"{data_dir}/val_features.pt", f"{data_dir}/val_features_backup.pt")
os.rename(f"{data_dir}/val_labels.pt", f"{data_dir}/val_labels_backup.pt")

# Lưu tập Validation mới đè lên tên cũ (để code train_router.py vẫn chạy bình thường không cần sửa)
torch.save(new_val_features, f"{data_dir}/val_features.pt")
torch.save(new_val_labels, f"{data_dir}/val_labels.pt")

# Lưu tập Test mới
torch.save(test_features, f"{data_dir}/test_features.pt")
torch.save(test_labels, f"{data_dir}/test_labels.pt")

print("\nĐÃ XỬ LÝ XONG!")
print(f"Tập TEST mới được tạo   -> Nhãn 0: {N_TEST}, Nhãn 1: {N_TEST}")
print(f"Tập VALID mới được cập nhật -> Nhãn 0: {len(new_val_idx_0)}, Nhãn 1: {len(new_val_idx_1)}")
