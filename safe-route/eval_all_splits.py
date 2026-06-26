import json
import os
import random
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import torch
from models import BNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Biến cố định Threshold (có thể chỉnh sửa tại đây)
FIXED_THRESHOLD = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BNN(input_dim=2048).to(device)

ckpt = torch.load(
    "/home/user04/save/dynaguard_1p7b_8b/bnn_small/model.pt",
    map_location=device,
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 1. Load toàn bộ 1086 features gốc từ file backup
full_features = torch.load(
    os.path.join(BASE_DIR, "data/dynaguard_1p7b_8b/val_features_backup.pt"),
    map_location="cpu",
)
full_labels = (
    torch.load(os.path.join(BASE_DIR, "data/dynaguard_1p7b_8b/val_labels_backup.pt"), map_location="cpu")
    .long()
    .numpy()
)

with open(os.path.join(BASE_DIR, "valid_router.json"), "r", encoding="utf-8") as f:
    vr = json.load(f)

full_sources = np.array([x["source"] for x in vr])

# 2. Tái hiện chuẩn xác phép tách Test (200 mẫu) và Valid (886 mẫu) theo spit_valid.py
idx_0 = np.where(full_labels == 0)[0].tolist()
idx_1 = np.where(full_labels == 1)[0].tolist()

random.seed(42)  # Cố định seed 42 y hệt spit_valid.py
test_idx_0 = random.sample(idx_0, 100)
test_idx_1 = random.sample(idx_1, 100)

test_indices = test_idx_0 + test_idx_1
val_indices = list(set(range(len(full_labels))) - set(test_indices))

# 3. Chạy dự đoán trên toàn bộ 1086 mẫu
all_scores = []
with torch.no_grad():
    for i in range(0, len(full_features), 64):
        batch_x = full_features[i : i + 64].to(device)
        scores = torch.sigmoid(model(batch_x)).cpu().numpy().flatten()
        all_scores.append(scores)
all_scores = np.concatenate(all_scores)


# 4. Hàm thống kê từng tập
def evaluate_split(title, indices):
    sub_lbl = full_labels[indices]
    sub_scr = all_scores[indices]
    sub_src = full_sources[indices]

    print(f"\n{'=' * 85}")
    print(f"### {title.upper()} (Tổng: {len(indices)} mẫu) ###")
    print(f"{'=' * 85}")

    for src_name, mask_src in [
        ("TỔNG HỢP (CHUNG)", np.ones(len(indices), dtype=bool)),
        ("dynabench_latest", sub_src == "dynabench_latest"),
        ("montehoover/DynaBench", sub_src == "montehoover/DynaBench"),
    ]:
        lbl = sub_lbl[mask_src]
        scr = sub_scr[mask_src]
        if len(lbl) == 0:
            continue
        auprc = average_precision_score(lbl, scr) if len(set(lbl)) > 1 else 0.0
        total = len(lbl)

        # Đánh giá tại FIXED_THRESHOLD cố định
        p_fix = (scr > FIXED_THRESHOLD).astype(int)
        f1_fix = f1_score(lbl, p_fix, zero_division=1.0)
        prec_fix = precision_score(lbl, p_fix, zero_division=1.0)
        rec_fix = recall_score(lbl, p_fix, zero_division=1.0)
        acc_fix = np.mean(p_fix == lbl)

        print(f"\n---> [{src_name}] | Tổng: {total} mẫu | AUPRC: {auprc:.4f}")
        print(f"     [CỐ ĐỊNH Thresh = {FIXED_THRESHOLD:.2f}]: F1: {f1_fix:.4f} | Prec: {prec_fix:.4f} | Rec: {rec_fix:.4f} | Acc: {acc_fix:.4f}")

        gt_1p7b, gt_8b = np.sum(lbl == 0), np.sum(lbl == 1)
        pred_1p7b, pred_8b = np.sum(p_fix == 0), np.sum(p_fix == 1)

        print(f"     [ROUTING RATIO @ {FIXED_THRESHOLD:.2f}]:")
        print(f"       + GROUNDTRUTH : Gọi 1.7B: {gt_1p7b:3d} lần ({gt_1p7b/total*100:5.1f}%)  |  Gọi 8B: {gt_8b:3d} lần ({gt_8b/total*100:5.1f}%)")
        print(f"       + PREDICTED   : Gọi 1.7B: {pred_1p7b:3d} lần ({pred_1p7b/total*100:5.1f}%)  |  Gọi 8B: {pred_8b:3d} lần ({pred_8b/total*100:5.1f}%)")
        
        diff_8b = pred_8b - gt_8b
        status = "Gọi 8B NHIỀU hơn lý tưởng" if diff_8b > 0 else ("Gọi 8B ÍT hơn lý tưởng" if diff_8b < 0 else "Khớp hoàn hảo 100%")
        print(f"       => Đánh giá   : {status} (Lệch {abs(diff_8b)} mẫu ~ {abs(diff_8b)/total*100:.1f}%)")


evaluate_split("1. Toàn bộ 1086 mẫu gốc (Full valid_router.json)", range(1086))
evaluate_split(
    "2. Tập Test 200 mẫu (đang dùng đánh giá trong train_router.py)",
    test_indices,
)
evaluate_split("3. Tập Validation 886 mẫu còn lại", val_indices)
