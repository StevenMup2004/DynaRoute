import json
import os
import random
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from models import BNN

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Biến cố định Threshold (có thể chỉnh sửa tại đây)
FIXED_THRESHOLD = 0.6


def normalize_pred(text):
    text = str(text).strip()
    if text.startswith("PASS"):
        return "PASS"
    if text.startswith("FAIL"):
        return "FAIL"
    return text.split()[0] if text else "PASS"


# 1. Load Router Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BNN(input_dim=2048).to(device)
ckpt = torch.load(
    "/home/user04/save/dynaguard_1p7b_8b/bnn_small/model.pt",
    map_location=device,
)
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 2. Load Features & Outputs 1.7B / 8B từ log DynaGuard
features = torch.load(
    os.path.join(BASE_DIR, "data/dynaguard_1p7b_8b/val_features_backup.pt"),
    map_location="cpu",
)

sf_path = os.path.join(BASE_DIR, "../DynaGuard/log/tomg-group-umd/DynaGuard-1.7B/1781840603425104253-valid/outputs.jsonl")
lf_path = os.path.join(BASE_DIR, "../DynaGuard/log/tomg-group-umd/DynaGuard-8B/1781840866798715444-valid/outputs.jsonl")

small_preds, large_preds, gts = [], [], []

with (
    open(sf_path, "r", encoding="utf-8") as fs,
    open(lf_path, "r", encoding="utf-8") as fl,
):
    for ls, ll in zip(fs, fl):
        ds, dl = json.loads(ls), json.loads(ll)
        gts.append(1 if ds["metadata"]["label"] == "FAIL" else 0)
        small_preds.append(1 if normalize_pred(ds["output"]) == "FAIL" else 0)
        large_preds.append(1 if normalize_pred(dl["output"]) == "FAIL" else 0)

gts, small_preds, large_preds = (
    np.array(gts),
    np.array(small_preds),
    np.array(large_preds),
)

with open(os.path.join(BASE_DIR, "valid_router.json"), "r", encoding="utf-8") as f:
    vr = json.load(f)
sources = np.array([x["source"] for x in vr])

# 3. Tái hiện chỉ số Test (200 mẫu) & Valid (886 mẫu)
full_router_labels = (
    torch.load(os.path.join(BASE_DIR, "data/dynaguard_1p7b_8b/val_labels_backup.pt"), map_location="cpu")
    .long()
    .numpy()
)
idx_0 = np.where(full_router_labels == 0)[0].tolist()
idx_1 = np.where(full_router_labels == 1)[0].tolist()
random.seed(42)
test_indices = random.sample(idx_0, 100) + random.sample(idx_1, 100)
val_indices = list(set(range(len(gts))) - set(test_indices))

# 4. Chạy suy luận Router MLP
router_scores = []
with torch.no_grad():
    for i in range(0, len(features), 64):
        sc = (
            torch.sigmoid(model(features[i : i + 64].to(device)))
            .cpu()
            .numpy()
            .flatten()
        )
        router_scores.append(sc)
router_scores = np.concatenate(router_scores)


def calc_metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    prec = precision_score(y_true, y_pred, zero_division=1.0)
    rec = recall_score(y_true, y_pred, zero_division=1.0)
    f1 = f1_score(y_true, y_pred, zero_division=1.0)
    return prec, rec, f1, acc


def evaluate_pipeline(title, indices):
    sub_gt, sub_sp, sub_lp = gts[indices], small_preds[indices], large_preds[indices]
    sub_rsc, sub_src, sub_rlbl = (
        router_scores[indices],
        sources[indices],
        full_router_labels[indices],
    )

    print(f"\n{'=' * 95}")
    print(f"### {title.upper()} (Tổng: {len(indices)} mẫu) ###")
    print(f"{'=' * 95}")

    for src_name, mask in [
        ("TỔNG HỢP CHUNG", np.ones(len(indices), dtype=bool)),
        ("dynabench_latest", sub_src == "dynabench_latest"),
        ("montehoover/DynaBench", sub_src == "montehoover/DynaBench"),
    ]:
        y_true, sp, lp = sub_gt[mask], sub_sp[mask], sub_lp[mask]
        rsc, rlbl = sub_rsc[mask], sub_rlbl[mask]
        if len(y_true) == 0:
            continue
        tot = len(y_true)

        # Baselines
        _, _, f1_1p7b, acc_1p7b = calc_metrics(y_true, sp)
        _, _, f1_8b, acc_8b = calc_metrics(y_true, lp)

        # Oracle Groundtruth Router
        oracle_pred = np.where(rlbl == 1, lp, sp)
        _, _, f1_oracle, acc_oracle = calc_metrics(y_true, oracle_pred)
        gt_8b_calls = np.sum(rlbl == 1)

        print(f"\n---> [{src_name}] ({tot} mẫu)")
        print("-" * 90)
        print(f"  + Baseline 1.7B độc lập : Safety F1: {f1_1p7b:.4f} | Acc: {acc_1p7b:.4f} | Gọi 8B:   0 lần (  0.0%)")
        print(f"  + Baseline 8B độc lập   : Safety F1: {f1_8b:.4f} | Acc: {acc_8b:.4f} | Gọi 8B: {tot:3d} lần (100.0%)")
        print(f"  + Oracle lý tưởng tuyệt đối: Safety F1: {f1_oracle:.4f} | Acc: {acc_oracle:.4f} | Gọi 8B: {gt_8b_calls:3d} lần ({gt_8b_calls/tot*100:5.1f}%)")

        # Đánh giá E2E tại FIXED_THRESHOLD cố định
        route_8b_fix = rsc > FIXED_THRESHOLD
        routed_pred_fix = np.where(route_8b_fix, lp, sp)
        p_f, r_f, f1_f, acc_f = calc_metrics(y_true, routed_pred_fix)
        calls_f = np.sum(route_8b_fix)

        diff = calls_f - gt_8b_calls
        cost_note = f"Gọi 8B NHIỀU hơn lý tưởng {diff} lần" if diff > 0 else (f"Gọi 8B ÍT hơn lý tưởng {abs(diff)} lần" if diff < 0 else "Khớp 100% chi phí")

        print(f"  * MLP ROUTER (@ Cố định {FIXED_THRESHOLD:.2f}): Safety F1: {f1_f:.4f} | Prec: {p_f:.4f} | Rec: {r_f:.4f} | Acc: {acc_f:.4f}")
        print(f"     => Tỉ lệ gọi 8B: {calls_f:3d} lần ({calls_f/tot*100:5.1f}%) | Chất lượng: Đạt {f1_f/f1_8b*100:.1f}% model 8B | {cost_note}")

        # Tính toán phân rã năng lực (Capability Breakdown)
        both_corr = np.sum((sp == y_true) & (lp == y_true))
        only_8b_corr = np.sum((sp != y_true) & (lp == y_true))
        only_1p7b_corr = np.sum((sp == y_true) & (lp != y_true))
        both_wrong = np.sum((sp != y_true) & (lp != y_true))
        print(f"     [PHÂN RÃ NĂNG LỰC ({tot} mẫu)]: Cả hai đúng: {both_corr:3d} ({both_corr/tot*100:4.1f}%) | Chỉ 8B đúng (khó): {only_8b_corr:3d} ({only_8b_corr/tot*100:4.1f}%) | Chỉ 1.7B đúng: {only_1p7b_corr:3d} ({only_1p7b_corr/tot*100:4.1f}%) | Cả hai sai: {both_wrong:3d} ({both_wrong/tot*100:4.1f}%)")


evaluate_pipeline("1. Toàn bộ 1086 mẫu gốc", range(len(gts)))
evaluate_pipeline("2. Tập Test 200 mẫu", test_indices)
evaluate_pipeline("3. Tập Valid 886 mẫu", val_indices)
