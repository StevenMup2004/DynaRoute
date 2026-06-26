# Tổng hợp kết quả đánh giá DynaGuard / SafeRoute trên dữ liệu tiếng Việt

*Cập nhật: 2026-06-25*

## 1. Cấu hình chung

| Mục | Giá trị |
|---|---|
| Model nhỏ (small) | `tomg-group-umd/DynaGuard-1.7B` |
| Model lớn (large) | `tomg-group-umd/DynaGuard-8B` |
| Router | `StevenMup2004/SafeRoute-final` (MLP 2048→1024→512→256→1, ngưỡng 0.6) |
| Chế độ | **non-CoT** (`enable_thinking=False`, prefill `<answer>\n`) |
| Giải mã | greedy / so logit `FAIL` vs `PASS` tại token cuối |
| Số seed | 1 (greedy) |
| Prompt | chuẩn DynaGuard gốc (`<rules>`/`<transcript>`) |

> Lưu ý: phần **Guard thuần** (Mục 3) dùng dự đoán bằng *generate*; phần **SafeRoute 3 chế độ** (Mục 4–5) dùng *so logit* → chênh nhau ~±0.5% ở Small/Large là bình thường.

## 2. Các bộ dữ liệu

| Bộ dữ liệu | Mô tả | Số mẫu | PASS / FAIL |
|---|---|---|---|
| **DynaBench-VI** | Dịch tập test DynaBench sang tiếng Việt (policy diễn đạt đa dạng, transcript trung thực) | 543 | 276 / 267 |
| **DynaBench-EN** | Bản gốc tiếng Anh (đối chứng, cùng harness) | 543 | 276 / 267 |
| **VN-native** | Dữ liệu tiếng Việt tự sinh (chủ quyền VN, bảo mật nội bộ, lăng mạ…) | 504 | 288 / 216 |

---

## 3. Guard thuần — DynaBench: Anh vs Việt ("thuế ngôn ngữ")

| Model | Lang | Accuracy | Macro-F1 | F1(FAIL) | F1(PASS) | Dự đoán FAIL/PASS |
|---|---|---|---|---|---|---|
| 1.7B | EN | 0.6483 | 0.6482 | 0.6456 | 0.6508 | 272 / 271 ⚖️ |
| 1.7B | VI | 0.6059 | 0.5940 | 0.6635 | 0.5244 | 369 / 174 (thiên FAIL) |
| 8B | EN | 0.7772 | 0.7722 | 0.7387 | 0.8058 | 196 / 347 |
| 8B | VI | 0.7348 | 0.7294 | 0.6910 | 0.7677 | 199 / 344 |

**Thuế ngôn ngữ (EN − VI):**

| Model | Δ Accuracy | Δ Macro-F1 | Tín hiệu VI giữ lại* |
|---|---|---|---|
| 1.7B | **−4.24** | −5.42 | 71% |
| 8B | **−4.24** | −4.28 | 85% |

\* phần (acc − 50%) còn giữ được so với EN.

**Nhận định:**
- Thuế accuracy ~**4.2 điểm**, **ổn định bất kể kích thước** model. Scale up nâng cả EN lẫn VI (~+13 điểm) nhưng **không xóa** khoảng cách VI–EN.
- Ở 1.7B, tiếng Việt **làm đảo bias**: EN cân bằng (272/271) → VI nghiêng hẳn về FAIL (369/174) ⇒ "không chắc thì báo vi phạm".
- Ở 8B, bias **không đổi hướng** (vẫn PASS-lean) → tiếng Việt chỉ kéo accuracy xuống đều, cấu trúc lỗi ổn định hơn.
- Đối chứng bài báo (English, CoT, 6 seed): 1.7B F1 65.2 / 8B F1 73.1 — bản tái hiện EN non-CoT của ta khớp trong ~0.6 điểm ⇒ **harness/prompt đúng**.

---

## 4. SafeRoute 3 chế độ — DynaBench-VI (543 mẫu)

| Chế độ | Accuracy | Macro-F1 | F1(FAIL) | %dùng 8B |
|---|---|---|---|---|
| Small (1.7B) | 0.6041 | 0.5934 | 0.6593 | 0% |
| Large (8B) | 0.7366 | 0.7311 | 0.6925 | 100% |
| **SafeRoute** | **0.7532** | **0.7517** | 0.7320 | **54.5%** |
| Oracle (trần) | 0.8674 | 0.8673 | 0.8631 | — |

Routing F1 = **0.5695** · mẫu-khó thực = 143 · route-8B = 296

**Phân rã ai đúng (543):** cả-hai-đúng 257 · chỉ-1.7B-đúng 71 · chỉ-8B-đúng (khó) 143 · cả-hai-sai 72

**Nhận định:**
- **SafeRoute VƯỢT Large** (0.753 > 0.737) trong khi chỉ dùng 8B cho 54.5% mẫu ⇒ chính xác hơn **và** rẻ hơn ~45% so với always-8B.
- Trên tiếng Việt **hai guard bổ sung nhau mạnh** (bất đồng 214 mẫu; cả-hai-sai chỉ 72) ⇒ Oracle vọt lên 0.867.
- **Dư địa → Oracle = 11.4 điểm** (gấp đôi EN) ⇒ train router VI/bilingual rất đáng.

*(Đối chứng EN cùng harness: Small 0.6501 · Large 0.7753 · SafeRoute 0.7753 (=Large) · Oracle 0.8398 · Routing F1 0.4738 · %8B 40.9. Phân rã EN: 318/35/103/87.)*

---

## 5. SafeRoute 3 chế độ — VN-native (504 mẫu)

| Chế độ | Accuracy | Macro-F1 | F1(FAIL) | %dùng 8B |
|---|---|---|---|---|
| **Small (1.7B)** | **0.8988** | **0.8944** | 0.8728 | 0% |
| Large (8B) | 0.8591 | 0.8468 | 0.8033 | 100% |
| SafeRoute | 0.8671 | 0.8561 | 0.8164 | 32.1% |
| Oracle (trần) | 0.9187 | 0.9143 | 0.8951 | — |

Routing F1 = **0.1163** · mẫu-khó thực = 10 · route-8B = 162

**Phân rã ai đúng (504):** cả-hai-đúng 423 · chỉ-1.7B-đúng 30 · chỉ-8B-đúng (khó) 10 · cả-hai-sai 41

**Nhận định (đảo ngược hoàn toàn):**
- **1.7B (0.899) > 8B (0.859)** — model nhỏ chính xác hơn model lớn ở domain này.
- ⇒ **SafeRoute (0.867) còn TỆ HƠN Small-only (0.899)**: router đẩy 32% mẫu sang đúng model kém hơn.
- `chỉ-1.7B-đúng (30) ≫ chỉ-8B-đúng (10)` → Oracle chỉ hơn Small +2 điểm; **không có dư địa routing**.
- Dataset này **dễ/rõ ràng hơn** (cả hai 85–90%) và 8B yếu hơn ở chủ đề nhạy cảm.
- ⇒ Với domain VN bản địa: **dùng thẳng 1.7B** (chính xác nhất + rẻ nhất). Routing là công cụ sai.

---

## 6. Bảng tổng hợp & kết luận

| | DynaBench-VI | VN-native |
|---|---|---|
| Thứ bậc model | **8B > 1.7B** | **1.7B > 8B** ⚠️ |
| Small acc | 0.604 | 0.899 |
| Large acc | 0.737 | 0.859 |
| SafeRoute acc | **0.753** (> Large) | **0.867** (< Small) |
| Oracle acc | 0.867 | 0.919 |
| Dư địa → Oracle | **11.4** (lớn) | **2.0** (nhỏ) |
| Routing có ích? | **Có** | **Không** |
| Khuyến nghị | Train router bilingual | Dùng 1.7B; cân nhắc fine-tune 1.7B |

**Kết luận chính:**
1. **Thuế ngôn ngữ DynaGuard ≈ 4 điểm accuracy**, ổn định theo kích thước; scale up nâng mặt bằng nhưng không xóa khoảng cách.
2. **Giá trị của routing phụ thuộc domain**: chỉ có ý nghĩa khi 8B > 1.7B (DynaBench-VI). Ở VN-native, 1.7B mạnh hơn nên routing phản tác dụng.
3. **Router EN transfer sang VI tốt bất ngờ** trên DynaBench-VI (vượt cả Large), nhưng còn dư địa lớn (11.4 điểm) cho việc train router VI/bilingual.

**Cảnh báo chung:** tất cả là 1 seed / greedy / non-CoT. Trước khi báo cáo chính thức nên multi-seed (3–6) để vượt nhiễu; với guardrail nên xem thêm **recall(FAIL)** (bắt sót vi phạm nguy hiểm hơn báo nhầm).

---

## Phụ lục — Confusion matrix (hàng = gold, cột = pred; thứ tự [FAIL, PASS])

**DynaBench guard thuần**
| | FAIL→ | PASS→ |
|---|---|---|
| 1.7B EN | [174, 93] | [98, 178] |
| 1.7B VI | [211, 56] | [158, 118] |
| 8B EN | [171, 96] | [25, 251] |
| 8B VI | [161, 106] | [38, 238] |

**VN-native — SafeRoute (504):** FAIL precision 1.000 / recall 0.690 · PASS precision 0.811 / recall 1.000
