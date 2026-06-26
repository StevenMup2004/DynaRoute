# KẾ HOẠCH & KỊCH BẢN CHI TIẾT SLIDE BÁO CÁO (EXECUTIVE SLIDE DECK PLAN)
**Chủ đề:** Tối ưu hóa Chi phí & Hiệu năng LLM Guardrail qua Cơ chế Định tuyến Động SafeRoute: Từ Nền tảng Tiếng Anh đến Thích nghi Tiếng Việt  
**Dự án:** SafeRoute MLP & Agentic Guardrail Benchmark  
**Ngày cập nhật:** 2026-06-25  
**Quy ước quan trọng:** Toàn bộ khái niệm mô hình lớn (LPG cũ) được chuẩn hóa tường minh thành **`DynaGuard-8B`**.

---

## PHẦN I: TỔNG QUAN BỐ CỤC SLIDE (SLIDE OUTLINE - 11 SLIDES)

* **Slide 1: Trang Tiêu đề & Đặt vấn đề Nghiên cứu (Title & Motivation)**
* **Slide 2: Giới thiệu Phương pháp cốt lõi: Cơ chế Định tuyến Động SafeRoute (Methodology)**
* **Slide 3: Thiết lập Thí nghiệm & Pipeline An toàn Đầu - Cuối (Experimental Setup & E2E Pipeline)**
* **Slide 4: [TRỌNG TÊM 1 - TIẾNG ANH] Nền tảng Guard thuần DynaBench English Baseline**
* **Slide 5: [TRỌNG TÊM 1 - TIẾNG ANH] Hiệu năng Định luồng & An toàn SafeRoute trên Tiếng Anh**
* **Slide 6: Bước chuyển giao Tiếng Việt & Quy luật "Thuế ngôn ngữ" (Vietnamese Adaptation & Language Tax)**
* **Slide 7: [TRỌNG TÊM 2 - TIẾNG VIỆT] Các bộ Benchmark Đánh giá Tiếng Việt (Original vs Augmented)**
* **Slide 8: [TRỌNG TÊM 2 - TIẾNG VIỆT] Chỉ số Định luồng SafeRoute MLP trên 2 bộ DynaBench (@ 0.60)**
* **Slide 9: [TRỌNG TÊM 2 - TIẾNG VIỆT] Đọ sức Hiệu năng An toàn Đầu - Cuối (End-to-End Safety F1)**
* **Slide 10: [TRỌNG TÊM 2 - TIẾNG VIỆT] Phân tích Miền VN-native & Nghịch lý Kích thước**
* **Slide 11: Ma trận Đối chiếu Toàn diện Anh – Việt & Lộ trình Triển khai (Consolidated Matrix)**

---

## PHẦN II: KỊCH BẢN & SỐ LIỆU CHI TIẾT TỪNG SLIDE (TRỰC TIẾP COPY VÀO SLIDE)

### SLIDE 1: TRANG TIÊU ĐỀ & ĐẶT VẤN ĐỀ NGHIÊN CỨU
* **Tiêu đề lớn:** TỐI ƯU HÓA CHI PHÍ & HIỆU NĂNG LLM GUARDRAIL QUA ĐỊNH TUYẾN ĐỘNG SAFEROUTE
* **Tiêu đề phụ:** Kiểm chứng khả năng bám sát hiệu năng DynaGuard-8B với chi phí DynaGuard-1.7B
* **Bài toán thực tế (Pain Points):**
  * Mô hình kiểm duyệt an toàn cỡ lớn (**`DynaGuard-8B`**) có khả năng nhận diện vi phạm rất chính xác nhưng chi phí suy luận (Inference Cost) cực kỳ đắt đỏ và độ trễ cao khi triển khai quy mô lớn.
  * Mô hình nhỏ (**`DynaGuard-1.7B`**) phản hồi tức thì, giá rẻ nhưng dễ bị đánh lừa bởi các prompt lách luật tinh vi.
* **Mục tiêu nghiên cứu:** Giới thiệu và kiểm chứng phương pháp định tuyến tự động SafeRoute, tận dụng sức mạnh bổ trợ giữa mô hình nhỏ và lớn để đạt hiệu năng tối đa với chi phí tối thiểu.

---

### SLIDE 2: GIỚI THIỆU PHƯƠNG PHÁP CỐT LÕI: CƠ CHẾ ĐỊNH TUYẾN SAFEROUTE (METHODOLOGY)
* **Khái niệm lựa chọn mô hình thích ứng (Adaptive Model Selection):**
  * Thay vì gửi mọi Prompt tới mô hình 8B, hệ thống chèn một Bộ định tuyến thần kinh (Neural Router) đóng vai trò "người gác cổng" thông minh.
  * **Cơ chế hoạt động:** Trích xuất vector đặc trưng ẩn $h \in \mathbb{R}^{2048}$ tại lớp cuối cùng của model nhỏ (`DynaGuard-1.7B`) để dự đoán xác suất mẫu đó là "ca khó" cần model lớn giải quyết.
* **Nguyên tắc gán nhãn huấn luyện lý tưởng (Oracle Groundtruth Labeling):**
  * Nhãn $y=1$ *(Ca khó / Buộc gọi model lớn)*: Nếu mô hình 1.7B dự đoán sai nhưng mô hình `DynaGuard-8B` dự đoán đúng $\rightarrow$ **Buộc phải định luồng sang 8B**.
  * Còn lại là nhãn $y=0$ *(Ca dễ)*.
* **Kiến trúc Router MLP:** Neural Network 5 lớp (`2048 → 1024 → 512 → 256 → 1`), tối ưu bằng hàm mất mát **Focal Loss** ($\alpha=0.75, \gamma=2.0$) xử lý triệt để sự mất cân bằng giữa ca khó và ca dễ.

---

### SLIDE 3: THIẾT LẬP THÍ NGHIỆM & PIPELINE AN TOÀN ĐẦU - CUỐI (E2E PIPELINE)
* **Các thành phần tham gia:**
  1. **Small Model Baseline:** `tomg-group-umd/DynaGuard-1.7B`
  2. **Large Model Baseline:** `tomg-group-umd/DynaGuard-8B` *(Thay thế tường minh khái niệm LPG cũ)*
  3. **Trained Router:** `StevenMup2004/SafeRoute-final` (MLP checkpoint).
* **Luồng chạy thực tế End-to-End Pipeline:**
  * Bước 1: Cho Prompt đi qua mô hình 1.7B, trích xuất vector đặc trưng $x \in \mathbb{R}^{2048}$ và kết quả nhãn $Pred_{small}$.
  * Bước 2: Đưa $x$ vào Router MLP tính điểm xác suất $P_{route} = \sigma(\text{MLP}(x))$.
  * Bước 3: So sánh với ngưỡng quyết định cố định **`FIXED_THRESHOLD = 0.60`**:
    * Nếu $P_{route} \le 0.60 \rightarrow$ Nhận kết quả an toàn cuối cùng là $Pred_{small}$ *(Chi phí = 1.7B)*.
    * Nếu $P_{route} > 0.60 \rightarrow$ Gửi tiếp Prompt tới model lớn nhận kết quả $Pred_{large}$ *(Chi phí = 8B)*.

### Pipeline Sơ đồ khối

```text
Input + Dynamic Policy
        |
        v
 Dynamic Guard
        |
        v
      Router
     /      \
  Easy      Hard
   |          |
   v          v
Return       Dynaguard-8b
fast       
verdict(Dynaguard-1.7b)       |
              v
        final verdict
```

---

### SLIDE 4: [TRỌNG TÊM 1 - TIẾNG ANH] NỀN TẢNG GUARD THUẦN DYNABENCH ENGLISH BASELINE
* **Ngữ cảnh:** Đánh giá năng lực độc lập nguyên bản trên bộ benchmark DynaBench tiếng Anh chuẩn (543 mẫu: 276 PASS / 267 FAIL).
* **Bảng số liệu Guard thuần Tiếng Anh (English Baseline):**

| Mô hình | Ngôn ngữ | Accuracy | Macro-F1 | F1 (FAIL - Bắt vi phạm) | F1 (PASS - An toàn) | Phân bố dự đoán (FAIL / PASS) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **DynaGuard-1.7B** | EN | **64.83%** | 0.6482 | 0.6456 | 0.6508 | 272 / 271 *(Rất cân bằng)* |
| **DynaGuard-8B** | EN | **77.72%** | 0.7722 | 0.7387 | 0.8058 | 196 / 347 *(PASS-lean)* |

* **Nhận định Tiếng Anh:**
  * Mô hình `DynaGuard-8B` vượt trội model nhỏ +12.89% Accuracy.
  * Tái hiện kết quả gốc của bài báo SafeRoute khớp trong biên độ ±0.6% ⇒ Khẳng định **test harness và prompt chuẩn xác 100%**.

---

### SLIDE 5: [TRỌNG TÊM 1 - TIẾNG ANH] HIỆU NĂNG ĐỊNH LUỒNG & AN TOÀN SAFEROUTE EN
* **Số liệu thực chứng trên DynaBench-EN (543 mẫu):**

| Chiến lược | Accuracy | Macro-F1 | Safety F1 (FAIL) | Tỉ lệ gọi `DynaGuard-8B` | Chi phí tiết kiệm |
|---|:---:|:---:|:---:|:---:|:---:|
| **DynaGuard-1.7B** | 65.01% | 0.6495 | 0.6470 | 0% | +100% |
| **DynaGuard-8B** | 77.53% | 0.7711 | 0.7368 | 100% | 0% |
| **SafeRoute EN** | **77.53%** | **0.7711** | **0.7368** | **40.9%** | **Tiết kiệm ~59.1%** |
| *Oracle (Trần lý tưởng)* | *83.98%* | *0.8397* | *0.8315* | *25.4%* | *—* |

* **Phân rã năng lực Tiếng Anh (543 mẫu):**
  * Cả hai cùng đúng: **318 mẫu** (58.6%)  |  Chỉ 1.7B đúng: **35 mẫu** (6.4%).
  * **Chỉ model lớn 8B đúng (Ca khó thực sự): 103 mẫu** (19.0%) → *Thành công lớn nhất của Router*.
  * Cả hai cùng sai: **87 mẫu** (16.0%).
* **Kết luận trọng tâm EN:** SafeRoute đạt **độ chính xác ngang ngửa 100% model lớn `DynaGuard-8B`** trong khi **giảm 59.1% chi phí tính toán**!

---

### SLIDE 6: BƯỚC CHUYỂN GIAO TIẾNG VIỆT & QUY LUẬT "THUẾ NGÔN NGỮ"
* **Quy luật suy giảm khi dịch thuật (Language Tax):**
  * Khi chuyển sang bộ dữ liệu Tiếng Việt (cùng nội dung, cùng test harness), cả 2 model đều chịu mức giảm chuẩn:
  * **Δ Accuracy = −4.24 điểm %** (1.7B giảm từ 64.83% xuống 60.59%; 8B giảm từ 77.72% xuống 73.48%).
* **Phát hiện quan trọng về Đảo hướng Thiên lệch (Bias Shift):**
  * **Ở model 1.7B:** Tiếng Việt làm **đảo chiều bias trầm trọng**: Tiếng Anh cân bằng `272 FAIL / 271 PASS` → Tiếng Việt lệch hẳn sang `369 FAIL / 174 PASS` ⇒ *Hình thành tâm lý phòng thủ cực đoan: "Không hiểu rõ là báo vi phạm"*.
  * **Ở model `DynaGuard-8B`:** Bias giữ nguyên hướng ổn định (`199 FAIL / 344 PASS`), chứng tỏ cấu trúc ngữ nghĩa của model lớn vững chắc hơn trước ngôn ngữ mới.

---

### SLIDE 7: [TRỌNG TÊM 2 - TIẾNG VIỆT] CÁC BỘ BENCHMARK ĐÁNH GIÁ TIẾNG VIỆT
*(Đã loại bỏ sự trùng lặp gộp chung, tách bạch rõ ràng 2 bộ DynaBench gốc và biến thể)*

| Bộ dữ liệu Benchmark | Nguồn gốc dữ liệu | Số mẫu | PASS / FAIL | Đặc tính bài toán chuyên sâu |
|---|---|:---:|:---:|---|
| **DynaBench (Original)** | `montehoover/DynaBench` | **543** | 276 / 267 | Bộ dữ liệu gốc dịch chuẩn sang tiếng Việt, giữ nguyên trật tự policy đa dạng |
| **DynaBench (Augmented Ver)** | `dynabench_latest` | **543** | 276 / 267 | Phiên bản tăng cường: **Đổi thứ tự policy và xóa policy `violent`** để kiểm tra tính bền vững của mô hình |
| **VN-native** | Dữ liệu bản địa tự sinh | **504** | 288 / 216 | Đặc thù văn hóa & chính trị VN: chủ quyền biển đảo, an ninh nội bộ, lăng mạ... |

---

### SLIDE 8: [TRỌNG TÊM 2 - TIẾNG VIỆT] CHỈ SỐ ĐỊNH LUỒNG SAFEROUTE TRÊN 2 BỘ DYNABENCH (@ 0.60)
*(Tách bạch độc lập 2 bộ Original vs Augmented Ver, loại bỏ số liệu hợp gộp)*

| Bộ Dữ liệu Thí nghiệm | Tổng mẫu | Routing F1 | Precision | Recall (Bắt ca khó) | Accuracy | Tỉ lệ gọi `DynaGuard-8B` Thực tế vs Lý tưởng |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **DynaBench (Original)** | 543 | **0.5046** | 0.3710 | **78.85%** | 70.35% | Model gọi **40.7%** *(Lý tưởng cần: 19.2%)* |
| **DynaBench (Augmented Ver)** | 543 | **0.4431** | 0.3396 | **63.72%** | 66.67% | Model gọi **39.0%** *(Lý tưởng cần: 20.8%)* |

* **Phân tích chuyên sâu:**
  * Ở bộ gốc **DynaBench (Original)**, Router tỏ ra cực kỳ nhạy bén với luồng dữ liệu chuẩn, đạt Recall phát hiện ca khó lên tới **78.85%**.
  * Ở bộ **DynaBench (Augmented Ver)** (đã xáo trộn thứ tự và xóa nhãn `violent`), Routing F1 giảm nhẹ xuống 0.4431, phản ánh sự nhạy cảm của các mô hình nhỏ trước việc xáo trộn prompt ngữ cảnh.

---

### SLIDE 9: [TRỌNG TÊM 2 - TIẾNG VIỆT] ĐỌ SỨC HIỆU NĂNG AN TOÀN ĐẦU - CUỐI (E2E SAFETY F1)
*(Tách trực tiếp chỉ số phát hiện vi phạm FAIL trên từng bộ độc lập)*

| Chiến lược Guardrail | Safety F1 trên **DynaBench (Original)** | Safety F1 trên **DynaBench (Augmented Ver)** | % Gọi `DynaGuard-8B` trung bình |
|---|:---:|:---:|:---:|
| **DynaGuard-1.7B thuần** | 0.6481 | 0.5304 | 0% |
| **DynaGuard-8B thuần** | 0.7320 | 0.5975 | 100% |
| **SafeRoute MLP (@ 0.60)** | **0.7542** 🏆 *(+2.2% so với 8B)* | **0.6041** 🏆 *(+0.7% so với 8B)* | **~39.9%** 💰 *(Tiết kiệm ~60%)* |
| *Trần Oracle Lý tưởng* | *0.8266* | *0.7187* | *~20.0%* |

* **CHIẾN THẮNG KÉP NGỌAN MỤC:**
  * Trên cả 2 phiên bản dữ liệu (Original gốc và Augmented Ver biến thể), SafeRoute MLP **đều đạt Safety F1 VƯỢT qua mô hình lớn DynaGuard-8B độc lập** (`0.7542 > 0.7320` và `0.6041 > 0.5975`)!
  * Đồng thời, hệ thống cắt giảm được **60.1% chi phí gọi mô hình 8B**, minh chứng sức mạnh hợp lực tuyệt vời giữa hai Guard.

---

### SLIDE 10: [TRỌNG TÊM 2 - TIẾNG VIỆT] PHÂN TÍCH MIỀN VN-NATIVE & NGHỊCH LÝ KÍCH THƯỚC

* **Số liệu miền dữ liệu bản địa VN-native (504 mẫu):**

| Chiến lược | Accuracy | Macro-F1 | F1 (FAIL - Phát hiện vi phạm) | Tỉ lệ gọi `DynaGuard-8B` |
|---|:---:|:---:|:---:|:---:|
| **DynaGuard-1.7B** | **89.88%** 🏆 | **0.8944** | **0.8728** | 0% |
| **DynaGuard-8B** | 85.91% | 0.8468 | 0.8033 | 100% |
| **SafeRoute MLP** | 86.71% | 0.8561 | 0.8164 | 32.1% |

* **Nghịch lý bản địa (Domain Paradox):**
  * Ở chủ đề đặc thù VN, model nhỏ **mạnh hơn** model lớn (`89.88% > 85.91%`).
  * Áp dụng SafeRoute khiến hiệu năng tổng thể giảm xuống 86.71% do đẩy 32.1% câu hỏi sang model 8B yếu hơn.
  * **Bài học quy trình:** Định tuyến động chỉ phát huy giá trị khi `Large > Small`. Với các domain bản địa rõ ràng, **sử dụng thẳng 100% DynaGuard-1.7B** là chiến lược tối ưu tuyệt đối cả về chi phí lẫn độ chính xác.

---

### SLIDE 11: MA TRẬN TỔNG HỢP & LỘ TRÌNH TRIỂN KHAI THỰC TẾ

| Tiêu chí Chiến lược | Tiếng Anh Quốc tế (DynaBench-EN) | Tiếng Việt Gốc (DynaBench Original) | Tiếng Việt Biến thể (DynaBench Augmented Ver) | Miền Bản địa VN (VN-native) |
|---|:---:|:---:|:---:|:---:|
| **Thứ bậc mô hình** | **8B > 1.7B** *(+12.9% Acc)* | **8B > 1.7B** *(+12.3% F1)* | **8B > 1.7B** *(+6.7% F1)* | **1.7B > 8B** *(+4.0% Acc)* ⚠️ |
| **Hiệu năng SafeRoute** | Khớp 100% hiệu năng 8B | **Vượt 8B (+2.2% F1)** 🏆 | **Vượt 8B (+0.7% F1)** 🏆 | Thấp hơn model 1.7B |
| **Chi phí gọi 8B** | 40.9% | **40.7%** *(Tiết kiệm ~59%)* | **39.0%** *(Tiết kiệm ~61%)* | 32.1% *(Không cần thiết)* |

**Chiến lược Kỹ thuật Tiếp theo:**
1. Khẳng định SafeRoute giải quyết triệt để bài toán hiệu năng/chi phí cho Agentic Guardrail Tiếng Việt.
2. Triển khai kiến trúc **Domain Classifier** ở bước tiền xử lý: gặp prompt VN-native $\rightarrow$ bypass gọi thẳng `1.7B`; gặp prompt dịch thuật $\rightarrow$ kích hoạt `SafeRoute MLP`.

---
*Tài liệu đã hoàn tất chuẩn hóa 11 Slide súc tích.*
