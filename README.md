<div align="center">

# Predicting Product Sales

### From-scratch Time-Series Forecasting Pipeline with Linear Regression, Decision Tree, and Online Learning (Fine-Tuning)

<p>
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python 3.10" />
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter" alt="Jupyter Notebook" />
  <img src="https://img.shields.io/badge/Task-Time--Series%20Forecasting-brightgreen?style=for-the-badge" alt="Time-Series Forecasting" />
  <img src="https://img.shields.io/badge/Core-From%20Scratch-purple?style=for-the-badge" alt="From Scratch" />
  <img src="https://img.shields.io/badge/Models-Linear%20Regression%20%2B%20Decision%20Tree-red?style=for-the-badge" alt="Models" />
</p>

<table>
  <tr>
    <td><b>Notebook chính</b></td>
    <td><code>notebooks/04_timeseries_forecasting.ipynb</code> & <code>notebooks/05_online_learning.ipynb</code></td>
  </tr>
  <tr>
    <td><b>Dữ liệu</b></td>
    <td><code>data/raw/ecommerce_dataset_updated.csv</code></td>
  </tr>
  <tr>
    <td><b>Bài toán</b></td>
    <td>Dự báo Chuỗi thời gian (Time-Series Forecasting) cho Số lượng đơn hàng (Daily Sales Volume)</td>
  </tr>
  <tr>
    <td><b>Nguyên tắc chính</b></td>
    <td>100% thuật toán tự lập trình bằng Toán học (From Scratch), không sử dụng thư viện scikit-learn.</td>
  </tr>
</table>

<p>
  Project này xây dựng toàn bộ quy trình dự báo Doanh số bán hàng, đi từ dữ liệu raw, 
  data preparing, feature engineering, model selection, training, evaluation, 
  kỹ thuật Online Learning đến inference trực tiếp qua Web API.
</p>

</div>

---

## 1. Mục tiêu dự án

Ban đầu, dữ liệu (`ecommerce_dataset_updated.csv`) cung cấp lịch sử giao dịch. Thông thường, sinh viên sẽ lấy giá trị `Price` và `Discount` để đoán `Final_Price`. Tuy nhiên, điều này gây ra hiện tượng rò rỉ dữ liệu (Data Leakage) vì công thức tính là cố định.

Để giải quyết triệt để và mang lại giá trị thực tiễn cho doanh nghiệp, bài toán đã được chuyển đổi sang: **Dự báo Chuỗi thời gian (Time-Series Forecasting) cho Tổng số lượng Đơn hàng Bán ra mỗi ngày (Daily Sales Volume).** Việc dự đoán đúng số lượng đơn hàng trong tương lai giúp công ty tối ưu hóa kho bãi, nhân sự và chiến dịch Marketing.

Mục tiêu chính:
- Xây dựng pipeline Machine Learning hoàn chỉnh cho bài toán Time-Series Forecasting.
- Trực quan hóa dữ liệu (EDA), phân tích xu hướng bán hàng theo ngày/tháng/năm.
- Triển khai các thành phần học máy cốt lõi theo hướng from scratch, không dùng model/scaler có sẵn từ scikit-learn.
- So sánh hai thuật toán: Linear Regression (Tuyến tính) và Decision Tree Regressor (Phi tuyến tính).
- Khắc phục hiện tượng Rò rỉ dữ liệu bằng Custom StandardScaler và Time-Series Split.
- Tích hợp kỹ thuật **Online Learning (Fine-Tuning)** bằng Gradient Descent để mô hình tự học thêm kiến thức mỗi ngày.
- Đóng gói hoàn chỉnh thành một ứng dụng Web API (Flask) sẵn sàng cho Production.

---

## 2. Dữ liệu sử dụng & Xử lý (Data Overview)

Nguồn dữ liệu gốc chứa lịch sử giao dịch: `data/raw/ecommerce_dataset_updated.csv`.

### 2.1. Phân tích Dữ liệu thô (Raw Data)

**Bảng Preview dữ liệu gốc:**

| User_ID | Product_ID | Category | Price (Rs.) | Discount (%) | Final_Price | Payment_Method | Purchase_Date |
|---|---|---|---|---|---|---|---|
| 337c166f | f414122f-e | Sports | 36.53 | 15 | 31.05 | Net Banking | 12-11-2024 |
| d38a19bf | fde50f9c-5 | Clothing | 232.79 | 20 | 186.23 | Net Banking | 09-02-2024 |

**Vai trò của các cột trong dữ liệu gốc:**

| Cột | Ý nghĩa | Trạng thái sử dụng |
|---|---|---|
| `User_ID`, `Product_ID` | Mã định danh khách hàng và sản phẩm. | ❌ Bỏ qua (Không mang tính dự báo vĩ mô) |
| `Category`, `Payment_Method` | Phân loại hàng hóa và hình thức thanh toán. | ❌ Bỏ qua trong bài toán Chuỗi thời gian |
| `Price`, `Discount`, `Final_Price` | Thông tin giá cả của 1 đơn hàng cụ thể. | ❌ Bỏ qua (Để tránh hiện tượng Rò rỉ dữ liệu - Data Leakage) |
| `Purchase_Date` | Ngày giao dịch (Định dạng DD-MM-YYYY). | ✅ **Giữ lại làm nòng cốt để tạo Chuỗi thời gian** |

### 2.2. Chuyển đổi thành Chuỗi thời gian (Time-Series Transformation)

Thay vì dự đoán cột giá tiền cho từng khách hàng nhỏ lẻ, toàn bộ các giao dịch trong cùng 1 ngày được **gom nhóm (groupby)** theo `Purchase_Date` để đếm tổng số lượng đơn. Qua đó tạo ra một biến mục tiêu (Target Variable) mang tính chiến lược vĩ mô hơn cho doanh nghiệp: `Daily_Volume` (Tổng số lượng đơn hàng/ngày).

### 2.3. Trích xuất Đặc trưng (Feature Engineering)

Từ một cột thời gian duy nhất, hệ thống tự động bóc tách (Feature Extraction) thành các đặc trưng số học để đưa vào mô hình học máy:

| Đặc trưng (Feature) | Ý nghĩa | Khung giá trị |
|---|---|---|
| `Purchase_Year` | Năm giao dịch | 2024 |
| `Purchase_Month` | Tháng giao dịch | 1 đến 12 |
| `Purchase_Day` | Ngày trong tháng | 1 đến 31 |
| `Purchase_DayOfWeek` | Ngày trong tuần | 0 (Thứ 2) đến 6 (Chủ nhật) |
| `Purchase_IsWeekend` | Cờ đánh dấu ngày nghỉ (Thứ 7, CN) | 0 (Ngày thường), 1 (Cuối tuần) |
| **`Daily_Volume` (Target)** | **Tổng số lượng đơn hàng bán được trong ngày đó** | Số nguyên dương (vd: 120, 150...) |

**Dữ liệu đặc trưng sau xử lý (Model Input Preview - `ecommerce_features.csv`):**

| Purchase_Year | Purchase_Month | Purchase_Day | Purchase_DayOfWeek | Purchase_IsWeekend | Target: Daily_Volume |
|---|---|---|---|---|---|
| 2024 | 1 | 1 | 0 | 0 | **120** |
| 2024 | 1 | 2 | 1 | 0 | **95** |
| 2024 | 1 | 6 | 5 | 1 | **150** |

Toàn bộ quá trình biến đổi phức tạp này được xử lý bởi logic tự động trong `02_feature_engineering.ipynb`. Dữ liệu sạch cuối cùng được xuất ra file `data/processed/ecommerce_features.csv` để sẵn sàng cho các mô hình AI.

---

## 3. Cấu trúc thư mục hiện tại

```text
Predicting_Product_Sales/
├── README.md
├── app.py
├── requirements.txt
├── data/
│   ├── raw/
│   │   └── ecommerce_dataset_updated.csv
│   └── processed/
│       └── ecommerce_features.csv
├── models/
│   ├── linear_regression.pkl
│   ├── decision_tree.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_baseline.ipynb
│   ├── 04_timeseries_forecasting.ipynb
│   └── 05_online_learning.ipynb
└── src/
    ├── data/
    ├── features/
    └── models/
        ├── linear_regression.py
        ├── decision_tree.py
        ├── linear_regression_gd.py
        ├── preprocessing.py
        └── metrics.py
```

| Thành phần | Vai trò |
|---|---|
| `data/raw/ecommerce_dataset_updated.csv` | File dữ liệu e-commerce gốc. |
| `notebooks/` | Chứa toàn bộ các bước nghiên cứu: EDA, Feature Engineering, Training, và Online Learning. |
| `src/models/` | Core Engine: Chứa mã nguồn thuật toán được **tự lập trình từ đầu** bằng Toán học. |
| `app.py` | Web API Deployment bằng Flask để phục vụ dự đoán theo thời gian thực. |

---

## 4. From-scratch compliance (Điểm nhấn Kỹ thuật)

Các thành phần cốt lõi được tự triển khai 100% bằng NumPy và Pandas:

| Thành phần | Class / logic chính | Ghi chú |
|---|---|---|
| Chống Rò rỉ Dữ liệu | `StandardScaler` | Tự viết Class chuẩn hóa dữ liệu. Chỉ học Mean/Std trên Train set, sau đó Transform lên Test set. |
| Time-Series Split | `time_series_split` | Tự code hàm chia dữ liệu theo đúng trật tự thời gian thay vì xáo trộn ngẫu nhiên. |
| Linear Regression | `LinearRegression` | Tự giải thuật toán bằng công thức Ma trận giả nghịch đảo (Normal Equation). |
| Decision Tree | `DecisionTreeRegressor` | Tự code thuật toán rẽ nhánh đệ quy dựa trên tiêu chí **Giảm phương sai (Variance Reduction)**. |
| Online Learning | `LinearRegressionGD` | Thuật toán Hạ dốc đạo hàm có tính năng `partial_fit` để học liên tục. |
| Chống Overreacting | `alpha`, `clip_value` | Áp dụng L2 Regularization và Gradient Clipping để giới hạn Trọng số, chống nhiễu dữ liệu. |
| Metrics | `mean_absolute_error`, `root_mean_squared_error` | Tự code các hàm đánh giá sai số. |

---

## 5. Hướng dẫn Chạy Dự án (How to Run)

### A. Chạy Khối lượng Nghiên cứu (Notebooks)
1. Cài đặt các thư viện cơ bản: 
   ```bash
   pip install -r requirements.txt
   ```
2. Khởi chạy Jupyter Notebook hoặc mở trực tiếp trên VS Code.
3. Chạy lần lượt các notebook từ `01` đến `05`. 
*(Lưu ý: Mọi thuật toán tính toán bên dưới đều lấy từ thư mục `src/models/` do sinh viên tự lập trình).*

### B. Chạy Triển khai API (Model Deployment)
1. Tại thư mục gốc của dự án, mở Terminal và gõ:
   ```bash
   python app.py
   ```
2. Mở một Terminal khác (hoặc dùng Postman), gửi dữ liệu ngày tháng giả định vào API để lấy kết quả dự đoán:
   *(Ví dụ dùng PowerShell trên Windows)*
   ```powershell
   Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -ContentType "application/json" -Body '{"Purchase_Year": 2024, "Purchase_Month": 12, "Purchase_Day": 24, "Purchase_DayOfWeek": 1, "Purchase_IsWeekend": 0}'
   ```
3. Hệ thống sẽ trả về số lượng đơn hàng dự báo bán được trong ngày Noel 24/12/2024.

---


Thank you for visiting this project.
