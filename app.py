from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import sys
import os

# Import custom modules từ thư mục src
from src.models.decision_tree import DecisionTreeRegressor, Node
from src.models.preprocessing import StandardScaler

app = Flask(__name__)

# Khởi tạo biến toàn cục cho model và scaler
model = None
scaler = None

def load_models():
    """Hàm nạp mô hình và scaler khi server khởi động"""
    global model, scaler
    try:
        model = DecisionTreeRegressor().load('./models/decision_tree.pkl')
        scaler = StandardScaler().load('./models/scaler.pkl')
        print("Đã nạp mô hình Decision Tree và Scaler thành công!")
    except Exception as e:
        print(f"Lỗi nạp mô hình: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint dự đoán doanh số.
    Nhận dữ liệu dạng JSON:
    {
        "Purchase_Year": 2024,
        "Purchase_Month": 1,
        "Purchase_Day": 15,
        "Purchase_DayOfWeek": 1,
        "Purchase_IsWeekend": 0
    }
    """
    if model is None or scaler is None:
        return jsonify({"error": "Mô hình chưa được nạp (Model not loaded)"}), 500

    try:
        # Lấy dữ liệu từ Request
        data = request.get_json()
        
        # Đưa vào DataFrame (1 dòng)
        df = pd.DataFrame([data])
        
        # Các cột dạng số cần chuẩn hóa
        numeric_cols = ['Purchase_Year', 'Purchase_Month', 'Purchase_Day', 'Purchase_DayOfWeek']
        
        # Transform dữ liệu (Không fit lại)
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
        
        # Dự đoán
        prediction = model.predict(df_scaled)
        
        # Trả về kết quả
        return jsonify({
            "status": "success",
            "input_data": data,
            "predicted_daily_volume": float(prediction[0])
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "API đang hoạt động bình thường!"})

@app.route('/', methods=['GET'])
def index():
    return """
    <h1>Predicting Product Sales API</h1>
    <p>Hệ thống dự báo doanh số bán hàng đang hoạt động.</p>
    <ul>
        <li>Trạng thái: <a href="/health">/health</a> (GET)</li>
        <li>Dự đoán: <code>/predict</code> (POST)</li>
    </ul>
    """

if __name__ == '__main__':
    load_models()
    # Chạy server ở port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
