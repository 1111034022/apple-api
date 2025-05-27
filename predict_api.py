from flask import Flask, request, jsonify
import numpy as np
import cv2
import pickle

# 初始化 Flask App
app = Flask(__name__)

# 載入模型與等級設定（使用 pickle）
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

GRADE_MAPPING = {
    0: 'U.S. Fancy',
    1: 'U.S. No.1',
    2: 'U.S. No.2',
    3: 'Cider'
}

PRICE_MAPPING = {
    'U.S. Fancy': 56.3 * 0.9,
    'U.S. No.1': 56.3 * 0.6,
    'U.S. No.2': 56.3 * 0.3,
    'Cider': 56.3 * 0.1
}

# API 路由：POST 圖片並回傳預測結果
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']

    try:
        # 讀取圖片內容並預處理
        img_bytes = file.read()
        img_np = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image'}), 400

        # 預處理與模型一致：調整大小、轉 HSV、直方圖均衡化
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # 壓平成向量
        img_flat = img.flatten().reshape(1, -1)

        # 預測
        prediction = model.predict(img_flat)[0]
        grade = GRADE_MAPPING.get(prediction, 'Unknown')
        price = round(PRICE_MAPPING.get(grade, 0), 1)

        return jsonify({
            'grade': grade,
            'price': price,
            'message': f"📦 預測等級：{grade}，建議售價：NT${price}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# 啟動 Flask app（Render 會自動分配 port）
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
