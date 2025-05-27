import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 設定資料夾與分類
BASE_PATH = './'
CATEGORIES = [
    'U.S. Fancy apples',
    'U.S. No.1 apples',
    'U.S. No.2 apples',
    'Inedible (or process) apples'
]
CATEGORY_LABELS = {category: idx for idx, category in enumerate(CATEGORIES)}

# 建立資料集
data = []
labels = []

for category in CATEGORIES:
    folder_path = os.path.join(BASE_PATH, category)
    label = CATEGORY_LABELS[category]

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 圖像預處理
        img = cv2.resize(img, (100, 100))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # 壓平成一維向量
        img_flat = img.flatten()

        data.append(img_flat)
        labels.append(label)

# 轉成 numpy array
X = np.array(data)
y = np.array(labels)

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# 驗證模型
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ 模型準確率：{acc:.2f}")

# 儲存模型（用 pickle）
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ 模型已儲存為 model.pkl")
