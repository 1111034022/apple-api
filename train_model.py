import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# 1. 資料夾設定
BASE_PATH = './'
CATEGORIES = [
    'U.S. Fancy apples',
    'U.S. No.1 apples',
    'U.S. No.2 apples',
    'Inedible (or processed) apples'
]

# 2. 資料與標籤蒐集
data = []
labels = []

for idx, category in enumerate(CATEGORIES):
    folder = os.path.join(BASE_PATH, category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        try:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))
            img_flat = img.flatten()
            data.append(img_flat)
            labels.append(idx)
        except:
            continue

data = np.array(data)
labels = np.array(labels)

# 3. 訓練模型
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# 4. 儲存模型為 model.pkl
with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

print("✅ 模型已儲存為 model.pkl")
