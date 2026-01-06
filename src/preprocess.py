import os
import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 데이터 저장 경로 (PVC 마운트 경로 예정)
DATA_PATH = "/mnt/data"
os.makedirs(DATA_PATH, exist_ok=True)

print("Starting Preprocessing...")
iris = load_iris()
X, y = iris.data, iris.target

# 학습/테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 저장 (피클링)
with open(os.path.join(DATA_PATH, "train_data.pkl"), "wb") as f:
    pickle.dump((X_train, y_train), f)
with open(os.path.join(DATA_PATH, "test_data.pkl"), "wb") as f:
    pickle.dump((X_test, y_test), f)

print(f"Data saved to {DATA_PATH}")
