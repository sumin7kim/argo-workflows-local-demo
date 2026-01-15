import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 로그가 Argo UI에서 즉시 보이도록 설정
def log(msg):
    print(f"[Pipeline] {msg}", flush=True)

def main():
    log("🚀 ML Pipeline Started...")

    # 1. 더미 데이터 생성 (Data Generation)
    log("Creating dummy dataset...")
    X = np.random.rand(100, 4)  # 100개의 샘플, 4개의 feature
    y = np.random.randint(0, 2, 100) # 0 또는 1 (이진 분류)
    
    df = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4'])
    df['target'] = y
    
    # 데이터 저장 (볼륨에 저장됨)
    data_path = "dataset.csv"
    df.to_csv(data_path, index=False)
    log(f"Dataset saved to {os.path.abspath(data_path)}")

    # 2. 데이터 전처리 (Preprocessing)
    log("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('target', axis=1), 
        df['target'], 
        test_size=0.2, 
        random_state=42
    )

    # 3. 모델 학습 (Training)
    log("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # 4. 평가 (Evaluation)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    log(f"🔥 Model Accuracy: {acc:.4f}")

    # 5. 모델 저장 (Model Saving)
    model_path = "model.pkl"
    joblib.dump(model, model_path)
    log(f"✅ Model saved to {os.path.abspath(model_path)}")
    
    log("🎉 Pipeline Finished Successfully.")

if __name__ == "__main__":
    main()