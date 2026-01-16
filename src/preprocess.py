import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib

DATA_DIR = "/mnt/workspace/data"
ARTIFACT_DIR = "/mnt/workspace/artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def preprocess_data():
    print("⚙️ [Step 2] Preprocessing Data...")
    
    # 1. Load Raw Data
    raw_path = os.path.join(DATA_DIR, "raw_housing.csv")
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"{raw_path} not found. Run ingest first.")
    
    df = pd.read_csv(raw_path)
    
    # 2. Feature Engineering (간단한 예시)
    # 범주형 데이터 'ocean_proximity' 제거 (데모 복잡도 조절)
    df = df.drop("ocean_proximity", axis=1)
    
    # 3. 결측치 처리 (중위값으로 채움)
    imputer = SimpleImputer(strategy="median")
    X = df.drop("median_house_value", axis=1)
    y = df["median_house_value"]
    
    X_filled = imputer.fit_transform(X)
    
    # 4. Scaling (정규화)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    
    # Scaler 저장 (추후 추론을 위해)
    joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
    
    # 5. Split Train/Test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 6. Save Processed Data
    np.save(os.path.join(DATA_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(DATA_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(DATA_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(DATA_DIR, "y_test.npy"), y_test)
    
    print(f"✅ Preprocessing Complete. Train shape: {X_train.shape}")

if __name__ == "__main__":
    preprocess_data()