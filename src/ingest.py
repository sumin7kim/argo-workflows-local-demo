import os
import requests
import pandas as pd

# 데이터 저장 경로 (Volume Mount 경로)
DATA_DIR = "/mnt/workspace/data"
os.makedirs(DATA_DIR, exist_ok=True)

def ingest_data():
    print("🚀 [Step 1] Ingesting Data...")
    
    # 캘리포니아 주택 데이터셋 URL (CSV)
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    save_path = os.path.join(DATA_DIR, "raw_housing.csv")
    
    print(f"Downloading from {url}...")
    response = requests.get(url)
    with open(save_path, "wb") as f:
        f.write(response.content)
        
    print(f"✅ Data saved to {save_path}")
    
    # 데이터 확인
    df = pd.read_csv(save_path)
    print(f"Data Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    ingest_data()