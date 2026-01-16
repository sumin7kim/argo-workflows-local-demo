import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

DATA_DIR = "/mnt/workspace/data"
ARTIFACT_DIR = "/mnt/workspace/artifacts"

def train_model():
    print("🔥 [Step 3] Training Model...")
    
    # 1. Load Processed Data
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    
    # 2. Train Model (Random Forest)
    print("Training RandomForestRegressor... (this may take a moment)")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # 3. Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)
    
    print(f"📊 Evaluation Results:")
    print(f"   - RMSE: {rmse:.2f}")
    print(f"   - R2 Score: {r2:.4f}")
    
    # 4. Save Model
    model_path = os.path.join(ARTIFACT_DIR, "housing_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    train_model()