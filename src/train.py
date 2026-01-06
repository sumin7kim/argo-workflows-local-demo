import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

DATA_PATH = "/mnt/data"

print("Loading Data...")
with open(os.path.join(DATA_PATH, "train_data.pkl"), "rb") as f:
    X_train, y_train = pickle.load(f)
with open(os.path.join(DATA_PATH, "test_data.pkl"), "rb") as f:
    X_test, y_test = pickle.load(f)

print("Training Model...")
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 평가
predictions = clf.predict(X_test)
acc = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {acc:.4f}")

# 모델 저장
model_path = os.path.join(DATA_PATH, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Model saved to {model_path}")