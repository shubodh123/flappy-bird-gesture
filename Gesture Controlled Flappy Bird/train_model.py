import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("hand_landmarks.csv")

X = df.drop(columns=["label"])
y = df["label"]

label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y = y.map(label_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "gesture_model.pkl")
joblib.dump(label_mapping, "label_mapping.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")  # Save feature names

accuracy = model.score(X_test, y_test)
print(f"Model trained! Accuracy: {accuracy:.2f}")
