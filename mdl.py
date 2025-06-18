import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# Load dataset
df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")
df = df.dropna()  # Drop missing rows

# Define features and target
target_column = "target"
X = df.drop(columns=['patientid', target_column])
y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features (done here, but NOT saved)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Synthetic Minority Over-sampling Technique-Handle class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Train model
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability = True,random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Save model and scaler
with open("cvd_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)

with open("cvd_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved successfully.")


