import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load dataset
df = pd.read_csv(r"C:\Mental\student_depression_dataset.csv")

# Drop non-informative columns
df = df.drop(columns=['id', 'City', 'Profession', 'Degree'], errors='ignore')

# Convert range values to average float
def convert_range_to_avg(val):
    try:
        val = str(val).replace("'", "").strip()
        if "-" in val:
            low, high = map(float, val.split("-"))
            return (low + high) / 2
        return float(val)
    except:
        return np.nan

# Replace '?' with NaN
df = df.replace("?", np.nan)

# Clean range fields
df['Sleep Duration'] = df['Sleep Duration'].apply(convert_range_to_avg)
df['Work/Study Hours'] = df['Work/Study Hours'].apply(convert_range_to_avg)

# Convert to numeric and fill missing
for col in ['Sleep Duration', 'Work/Study Hours']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].mean())

# Fill missing categorical values with mode
for col in ['Gender', 'Academic Pressure', 'Dietary Habits']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Drop rows where target is missing
df = df.dropna(subset=['Have you ever had suicidal thoughts ?'])

# Encode categorical columns
label_encoders = {}
categorical_cols = ['Gender', 'Academic Pressure', 'Dietary Habits']
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target
df['Have you ever had suicidal thoughts ?'] = df['Have you ever had suicidal thoughts ?'].map({'Yes': 1, 'No': 0})

# Separate features and target
target_col = 'Have you ever had suicidal thoughts ?'
X = df.drop(columns=[target_col])
y = df[target_col]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model and scaler
joblib.dump(model, "mental_health_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
