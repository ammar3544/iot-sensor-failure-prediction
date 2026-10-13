import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# ================= ================= =================
# 1. SIMULASI DATA STREAMING SENSOR IoT
# ================= ================= =================
np.random.seed(42)
n_samples = 5000

data = {
    'sensor_id': np.random.choice([f'SNSR-{i:03d}' for i in range(1, 11)], size=n_samples),
    'temperature': np.random.normal(loc=35.0, scale=5.0, size=n_samples),
    'humidity': np.random.normal(loc=60.0, scale=10.0, size=n_samples),
    'vibration': np.random.normal(loc=0.5, scale=0.2, size=n_samples),
    'power_voltage': np.random.normal(loc=12.0, scale=1.0, size=n_samples),
    # Target: Gagal/Anomali (1) dengan rasio kecil ~10% (Class Imbalance)
    'is_failure': np.random.choice([0, 1], size=n_samples, p=[0.90, 0.10]),
    # FEATURE LEAKAGE (Variabel yang baru diketahui setelah perangkat benar-benar rusak)
    'maintenance_downtime_hours': np.zeros(n_samples),
}

df = pd.DataFrame(data)
# Simulasi data leakage: Mengisi waktu downtime berdasarkan status gagal
df.loc[df['is_failure'] == 1, 'maintenance_downtime_hours'] = np.random.uniform(
    24, 72, size=(df['is_failure'] == 1).sum()
)

print(f'Total Dataset Sensor IoT: {len(df)} baris')
print(f"Rasio Kegagalan (Failure Rate): {df['is_failure'].mean()*100:.2f}%\n")

# ================= ================= =================
# 2. DATA PREPARATION & PENCEGAHAN DATA LEAKAGE
# ================= ================= =================
# WAJIB: Buang kolom ID dan 'maintenance_downtime_hours' (Data Leakage)
X = df.drop(columns=['sensor_id', 'is_failure', 'maintenance_downtime_hours'])
y = df['is_failure']

# Split Data secara stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= ================= =================
# 3. EKSPERIMEN MODEL & HANDLING IMBALANCED DATA
# ================= ================= =================
print('--- MODEL 1: BASELINE (Tanpa Penanganan Khusus) ---')
rf_base = RandomForestClassifier(random_state=42)
rf_base.fit(X_train, y_train)
y_pred_base = rf_base.predict(X_test)
print(classification_report(y_test, y_pred_base, target_names=['Normal', 'Failure']))

print('--- MODEL 2: DENGAN CLASS WEIGHT BALANCED ---')
rf_weighted = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_weighted.fit(X_train, y_train)
y_pred_weighted = rf_weighted.predict(X_test)
print(
    classification_report(y_test, y_pred_weighted, target_names=['Normal', 'Failure'])
)

print('--- MODEL 3: DENGAN SMOTE (OVERSAMPLING) ---')
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_res, y_train_res)
y_pred_smote = rf_smote.predict(X_test)
print(classification_report(y_test, y_pred_smote, target_names=['Normal', 'Failure']))