import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import top_k_accuracy_score

# =========================
# LOAD DATA
# =========================
df = pd.read_csv('/home/adith6452/Documents/cropprediction/data/crop_production.csv')

# =========================
# BASIC CLEANING
# =========================
df = df.drop_duplicates()

# merge rare crop classes BEFORE encoding
counts = df['Crop'].value_counts()
rare = counts[counts < 120].index
df['Crop'] = df['Crop'].replace(rare, "other")

# label encode target AFTER merge
le = LabelEncoder()
df['Crop'] = le.fit_transform(df['Crop'])

# =========================
# SPLIT FIRST (PREVENT LEAKAGE)
# =========================
X_raw = df.drop(['Crop', 'Production'], axis=1)
y = df['Crop']

x_train_raw, x_test_raw, y_train, y_test = train_test_split(
    X_raw, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# FIT ENCODERS ON TRAIN ONLY
# =========================

# frequency encoding — train only
state_freq = x_train_raw['State_Name'].value_counts()
dist_freq  = x_train_raw['District_Name'].value_counts()

def apply_geo_encoding(frame):
    frame = frame.copy()
    frame['State_Name'] = frame['State_Name'].map(state_freq).fillna(0)
    frame['District_Name'] = frame['District_Name'].map(dist_freq).fillna(0)
    return frame

x_train = apply_geo_encoding(x_train_raw)
x_test  = apply_geo_encoding(x_test_raw)

# one-hot encode season using train columns
x_train = pd.get_dummies(x_train, columns=['Season'])
x_test  = pd.get_dummies(x_test, columns=['Season'])

# align columns so test matches train
x_test = x_test.reindex(columns=x_train.columns, fill_value=0)

# =========================
# OUTLIER CLIPPING (NUMERIC ONLY)
# =========================
num_cols = ['Area', 'Crop_Year']

for c in num_cols:
    lo = x_train[c].quantile(0.02)
    hi = x_train[c].quantile(0.98)
    x_train[c] = x_train[c].clip(lo, hi)
    x_test[c]  = x_test[c].clip(lo, hi)

# =========================
# RANDOM FOREST — REGULARIZED
# =========================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=30,
    min_samples_leaf=15,
    max_features=0.5,
    bootstrap=True,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train, y_train)

# =========================
# FEATURE IMPORTANCE PRUNING
# =========================
imp = pd.Series(rf.feature_importances_, index=x_train.columns)
drop_cols = imp[imp < 0.01].index

x_train2 = x_train.drop(columns=drop_cols)
x_test2  = x_test.drop(columns=drop_cols)

# retrain after pruning
rf.fit(x_train2, y_train)

# =========================
# TOP-6 PREDICTION
# =========================
probs = rf.predict_proba(x_test2)

top6_idx = np.argsort(probs, axis=1)[:, -6:]
top6_crops = [le.inverse_transform(row) for row in top6_idx]
print("Top 6 crops for first 3 samples:")
for i in range(min(3, len(top6_crops))):
    print(f"Sample {i+1}: {top6_crops[i]}")


# =========================
# TOP-6 TEST ACCURACY
# =========================
top6_acc = top_k_accuracy_score(y_test, probs, k=6)
print("Top-6 Test Accuracy:", round(top6_acc, 3))

# =========================
# STRATIFIED TOP-6 CV
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = []

for tr_idx, val_idx in cv.split(x_train2, y_train):
    Xtr = x_train2.iloc[tr_idx]
    Xval = x_train2.iloc[val_idx]
    ytr = y_train.iloc[tr_idx]
    yval = y_train.iloc[val_idx]

    rf.fit(Xtr, ytr)
    p = rf.predict_proba(Xval)
    score = top_k_accuracy_score(yval, p, k=6)
    cv_scores.append(score)

print("Top-6 Stratified CV:", round(np.mean(cv_scores), 3))
