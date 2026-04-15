import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("crop_dataset.csv")

print("Dataset Shape:", df.shape)


# ==========================
# Remove Duplicate Rows
# ==========================

df = df.drop_duplicates()


# ==========================
# Remove Leakage Columns
# ==========================

drop_cols = [
    "TYPE_OF_CROP",
    "SOWN",
    "HARVESTED",
    "CROPDURATION",
    "CROPDURATION_MAX",
    "WATERREQUIRED",
    "WATERREQUIRED_MAX"
]

df = df.drop(columns=drop_cols)


# ==========================
# Convert Range Columns
# ==========================

df["SOIL_PH"] = (df["SOIL_PH"] + df["SOIL_PH_HIGH"]) / 2
df["TEMP"] = (df["TEMP"] + df["MAX_TEMP"]) / 2
df["RELATIVE_HUMIDITY"] = (df["RELATIVE_HUMIDITY"] + df["RELATIVE_HUMIDITY_MAX"]) / 2
df["N"] = (df["N"] + df["N_MAX"]) / 2
df["P"] = (df["P"] + df["P_MAX"]) / 2
df["K"] = (df["K"] + df["K_MAX"]) / 2


# Remove MAX columns
df = df.drop(columns=[
    "SOIL_PH_HIGH",
    "MAX_TEMP",
    "RELATIVE_HUMIDITY_MAX",
    "N_MAX",
    "P_MAX",
    "K_MAX"
])


# ==========================
# Encode Categorical Columns
# ==========================

cat_cols = ["SOIL", "SEASON", "WATER_SOURCE"]

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le


# Encode target
target_encoder = LabelEncoder()
df["CROPS"] = target_encoder.fit_transform(df["CROPS"])


# ==========================
# Feature Selection
# ==========================

X = df.drop("CROPS", axis=1)
y = df["CROPS"]


# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

g
# ==========================
# Train Model
# ==========================

model = RandomForestClassifier(

    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1

)

model.fit(X_train, y_train)


# ==========================
# Evaluate Model
# ==========================

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\nTrain Accuracy:", train_acc)
print("Test Accuracy:", test_acc)


# ==========================
# Save Model
# ==========================

joblib.dump(model, "crop_model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(target_encoder, "crop_encoder.pkl")

print("\nModel saved successfully")


from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)

print("Cross Validation Accuracy:", scores.mean())
print("Fold scores:", scores)
