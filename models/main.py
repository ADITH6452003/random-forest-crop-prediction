print("XGBoost accuraacy and prediction\n")

import joblib
import pandas as pd
import pandas as pd
df=pd.read_csv("/home/adith6452/Documents/cropprediction/data/Crop_Data.xlsx.csv")

# Load model
xgb = joblib.load("xgb_crop_model.pkl")
crop_map = joblib.load("crop_label_map.pkl")

sample = pd.DataFrame(
    [[29, 75, 6.8, 180]],
    columns=['temperature', 'humidity', 'ph', 'rainfall']
)
proba = xgb.predict_proba(sample)[0]

# Create mapping
crop_map = dict(zip(df['Label_Num'], df['label']))

# Convert to DataFrame
proba_df = pd.DataFrame({
    'Crop': [crop_map[i] for i in range(len(proba))],
    'Suitability_Probability': proba
})
# Sort by probability
proba_df = proba_df.sort_values(by='Suitability_Probability', ascending=False)
# Select Top 5 crops
top_crops = proba_df.head(3)
print(top_crops)
from sklearn.model_selection import train_test_split
x = df[['temperature', 'humidity', 'ph', 'rainfall']]
y = df['Label_Num'] 

x_train,x_text,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2,stratify=y)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, xgb.predict(x_text))
print("accuracy = ",accuracy)
print("Prediction completed without retraining.")
