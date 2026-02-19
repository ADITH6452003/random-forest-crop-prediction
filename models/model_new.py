import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

df=pd.read_csv('/home/adith6452/Documents/cropprediction/data/crop_production.csv')

freq = df['State_Name'].value_counts()
df['State_Name'] = df['State_Name'].map(freq)

df = pd.get_dummies(df, columns=['Season'])

freq_map = df['District_Name'].value_counts()
df['District_Name'] = df['District_Name'].map(freq_map)

le = LabelEncoder()
df['Crop'] = le.fit_transform(df['Crop'])

x = df.drop(['Crop', 'Production'], axis=1)
y_c = df['Crop']

print('model spliting')
x_train, x_test, y_train, y_test = train_test_split(x, y_c, random_state=42, test_size=0.2)


rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf}")  
acc_train_rf = accuracy_score(y_train, rf.predict(x_train))
print(f"Random Forest Train Accuracy: {acc_train_rf}")