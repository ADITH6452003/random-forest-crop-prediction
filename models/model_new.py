import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import dataset
df = pd.read_csv("/home/adith6452/Documents/cropprediction/data/crop_production.csv")

#data set datatypes
print(df.info())
# State name frequency encoding
freq = df['State_Name'].value_counts()
df['State_Name'] = df['State_Name'].map(freq)

# Season one-hot encoding
df = pd.get_dummies(df, columns=['Season'])

# District frequency encoding
freq_map = df['District_Name'].value_counts()
df['District_Name'] = df['District_Name'].map(freq_map)

# Crop label encoding
le = LabelEncoder()
df['Crop'] = le.fit_transform(df['Crop'])

# Split features and target
x = df.drop(['Crop', 'Production'], axis=1)
y_c = df['Crop']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y_c, random_state=42, test_size=0.2)

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc}")
