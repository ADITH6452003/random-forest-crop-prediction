import pandas as pd
import numpy as np
df = pd.read_csv("/home/adith6452/Documents/cropprediction/data/crop_production.csv")
print(df.info())
cat_cols = df.select_dtypes(include="object").columns
print(cat_cols)

for col in cat_cols:
    print(col, df[col].nunique())


df = pd.get_dummies(df, columns=["Season"])

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ["District_Name", "Crop" , "State_Name"]:
    df[col] = le.fit_transform(df[col])
x=df.drop(['Crop','Production'],axis=1)
y_c=df['Crop']
y_prod = df['Production']




from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y_c,random_state=42,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(random_state=42)
rf.fit(x_train,y_train)
y_pred = rf.predict(x_test)

from sklearn.metrics import accuracy_score,f1_score
acc = accuracy_score(y_test,y_pred)
print(acc)

