import pandas as pd
df=pd.read_csv("/home/adith6452/Documents/cropprediction/data/Crop_Data.xlsx.csv")
# print(df.head())
# print("missing values per column")
# print(df.isnull().sum())
# print("statastical summary")
# print(df.describe())
# Features and target
x = df[['temperature', 'humidity', 'ph', 'rainfall']]
y = df['Label_Num'] 

#import training and testing data

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2,stratify=y)


#random forest model training

from xgboost import XGBClassifier

from xgboost import XGBClassifier

xgb = XGBClassifier(
    objective="multi:softprob",   # probability output
    num_class=22,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(x_train, y_train)

#to convert thee output in numbers to the feature values
crop_map = dict(zip(df['Label_Num'], df['label']))
#saving the model
import joblib

joblib.dump(xgb, "xgb_crop_model.pkl")
joblib.dump(crop_map, "crop_label_map.pkl")

print("Model and label map saved successfully.")


y_pred = xgb.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
#evaluating the model



# print("clssification report\n")
# print(classification_report(y_test,y_pred))

# #confusion matrics
# import seaborn as sns
# import matplotlib.pyplot as plt

# cm = confusion_matrix(y_test,y_pred)
# plt.figure(figsize=(10,8))
# sns.heatmap(cm,cmap="Blues",cbar=False)
# plt.xlabel("predicted")
# plt.ylabel("actual")
# plt.title("confusion matrix")
# plt.show(block=True)


