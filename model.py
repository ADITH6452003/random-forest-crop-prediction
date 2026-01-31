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

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(x_train, y_train)
#to convert thee output in numbers to the feature values
crop_map = dict(zip(df['Label_Num'], df['label']))
#saving the model
import joblib

joblib.dump(rf, "rf_crop_model.pkl")
joblib.dump(crop_map, "crop_label_map.pkl")

print("Model and label map saved successfully.")


y_pred = rf.predict(x_test)
#evaluating the model

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

accuracy = accuracy_score(y_test,y_pred)
print("accuracy = ",accuracy)

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


