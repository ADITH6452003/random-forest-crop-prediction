import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/home/adith6452/Documents/cropprediction/data/crop_production.csv')
#total no of crops in the dataset
unique_val = df["Crop"].unique()
print(unique_val)
unique_count = df["Crop"].nunique()
print(unique_count)
#what are the crops in the datset
print(df["Crop"].value_counts())
#is there any  null values in the dataset
print(df.isnull().sum())
print((df.isnull().sum() / len(df)) * 100)
print(df.info())
#no of districts in each state
unique_dist = df["District_Name"].nunique()
print(unique_dist)
district_count = df.groupby("State_Name")["District_Name"].nunique()
print(district_count)
#toatal number of districts in tamilnadu
tn_districts = df[df["State_Name"] == "Tamil Nadu"]["District_Name"].unique()
print(tn_districts)
#what are the number of crops in the datast
tn = df[df["State_Name"] == "Tamil Nadu"]
crop_counts = tn.groupby("District_Name")["Crop"].nunique()
print(crop_counts)

#visualization(feture classification)
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm")
plt.show()



