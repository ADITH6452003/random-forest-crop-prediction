import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/home/adith6452/Documents/cropprediction/data/crop_production.csv')
df = df[df['State_Name'].isin(['Tamil Nadu', 'Kerala'])]
df.to_csv('/home/adith6452/Documents/cropprediction/data/crop_production.csv', index=False)
print(f"Dataset modified. New row count: {len(df)}")

def count_outliers(column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

# Check outliers for each numeric column
for col in ['Area', 'Production', 'Crop_Year']:
    count, lower, upper = count_outliers(col)
    print(f"{col}: {count} outliers (range: {lower:.2f} to {upper:.2f})")

# Visualize with outlier count
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[['Area', 'Production']])
plt.title(f"Area: {count_outliers('Area')[0]} outliers | Production: {count_outliers('Production')[0]} outliers")
plt.show()

# Show actual outlier values
Q1 = df['Area'].quantile(0.25)
Q3 = df['Area'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['Area'] < Q1 - 1.5 * IQR) | (df['Area'] > Q3 + 1.5 * IQR)]
print(f"\nOutlier rows:\n{outliers[['State_Name', 'District_Name', 'Area', 'Production']]}")

for col in ['Area', 'Production', 'Crop_Year']:
    mean = df[col].mean()
    median = df[col].median()
    print(f"{col} - Mean: {mean:.2f}, Median: {median:.2f}, Outliers: {'Yes' if median > mean else 'No'}")
