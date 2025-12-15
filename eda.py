import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Loading_Dataset
crop = pd.read_csv("Crop_recommendation.csv")

# 📊 1. Feature Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(crop.drop('label', axis=1).corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("🌡️ Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 📊 2. Nutrient & Weather Distributions
cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
plt.figure(figsize=(16,12))
for i, col in enumerate(cols):
    plt.subplot(3, 3, i+1)
    sns.histplot(crop[col], kde=True, color='teal')
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.show()

# 📊 3. Crop Count (Label Frequency)
plt.figure(figsize=(12,8))
sns.countplot(data=crop, y='label', order=crop['label'].value_counts().index, palette='Set3')
plt.title("🌾 Crop Distribution in Dataset")
plt.xlabel("Count")
plt.ylabel("Crop")
plt.tight_layout()
plt.show()

# 📊 4. Boxplots of Nutrients by Crop (optional, longer)
# Uncomment below if you want more insight per crop
# for col in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
#     plt.figure(figsize=(14, 6))
#     sns.boxplot(x='label', y=col, data=crop)
#     plt.title(f"{col} levels by Crop")
#     plt.xticks(rotation=90)
#     plt.tight_layout()
#     plt.show()
