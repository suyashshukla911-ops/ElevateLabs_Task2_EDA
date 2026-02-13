# Task 2: Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("titanic-Dataset.csv")

# -------------------------------
# 1️⃣ Summary Statistics
# -------------------------------
print("Basic Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# -------------------------------
# 2️⃣ Histograms for Numeric Features
# -------------------------------

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

for col in numeric_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.savefig(f"hist_{col}.png")
    plt.close()

print("Histograms saved ✅")

# -------------------------------
# 3️⃣ Boxplots for Numeric Features
# -------------------------------

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"boxplot_{col}.png")
    plt.close()

print("Boxplots saved ✅")

# -------------------------------
# 4️⃣ Correlation Matrix
# -------------------------------

plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.savefig("correlation_matrix.png")
plt.close()

print("Correlation matrix saved ✅")

# -------------------------------
# 5️⃣ Pairplot (Important)
# -------------------------------

sns.pairplot(df[numeric_cols])
plt.savefig("pairplot.png")
plt.close()

print("Pairplot saved ✅")

print("\nEDA Completed Successfully 🚀")
