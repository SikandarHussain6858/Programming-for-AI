import pandas as pd
import numpy as np
import os

print("=" * 80)
print("CALIFORNIA HOUSING DATASET ANALYSIS")
print("=" * 80)

# Load the housing dataset
housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(housing_url)

print("\n[SUCCESS] Dataset loaded successfully!")
print(f"\nFirst look at the data:")
print("-" * 80)
print(df.head())

# Data exploration
print("\n" + "=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)

print(f"\nDataset dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"That's {df.shape[0]:,} housing records with {df.shape[1]} features each.")

print(f"\nDataset structure and information:")
print("-" * 80)
print(df.info())

# Check for missing values
print("\n" + "=" * 80)
print("MISSING VALUES CHECK")
print("=" * 80)

missing_values = df.isnull().sum()
total_missing = missing_values.sum()

print(f"\nLooking for missing values in the dataset...")
print("-" * 80)
print(missing_values)

if total_missing > 0:
    print(f"\n[WARNING] Found {total_missing} missing values total.")
    print("Don't worry, we'll handle these next!")
else:
    print(f"\n[OK] Great! No missing values found.")

# Handling missing values in numerical columns
print("\n" + "=" * 80)
print("HANDLING MISSING VALUES")
print("=" * 80)

num_cols = df.select_dtypes(include=['int64','float64']).columns
print(f"\nIdentified {len(num_cols)} numerical columns:")
print("-" * 80)
for i, col in enumerate(num_cols, 1):
    print(f"   {i}. {col}")

# Calculate median for missing values
missing_val = df[num_cols].median()
print(f"\nMedian values for each numerical column:")
print("-" * 80)
print(missing_val)

# Fill missing values with median
print(f"\nFilling missing values with median values...")
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Verify missing values are handled
remaining_missing = df.isnull().sum().sum()
if remaining_missing == 0:
    print(f"[SUCCESS] All missing values have been handled.")
else:
    print(f"[WARNING] {remaining_missing} missing values still remain.")

# Identify categorical columns
print("\n" + "=" * 80)
print("CATEGORICAL DATA IDENTIFICATION")
print("=" * 80)

cat_cols = df.select_dtypes(include=['object']).columns
print(f"\nFound {len(cat_cols)} categorical column(s):")
print("-" * 80)
for i, col in enumerate(cat_cols, 1):
    unique_count = df[col].nunique()
    print(f"   {i}. {col} ({unique_count} unique values)")

# Normalize numerical columns using Min-Max scaling
print("\n" + "=" * 80)
print("DATA NORMALIZATION")
print("=" * 80)

print(f"\nNormalizing all numerical columns using Min-Max scaling...")
print("(Scaling values to range 0-1)")

num_cols = df.select_dtypes(include=np.number).columns
df[num_cols] = (df[num_cols] - df[num_cols].min()) / (df[num_cols].max() - df[num_cols].min())

print(f"\n[SUCCESS] Normalization complete! Here's a preview of normalized data:")
print("-" * 80)
print(df[num_cols].head())

print("\n" + "=" * 80)
print("DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns, all clean and normalized!")
print("=" * 80)

# Save cleaned dataset
print("\n" + "=" * 80)
print("SAVING CLEANED DATASET")
print("=" * 80)

# Create data directory if it doesn't exist
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"\n[INFO] Created directory: {data_dir}/")

# Save to CSV
output_path = os.path.join(data_dir, "cleaned_dataset.csv")
df.to_csv(output_path, index=False)
print(f"\n[SUCCESS] Cleaned dataset saved to: {output_path}")
print(f"File size: {os.path.getsize(output_path):,} bytes")

print("\n" + "=" * 80)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 80)
