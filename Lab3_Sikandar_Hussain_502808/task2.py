import pandas as pd
import numpy as np

print("=" * 80)
print("TASK 2: DATA TRANSFORMATION")
print("=" * 80)

# Load the housing dataset
housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(housing_url)

print(f"\n[SUCCESS] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# 1. Apply & Lambda - Categorize housing age
print("\n" + "=" * 80)
print("1. APPLY & LAMBDA: AGE CATEGORIZATION")
print("=" * 80)

df['age_category'] = df['housing_median_age'].apply(
    lambda age: 'New' if age <= 10 else 'Moderate' if age <= 30 else 'Old'
)

print("\nCategorizing housing age into: New (≤10), Moderate (11-30), Old (>30)")
print("-" * 80)
print(df[['housing_median_age', 'age_category']].head(10))

# 2. Apply & Lambda - Categorize income levels
print("\n" + "=" * 80)
print("2. APPLY & LAMBDA: INCOME CATEGORIZATION")
print("=" * 80)

df['income_category'] = df['median_income'].apply(
    lambda income: 'Low' if income < 3 else 'Medium' if income < 6 else 'High'
)

print("\nCategorizing income into: Low (<3), Medium (3-6), High (>6)")
print("-" * 80)
print(df[['median_income', 'income_category']].head(10))

# 3. Map - Transform ocean proximity to simplified categories
print("\n" + "=" * 80)
print("3. MAP: OCEAN PROXIMITY TRANSFORMATION")
print("=" * 80)

ocean_mapping = {
    'NEAR BAY': 'Coastal',
    'NEAR OCEAN': 'Coastal',
    '<1H OCEAN': 'Coastal',
    'INLAND': 'Inland',
    'ISLAND': 'Island'
}
df['location_type'] = df['ocean_proximity'].map(ocean_mapping)

print("\nMapping ocean proximity to simplified location types:")
print("-" * 80)
for original, simplified in ocean_mapping.items():
    print(f"   {original:15} -> {simplified}")
print("-" * 80)
print(df[['ocean_proximity', 'location_type']].head(10))

# 4. Lambda - Clean string column
print("\n" + "=" * 80)
print("4. LAMBDA: STRING CLEANING")
print("=" * 80)

df['ocean_proximity_clean'] = df['ocean_proximity'].apply(
    lambda x: x.strip().lower() if isinstance(x, str) else x
)

print("\nCleaning ocean_proximity column (strip whitespace, convert to lowercase)")
print("-" * 80)
print(df[['ocean_proximity', 'ocean_proximity_clean']].head(10))

# Summary
print("\n" + "=" * 80)
print("TASK 2 SUMMARY")
print("=" * 80)
print(f"\nNew columns created: 4")
print("   1. age_category (using apply & lambda)")
print("   2. income_category (using apply & lambda)")
print("   3. location_type (using map)")
print("   4. ocean_proximity_clean (using lambda)")
print(f"\nFinal dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("=" * 80)