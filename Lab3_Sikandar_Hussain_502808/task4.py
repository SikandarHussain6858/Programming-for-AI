import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 80)
print("TASK 4: DATA VISUALIZATION WITH PANDAS PLOTTING")
print("=" * 80)

# Load the housing dataset
housing_url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(housing_url)

print(f"\n[SUCCESS] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Create plots directory if it doesn't exist
plots_dir = "plots"
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
    print(f"\n[INFO] Created directory: {plots_dir}/")

# Set plot style for better appearance
plt.style.use('default')

# TASK 4.1: BAR CHART - Categorical Distribution
print("\n" + "=" * 80)
print("1. BAR CHART: CATEGORICAL DISTRIBUTION")
print("=" * 80)

print("\nPlotting distribution of ocean_proximity (categorical column)...")

# Count values and plot
ocean_counts = df['ocean_proximity'].value_counts()
print("\nOcean Proximity Distribution:")
print("-" * 80)
print(ocean_counts)

# Create bar chart
plt.figure(figsize=(10, 6))
ocean_counts.plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Distribution of Ocean Proximity', fontsize=16, fontweight='bold')
plt.xlabel('Ocean Proximity Category', fontsize=12)
plt.ylabel('Number of Properties', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
bar_chart_path = os.path.join(plots_dir, 'ocean_proximity_bar_chart.png')
plt.savefig(bar_chart_path, dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Bar chart saved to: {bar_chart_path}")
plt.close()

# TASK 4.2: HISTOGRAM - Numerical Distribution
print("\n" + "=" * 80)
print("2. HISTOGRAM: NUMERICAL DISTRIBUTION")
print("=" * 80)

print("\nPlotting distribution of median_income (numerical column)...")
print("\nMedian Income Statistics:")
print("-" * 80)
print(df['median_income'].describe())

# Create histogram
plt.figure(figsize=(10, 6))
df['median_income'].plot(kind='hist', bins=50, color='coral', edgecolor='black', alpha=0.7)
plt.title('Distribution of Median Income', fontsize=16, fontweight='bold')
plt.xlabel('Median Income (in tens of thousands)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
histogram_path = os.path.join(plots_dir, 'median_income_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Histogram saved to: {histogram_path}")
plt.close()

# TASK 4.3: SCATTER PLOT - Relationship Between Two Numerical Columns
print("\n" + "=" * 80)
print("3. SCATTER PLOT: RELATIONSHIP EXPLORATION")
print("=" * 80)

print("\nExploring relationship between median_income and median_house_value...")

# Calculate correlation
correlation = df['median_income'].corr(df['median_house_value'])
print(f"\nCorrelation coefficient: {correlation:.4f}")
print("-" * 80)

# Create scatter plot
plt.figure(figsize=(12, 7))
plt.scatter(df['median_income'], df['median_house_value'], 
            alpha=0.3, s=20, c='darkgreen', edgecolors='none')
plt.title('Relationship: Median Income vs Median House Value', 
          fontsize=16, fontweight='bold')
plt.xlabel('Median Income (in tens of thousands)', fontsize=12)
plt.ylabel('Median House Value ($)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Add correlation annotation
plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', 
         transform=plt.gca().transAxes, 
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()

# Save the plot
scatter_path = os.path.join(plots_dir, 'income_vs_value_scatter.png')
plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Scatter plot saved to: {scatter_path}")
plt.close()

# BONUS: Additional visualization - Housing Age Distribution
print("\n" + "=" * 80)
print("BONUS: HOUSING AGE HISTOGRAM")
print("=" * 80)

print("\nPlotting distribution of housing_median_age...")

plt.figure(figsize=(10, 6))
df['housing_median_age'].plot(kind='hist', bins=30, color='mediumpurple', 
                               edgecolor='black', alpha=0.7)
plt.title('Distribution of Housing Median Age', fontsize=16, fontweight='bold')
plt.xlabel('Housing Median Age (years)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Save the plot
age_histogram_path = os.path.join(plots_dir, 'housing_age_histogram.png')
plt.savefig(age_histogram_path, dpi=300, bbox_inches='tight')
print(f"\n[SUCCESS] Housing age histogram saved to: {age_histogram_path}")
plt.close()

# SUMMARY
print("\n" + "=" * 80)
print("TASK 4 SUMMARY")
print("=" * 80)

print("\nVisualizations Created:")
print("\n1. BAR CHART:")
print(f"   - Ocean proximity distribution")
print(f"   - Most common: {ocean_counts.index[0]} ({ocean_counts.values[0]:,} properties)")
print(f"   - File: {bar_chart_path}")

print("\n2. HISTOGRAM:")
print(f"   - Median income distribution")
print(f"   - Mean: ${df['median_income'].mean():.2f} (tens of thousands)")
print(f"   - File: {histogram_path}")

print("\n3. SCATTER PLOT:")
print(f"   - Median income vs median house value")
print(f"   - Correlation: {correlation:.4f} (positive correlation)")
print(f"   - File: {scatter_path}")

print("\n4. BONUS HISTOGRAM:")
print(f"   - Housing median age distribution")
print(f"   - File: {age_histogram_path}")

print(f"\n[SUCCESS] All plots saved in '{plots_dir}/' directory")
print(f"Total plots created: 4")
print("=" * 80)
