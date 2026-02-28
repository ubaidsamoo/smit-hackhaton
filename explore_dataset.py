import pandas as pd
import sys

# Load and examine the dataset
print("=" * 80)
print("EXPLORING ONTIME_REPORTING.csv")
print("=" * 80)

df = pd.read_csv('ONTIME_REPORTING.csv', nrows=20)

print("\nFirst 20 rows:")
print(df.head(20))

print("\n\nDataset Info:")
df_full = pd.read_csv('ONTIME_REPORTING.csv')
print(f"Shape: {df_full.shape}")
print(f"Columns: {df_full.columns.tolist()}")

print("\n\nData Types:")
print(df_full.dtypes)

print("\n\nMissing Values:")
print(df_full.isnull().sum())

print("\n\nDataset Summary Statistics:")
print(df_full.describe())
