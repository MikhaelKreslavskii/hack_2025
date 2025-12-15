import lasio
import pandas as pd

# Load LAS file
las = lasio.read('WELL_009.las')  # Replace with your file path

# Convert to DataFrame
df = las.df()

print(df.head())
print(df.columns)
df = df.reset_index()
df.columns = ['DEPTH', 'VALUE']  # Rename for clarity

print(df.head())
