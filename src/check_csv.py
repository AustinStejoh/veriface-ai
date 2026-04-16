import pandas as pd

df = pd.read_csv("data/train.csv")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 3 rows:")
print(df.head(3))