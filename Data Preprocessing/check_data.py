import zipfile
import pandas as pd
ZIP_PATH = r"Data Preprocessing\Data Set.zip"
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    csv_filename = [f for f in z.namelist() if f.endswith('.csv')][0]
    with z.open(csv_filename) as f:
        df = pd.read_csv(f, nrows=2)
print("--- DATASET COLUMNS ---")
print(df.columns.tolist())
print("\n--- FIRST ROW ---")
print(df.iloc[0])
