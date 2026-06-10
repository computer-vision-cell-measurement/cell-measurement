from pathlib import Path
import pandas as pd

df_y = pd.read_excel("data/results/wymiary_komorek_excel.xlsx")
stems = sorted({Path(str(x)).stem for x in df_y["Nazwa_Obrazu"].unique()})
for s in stems:
    print(s[:20])