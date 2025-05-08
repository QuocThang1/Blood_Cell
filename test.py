import pandas as pd

df = pd.read_csv("FinalProjectTest4/data/class labels.csv")
print(df["Categoriy"].value_counts())
