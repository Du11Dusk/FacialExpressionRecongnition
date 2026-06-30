import pandas as pd

data = pd.read_csv("data/fer2013.csv")

print(data.head())
print(data["emotion"].value_counts())