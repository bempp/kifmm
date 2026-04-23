import pandas as pd
import glob

prefix = "fp64_8e6"
arch = "m1"
files = glob.glob(f"*{prefix}*")

dfs = []
for file in files:
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs, ignore_index=True)
df.to_csv(f'{prefix}_{arch}.csv')


