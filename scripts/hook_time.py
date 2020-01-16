import pandas as pd
def set_pd_opts():
    pd.set_option("display.max_rows", 200)
    pd.options.display.max_colwidth = 100
set_pd_opts()
df = pd.read_csv("hook-time.csv",)
df.loc[:]["time"] *= 1000
df = df.groupby(["id", "name", "type", "method"]).agg({"time": ["mean", "min", "max", "count"]})
df = df.sort_values(("time", "mean"))
df = df.tail(100)
print(df)
df.to_csv("hook-time-sorted.csv")
