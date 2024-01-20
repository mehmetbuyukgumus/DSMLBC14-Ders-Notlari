########################
# PANDAS
########################

########################
# Pandas Series
########################

import pandas as pd
import numpy as np
import seaborn as sns

d = pd.Series([10, 2, 9, 4, 2])
type(d)
d.index
d.dtype
d.size
d.ndim
d.values
type(d.values)
d.head(3)
d.tail(2)

########################
# Veri Okuma
########################

df = pd.read_csv('datasets/Advertising.csv')
df.head()

########################
# Veriye Hızlı Bakış
########################
df = sns.load_dataset('titanic')
df.head()
df.tail()
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
df["sex"].value_counts()

########################
# Seçim İşlemleri
########################
df.index
df[0:3]
df.head()
df.drop(0, axis=0)
deleted_indexes = [1, 2, 5, 7]
df.drop(deleted_indexes, axis=0)

########################
# Değişkeni Index'e çevirmek
########################
df.index = df["age"]
df.drop("age", axis=1)
df.drop("age", axis=1, inplace=True)

########################
# Indexi Değişkene Çevirmek
########################
df.head()
df.drop("age", axis=1, inplace=True)
df.reset_index()

########################
# Değişkenler Üzerinde İşlemler
########################
df = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
df.head()
"age" in df
df[["age", "sex", "alive"]].head()
col_names = ["age", "sex"]
df[col_names].head()
df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]
df.drop("age2", axis=1, inplace=True)
col_deleted_names = ["who", "age3"]
df.drop(col_deleted_names, axis=1, inplace=True)
df.loc[:, ~df.columns.str.contains("age")].head()

########################
# Loc & Iloc
########################

# iloc
df.iloc[0:3]
df.iloc[0, 0]

# loc
df.loc[0:3]

df.iloc[0:5, 2:3]
df.loc[0:5, "sex"]

########################
# Koşullu İşlemler
########################

df[df["age"] > 50].head()
df[df["age"] > 50].count()
df.loc[df["age"] > 50, ["class", "sex"]].head()

df.loc[((df["class"] == "First")
        & (df["sex"] == "male")
        & ((df["embark_town"] == "Southampton") | (df["embark_town"] == "Cherbourg"))), ["class", "sex"]].head()

########################
# Toplulaştırma, Gruplama
########################
df = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
df.head()
df.groupby("sex")["age"].mean()
df.groupby(["sex", "embark_town"]).agg({"age": ["mean", "sum"],
                                        "survived": "mean"})
df.groupby(["sex", "embark_town", "class"]).agg({"age": ["mean"],
                                                 "survived": "mean",
                                                 "sex": "count"})
########################
# Pivot Table
########################
df = sns.load_dataset("titanic")
df.head()
df.pivot_table(index="embark_town", columns="sex", values="survived", aggfunc="count")
df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])
df.pivot_table(index=["sex", "embark_town"], columns="new_age", values="survived", aggfunc="mean")

########################
# Apply ve Lambda
########################
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 700)
df.head()
df["new_age"] = df["age"] * 2
df["new_age2"] = df["age"] / 3
df[["age", "new_age", "new_age2"]].apply(lambda x: x /10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: x /10).head()
df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / (x.std())).head()

########################
# Birleştirme İşlemleri
########################

# concat
m = np.random.randint(1,30, (5,3))
df1 = pd.DataFrame(m, columns=["var1", "var2", "var3"])
df2 = df1 + 99
pd.concat([df1, df2], ignore_index=True, axis = 0)

# merge
df1 = pd.DataFrame({"employees": ["John", "Dennis", "Mark", "Maria"],
                    "group": ["Accounting", "Engineering", "Engineering", "HR"]})
df2 = pd.DataFrame({"employees": ["John", "Dennis", "Mark", "Maria"],
                    "start_date": [2010, 2009, 2014, 2011]})

df3 = pd.merge(df1, df2, on="employees")
df4 = pd.DataFrame({"group": ["Accounting", "Engineering", "HR"],
                    "manager": ["Caner", "Mustafa", "Berkcan"]})

df5 = pd.merge(df3, df4, on="group")
