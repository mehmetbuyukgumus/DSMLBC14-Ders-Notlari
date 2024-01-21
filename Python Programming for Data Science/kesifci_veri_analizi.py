########################
# Genel Bakış
########################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

data = sns.load_dataset('titanic')
df = data.copy()
df.head()
df.shape
df.columns
df.index
df.info()
df.describe().T
df.isnull().sum()
np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])


def check_df(dataframe, head=5):
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head ##################")
    print(dataframe.head())
    print("################## Tail ##################")
    print(dataframe.tail())
    print("################## Isnull ##################")
    print(dataframe.isnull().sum())
    print("################## Describe ##################")
    print(dataframe.describe().T)


check_df(df)

data2 = sns.load_dataset('tips')
df2 = data2.copy()
check_df(df2)

df["embark_town"].value_counts()
df["sex"].unique()
df["sex"].nunique()
df[["sex", "class"]].nunique()

cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float", ]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 10 and str(df[col].dtypes) in ["object", "category"]]

cat_cols = cat_cols + num_but_cat
df[cat_cols]
df[cat_cols].nunique()


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name],
                      data=dataframe)
        plt.show(block=True)


cat_summary(df, "sex", plot=True)

for col in cat_cols:
    if df[col].dtypes == "Bool":
        print("sadkflsajsk")
    else:
        cat_summary(df, col, plot=True)

df["adult_male"].astype(int)

for col in cat_cols:
    if df[col].dtypes == "Bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)

########################
# Sayısal Değişken Analizi
########################

df[["age", "fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]

num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numeric_cols):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numeric_cols].describe(quantiles).T)


num_summary(df, "age")

for col in num_cols:
    num_summary(df, col)


def num_summary(dataframe, numeric_cols, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numeric_cols].describe(quantiles).T)

    if plot:
        dataframe[numeric_cols].hist()
        plt.xlabel(numeric_cols)
        plt.title(numeric_cols)
        plt.show(block=False)


num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)


########################
# Değişkenlerin Yakalanması
########################

def grap_cal_names(dataframe, cat_th=10, car_th=20):
    """
    
    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th: int, float
        numeric fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numeric değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi
    
    Notes
    -------
    
    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_colsun içinde
        Return olan üç liste toplamı toplam değişken sayısına eşittir
    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float", ]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 10 and str(df[col].dtypes) in ["object", "category"]]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation : {dataframe.shape[0]}")
    print(f"Observation : {dataframe.shape[1]}")
    print(f"Cat Cols : {len(cat_cols)}")
    print(f"Num Cols : {len(num_cols)}")

    return cat_cols, num_cols


cat_cols, num_cols = grap_cal_names(df)

for col in cat_cols:
    grap_cal_names(df, col)

# Bonus

data = sns.load_dataset('titanic')
df = data.copy()
df.info()

for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype('int')

cat_cols, num_cols = grap_cal_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name],
                      data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df, col, plot=True)

########################
# Hedef Değişken Analizi
########################

for col in df.columns:
    if df[col].dtype == 'bool':
        df[col] = df[col].astype('int')


def grap_cal_names(dataframe, cat_th=10, car_th=20):
    """

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframedir
    cat_th: int, float
        numeric fakat kategorik olan değişkenler için sınıf eşik değeri
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numeric değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -------

    cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_colsun içinde
        Return olan üç liste toplamı toplam değişken sayısına eşittir
    """

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["object", "category", "bool"]]
    num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float", ]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 10 and str(df[col].dtypes) in ["object", "category"]]
    cat_cols = cat_cols + num_but_cat
    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation : {dataframe.shape[0]}")
    print(f"Observation : {dataframe.shape[1]}")
    print(f"Cat Cols : {len(cat_cols)}")
    print(f"Num Cols : {len(num_cols)}")

    return cat_cols, num_cols


cat_cols, num_cols = grap_cal_names(df)

########################
# Hedef Değişkenin Kategorik Değişken İle Analizi
########################

df.groupby("sex").agg({"survived": ["mean", "count"]})


def target_summary_with_cat(dataframe, target, categorical_col):
    result_df = dataframe.groupby(categorical_col).agg({target: "mean"})
    print(result_df)


target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived", col)

########################
# Hedef Değişkenin Sayısal Değişken İle Analizi
########################

df.groupby("survived").agg({"age": "mean"})

target_summary_with_cat(df, "survived", "age")

for col in num_cols:
    target_summary_with_cat(df, "survived", col)

########################
# Kolerasyon Analizi
# data link = https://www.kaggle.com/datasets/imtkaggleteam/breast-cancer
########################

data = pd.read_csv("datasets/breast-cancer.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = data.copy()
df = df.iloc[:, 1:-1]
df.head()

num_col = [col for col in df.columns if df[col].dtype in ["int", float]]

corr = df[num_col].corr(method="pearson")

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()
