################ Gözetimsiz Öğrenme ile Müşteri Segmentasyonu ################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre
# gruplar oluşturulacak.

import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


def load_dataset():
    dataframe = pd.read_csv('datasets/flo_data_20k.csv')
    return dataframe


def analysis_date(dataframe):
    date_cols = [col for col in dataframe.columns if "date" in col]
    for col in date_cols:
        dataframe[col] = pd.to_datetime(dataframe[col])

    analysis_date = dataframe["last_order_date"].max()
    analysis_date = analysis_date + pd.to_timedelta(5, "D")
    return analysis_date




df = load_dataset()
analysis_date = analysis_date(df)
