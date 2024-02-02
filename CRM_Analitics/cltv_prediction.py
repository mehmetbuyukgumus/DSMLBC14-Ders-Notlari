##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

# 1. Verinin Hazırlanması (Data Preperation)
# 2. BG-NBD Modeli ile Expected Number of Transaction
# 3. Gamma-Gamma Modeli ile Expected Average Profit
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
# 6. Çalışmanın fonksiyonlaştırılması

##############################################################
# 1. Verinin Hazırlanması (Data Preperation)
##############################################################
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: "%.4f" % x)


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


########################################
# Verinin Okunması
########################################
df_ = pd.read_excel("datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()

########################################
# Veri Önişleme
########################################
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]
df = df[df["Price"] > 0]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

########################################
# Lifetime Veri Yapısının Hazırlanması
########################################

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç

cltv_c = df.groupby("Customer ID").agg({"TotalPrice": lambda totalprice: totalprice.sum(),
                                        "Invoice": lambda invoice: invoice.nunique(),
                                        "InvoiceDate": [
                                            lambda invoice_date: (invoice_date.max() - invoice_date.min()).days,
                                            lambda invoice_date: (today_date - invoice_date.min()).days]})

cltv_c.columns = cltv_c.columns.droplevel(0)
cltv_c.columns = ["montary", "frequency", "recency", "T"]
cltv_c = cltv_c[cltv_c["frequency"] > 1]
cltv_c["recency"] = cltv_c["recency"] / 7
cltv_c["T"] = cltv_c["T"] / 7

##############################################################
# 2. BG-NBD Modeli ile Expected Number of Transaction
##############################################################

bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_c["frequency"],
        cltv_c["recency"],
        cltv_c["T"])

################################################################
# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri kimdir?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_c["frequency"],
                                                        cltv_c["recency"],
                                                        cltv_c["T"]).sort_values(ascending=False).head(10)
bgf.predict(1,
            cltv_c["frequency"],
            cltv_c["recency"],
            cltv_c["T"]).sort_values(ascending=False).head(10)
# Haftalık Tahmin
cltv_c["expected_purc_1_week"] = bgf.predict(1,
                                             cltv_c["frequency"],
                                             cltv_c["recency"],
                                             cltv_c["T"])
# Aylık Tahmin
cltv_c["expected_purc_1_month"] = bgf.predict(4,
                                             cltv_c["frequency"],
                                             cltv_c["recency"],
                                             cltv_c["T"])

bgf.predict(4,
        cltv_c["frequency"],
        cltv_c["recency"],
        cltv_c["T"]).sum()
cltv_c["expected_purc_1_month"].sum()

# 3 Aylık Tahmin
bgf.predict(4 * 3,
            cltv_c["frequency"],
            cltv_c["recency"],
            cltv_c["T"]).sum()

# Tahmin Sonuçlarının Değerlendirilmesi
plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. Gamma-Gamma Modeli ile Expected Average Profit
##############################################################

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_c["frequency"], cltv_c["montary"])
ggf.conditional_expected_average_profit(cltv_c["frequency"], cltv_c["montary"]).head(10)
ggf.conditional_expected_average_profit(cltv_c["frequency"], cltv_c["montary"]).sort_values(ascending=False)
cltv_c["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_c["frequency"], cltv_c["montary"])
cltv_c["expected_average_profit"].sort_values(ascending=False)

##############################################################
# 4. BG-NBD ve Gamma-Gamma Modeli ile CLTV'nin Hesaplanması
##############################################################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_c["frequency"],
                                   cltv_c["recency"],
                                   cltv_c["T"],
                                   cltv_c["montary"],
                                   time=3,
                                   freq="W",
                                   discount_rate=0.01)
cltv.head()
cltv = cltv.reset_index()
cltv_final = cltv_c.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

##############################################################
# 5. CLTV'ye Göre Segmentlerin Oluşturulması
##############################################################

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.groupby("segment").agg({"sum", "count", "mean"})

##############################################################
# 6. Çalışmanın fonksiyonlaştırılması
##############################################################

def create_cltv_p(dataframe, month=3):
    # 1. Veri Ön İşleme
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Modelinin Kurulması
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Modelinin Kurulması
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. BG-NBD ve GG modeli ile CLTV'nin hesaplanması.
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final

df = df_.copy()
cltv_final2 = create_cltv_p(df)
