##############################################################
# BG-NBD ve Gamma-Gamma ile CLTV Prediction
##############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO satış ve pazarlama faaliyetleri için roadmap belirlemek istemektedir.
# Şirketin orta uzun vadeli plan yapabilmesi için var olan müşterilerin gelecekte şirkete sağlayacakları potansiyel
# değerin tahmin edilmesi gerekmektedir.


###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak yapan
# müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel : Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile, Offline)
# last_order_channel : En son alışverişin yapıldığı kanal
# first_order_date : Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date : Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online : Muşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline : Muşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online : Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline : Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline : Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online : Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 : Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi


###############################################################
# GÖREVLER
###############################################################
# GÖREV 1: Veriyi Hazırlama
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

# 1. flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
data = pd.read_csv('datasets/flo_data_20k.csv')
df = data.copy()


# 2. Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını
# tanımlayınız.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round()
# ile yuvarlayınız.
# 3. "order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.
replace_with_thresholds(df, "order_num_total_ever_online")
replace_with_thresholds(df, "order_num_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_offline")
replace_with_thresholds(df, "customer_value_total_ever_online")

# Elif'in Çözümü:

for column in df.columns:
    if 'ever' in column:
        replace_with_thresholds(df, column)

# Ali'nin Çözümü:

columns = ["order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
           "customer_value_total_ever_online"]

for col in columns:
    replace_with_thresholds(df, col)

# Seviç'in Çözümü:

[replace_with_thresholds(df, col) for col in df.columns if df[col].dtypes in ["float64"]]

# 4. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.
df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_bill"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 5. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x))

# GÖREV 2: CLTV Veri Yapısının Oluşturulması
# 1.Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.
analysis_date = df["last_order_date"].max() + pd.Timedelta(days=2)
# https://stackoverflow.com/questions/16385785/add-days-to-dates-in-dataframe

# 2.customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir
# cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten
# ifade edilecek.
cltv = pd.DataFrame({"customer_id": df["master_id"],
                     "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]) / 7,
                     "T_weekly": (df["first_order_date"].apply(lambda x: analysis_date - x) / 7),
                     "frequency": df["total_order"],
                     "monetary_cltv_avg": df["total_bill"] / df["total_order"]})
cltv["recency_cltv_weekly"] = cltv["recency_cltv_weekly"].dt.days
cltv["T_weekly"] = cltv["T_weekly"].dt.days

# GÖREV 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması, CLTV'nin hesaplanması
# 1. BG/NBD modelini fit ediniz.
bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv["frequency"],
        cltv["recency_cltv_weekly"],
        cltv["T_weekly"])

# a. 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv dataframe'ine
# ekleyiniz.
cltv["exp_sales_3_month"] = bgf.predict(4 * 3,
                                        cltv["monetary_cltv_avg"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"])

# b. 6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv dataframe'ine
# ekleyiniz.
cltv["exp_sales_6_month"] = bgf.predict(4 * 6,
                                        cltv["monetary_cltv_avg"],
                                        cltv["recency_cltv_weekly"],
                                        cltv["T_weekly"])

# 2. Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak
# cltv dataframe'ine ekleyiniz.
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv["frequency"], cltv["monetary_cltv_avg"])
cltv["exp_average_value"] = ggf.conditional_expected_average_profit(cltv["frequency"], cltv["monetary_cltv_avg"])

# 3. 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.

# Ali'nin çöüzmü:
cltv["cltv"] = ggf.customer_lifetime_value(bgf,
                                           cltv["frequency"],
                                           cltv["recency_cltv_weekly"],
                                           cltv["T_weekly"],
                                           cltv["monetary_cltv_avg"],
                                           time=6,
                                           freq="W")

# b. Cltv değeri en yüksek 20 kişiyi gözlemleyiniz.
cltv.sort_values(by="cltv", ascending=False).head(20)

# GÖREV 4: CLTV'ye Göre Segmentlerin Oluşturulması
# 1. 6 aylık tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz. cltv_segment ismi
# ile dataframe'e ekleyiniz.
cltv["cltv_segment"] = pd.qcut(cltv["cltv"], 4, labels=["D", "C", "B", "A"])
# 2. 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz
budget = cltv.groupby("cltv_segment").agg({"cltv": "sum",
                                           "exp_sales_6_month": "sum"})
## Yönetime öneri:
budget.reset_index(inplace=True)
budget["percentage"] = budget["cltv"].apply(lambda x: x / budget["cltv"].sum())

# Şirketin "percentage" değişkeninde hesaplanan yüzdelikler şeklinde pazarlama bütçesini ilgili segmentlere dağıtması
# gerekir.
