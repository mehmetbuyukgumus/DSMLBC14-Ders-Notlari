###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################

###############################################################
# İş Problemi (Business Problem)
###############################################################
# FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranış öbeklenmelerine göre gruplar oluşturulacak..

###############################################################
# Veri Seti Hikayesi
###############################################################

# Veri seti son alışverişlerini 2020 - 2021 yıllarında OmniChannel(hem online hem offline alışveriş yapan) olarak
# yapan müşterilerin geçmiş alışveriş davranışlarından
# elde edilen bilgilerden oluşmaktadır.

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

# GÖREV 1: Veriyi Anlama (Data Understanding) ve Hazırlama
import pandas as pd
import datetime as dt

# 1. flo_data_20K.csv verisini okuyunuz.
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
data = pd.read_csv('datasets/flo_data_20k.csv')
df = data.copy()

# 2. Veri setinde
# a. İlk 10 gözlem,
df.head(10)

# b. Değişken isimleri,
df.columns

# c. Betimsel istatistik,
df.describe().T

# d. Boş değer,
df.isnull().sum()

# e. Değişken tipleri, incelemesi yapınız.
df.info()

# 3. Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir.
# Herbir müşterinin toplam alışveriş sayısı ve harcaması için yeni değişkenler oluşturun.

df["total_order"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["total_bill"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

# 4. Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.
date_columns = [col for col in df.columns if "date" in col]
df[date_columns] = df[date_columns].apply(lambda x: pd.to_datetime(x))
# For more information for How to change the datetime format in Pandas:
# https://stackoverflow.com/questions/38067704/how-to-change-the-datetime-format-in-pandas

# 5. Alışveriş kanallarındaki müşteri sayısının, ortalama alınan ürün sayısının ve ortalama harcamaların dağılımına
# bakınız.
df.groupby("order_channel").agg({"master_id": "count",
                                 "total_order": "mean",
                                 "total_bill": "mean"})

# 6. En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
df.sort_values(by="total_bill", ascending=False).head(10)

# 7. En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
df.sort_values(by="total_order", ascending=False).head(10)


# 8. Veri ön hazırlık sürecini fonksiyonlaştırınız.
def preprocessing(dataframe):
    # Yeni değişkenleri oluşturulması
    dataframe["total_order"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["total_bill"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]
    # Değişken tiplerinin değiştirilmesi
    date_column = [col for col in dataframe.columns if "date" in col]
    dataframe[date_column] = dataframe[date_column].apply(lambda x: pd.to_datetime(x))
    return dataframe


preprocessing(df)
# GÖREV 2: RFM Metriklerinin Hesaplanması
df["recency"] = df["last_order_date"] - df["first_order_date"]
rfm = df.groupby("master_id").agg({"recency": lambda x: x,
                                   "total_bill": lambda x: x,
                                   "total_order": lambda x: x})
# GÖREV 3: RF ve RFM Skorlarının Hesaplanması
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
rfm["monetary_score"] = pd.qcut(rfm["total_bill"], 5, labels=[1, 2, 3, 4, 5])
rfm["frequency_score"] = pd.qcut(rfm["total_order"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm["RFM_Scores"] = rfm["frequency_score"].astype(str) + rfm["recency_score"].astype(str)
rfm.columns = ['recency', 'monetary', 'frequency', 'recency_score', 'monetary_score', 'frequency_score', 'RFM_Scores']
rfm.reset_index(inplace=True)
# GÖREV 4: RF Skorlarının Segment Olarak Tanımlanması
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}
rfm["segments"] = rfm["RFM_Scores"].replace(seg_map, regex=True)
# GÖREV 5: Aksiyon zamanı!
# 1. Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.
rfm.groupby('segments').agg({"recency": "mean",
                             "monetary": "mean",
                             "frequency": "mean"})
# 2. RFM analizi yardımı ile 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv ye kaydediniz.
# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
# tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle
# özel olarak iletişime geçeilmek isteniliyor. Sadık müşterilerinden(champions,loyal_customers), ortalama 250 TL üzeri
# ve kadın kategorisinden alışveriş yapan kişiler özel olarak iletişim kuralacak müşteriler. Bu müşterilerin id
# numaralarını csv dosyasına yeni_marka_hedef_müşteri_id.cvs olarak kaydediniz.
list_int_in_women_shoes = df[df["interested_in_categories_12"].str.contains("KADIN")]["master_id"]
rfm_w_s = rfm[rfm["master_id"].isin(list_int_in_women_shoes)]
rfm_w_s = rfm_w_s[(rfm_w_s["segments"] == "loyal_customers") | (rfm_w_s["segments"] == "champions")]
rfm_w_s = rfm_w_s[rfm_w_s["monetary"] > 250]
rfm_w_s["master_id"].to_csv("datasets/yeni_marka_hedef_müşteri_id.csv", index=False)

# b. Erkek ve Çoçuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen
# geçmişte iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve
# yeni gelen müşteriler özel olarak hedef alınmak isteniliyor. Uygun profildeki müşterilerin id'lerini csv dosyasına
# indirim_hedef_müşteri_ids.csv
# olarak kaydediniz.
list_int_in_man_and_child_shoes = df[df["interested_in_categories_12"].str.contains("ERKEK", "ÇOCUK")]["master_id"]
segmentation = ["cant_loose", "about_to_sleep", "new_customers"]
rfm_m_c_s = rfm[rfm["master_id"].isin(list_int_in_man_and_child_shoes)]
rfm_m_c_s[rfm_m_c_s["segments"].isin(segmentation)]["master_id"].to_csv("datasets/indirim_hedef_müşteri_ids.csv",
                                                                        index=False)


# GÖREV 6: Tüm süreci fonksiyonlaştırınız.
def rfm_analysis(dataframe, seg_map):
    # Veri Ön İşleme
    preprocessing(dataframe)
    # RFM Metriklerinin Hesaplanması
    dataframe["recency"] = dataframe["last_order_date"] - dataframe["first_order_date"]
    rfm = df.groupby("master_id").agg({"recency": lambda x: x,
                                       "total_bill": lambda x: x,
                                       "total_order": lambda x: x})
    # RFM Skorlarının Hesaplanması
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm["total_bill"], 5, labels=[1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm["total_order"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["RFM_Scores"] = rfm["frequency_score"].astype(str) + rfm["recency_score"].astype(str)
    rfm.columns = ['recency', 'monetary', 'frequency', 'recency_score', 'monetary_score', 'frequency_score',
                   'RFM_Scores']
    rfm.reset_index(inplace=True)
    # Segmentasyon
    rfm["segments"] = rfm["RFM_Scores"].replace(seg_map, regex=True)
    # Result 1
    list_int_in_man_and_child_shoes = \
        dataframe[dataframe["interested_in_categories_12"].str.contains("ERKEK", "ÇOCUK")]["master_id"]
    segmentation = ["cant_loose", "about_to_sleep", "new_customers"]
    rfm_m_c_s = rfm[rfm["master_id"].isin(list_int_in_man_and_child_shoes)]
    rfm_m_c_s[rfm_m_c_s["segments"].isin(segmentation)]["master_id"].to_csv("datasets/indirim_hedef_müşteri_ids.csv",
                                                                            index=False)


rfm_analysis(df,seg_map)
