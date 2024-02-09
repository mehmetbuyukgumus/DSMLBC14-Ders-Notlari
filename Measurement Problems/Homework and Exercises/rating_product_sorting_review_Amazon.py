###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde
# hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması
# ve satın alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların
# doğru bir şekilde sıralanması olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan
# etkileyeceğinden dolayı hem maddi kayıp hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde
# e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler ise satın alma yolculuğunu sorunsuz olarak
# tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################
import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import math
import scipy.stats as st

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################
data = pd.read_csv('datasets/amazon_review.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = data.copy()
df.head()

# Adım 1:   Ürünün ortalama puanını hesaplayınız.
df.groupby('asin').agg({"overall": "mean"})

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
# Adım 3:  Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.
###################################################
df["reviewTime"] = df["reviewTime"].apply(lambda x: pd.to_datetime(x))
current_day = df["reviewTime"].max()
df["days"] = (current_day - df["reviewTime"]).dt.days


def time_based_weighted_average(dataframe, w1=0.35, w2=0.25, w3=0.20, w4=0.10, w5=0.06, w6=0.04, plot=False):
    if plot:
        comments_table = pd.DataFrame({"<200": [df.loc[df["days"] < 200, "overall"].mean() * w1],
                                       ">=200": [df.loc[df["days"] >= 200, "overall"].mean() * w2],
                                       ">=400": [df.loc[df["days"] >= 400, "overall"].mean() * w3],
                                       ">=600": [df.loc[df["days"] >= 600, "overall"].mean() * w4],
                                       ">=800": [df.loc[df["days"] >= 800, "overall"].mean() * w5],
                                       ">=1000": [df.loc[df["days"] >= 1000, "overall"].mean() * w6]})
        sns.barplot(data=comments_table)
        plt.show()
    return (df.loc[df["days"] < 200, "overall"].mean() * w1) + \
        (df.loc[df["days"] >= 200, "overall"].mean() * w2) + \
        (df.loc[df["days"] >= 400, "overall"].mean() * w3) + \
        (df.loc[df["days"] >= 600, "overall"].mean() * w4) + \
        (df.loc[df["days"] >= 800, "overall"].mean() * w5) + \
        (df.loc[df["days"] >= 1000, "overall"].mean() * w6)


time_based_weighted_average(df)

###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.
df["help_no"] = df["total_vote"] - df["helpful_yes"]


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################
def score_pos_neg_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["score_pos_neg_diff"] = score_pos_neg_diff(df["helpful_yes"], df["help_no"])
df["score_average_rating"] = df[["helpful_yes", "help_no"]].apply(
    lambda x: score_average_rating(x["helpful_yes"], x["help_no"]), axis=1)
df["wilson_lower_bound"] = df[["helpful_yes", "help_no"]].apply(
    lambda x: wilson_lower_bound(x["helpful_yes"], x["help_no"]), axis=1)
##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values(by="wilson_lower_bound", ascending=False).head(20)
