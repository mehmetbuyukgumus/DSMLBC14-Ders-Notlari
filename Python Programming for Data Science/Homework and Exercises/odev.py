#######################################################
# PYTHON ALIŞTIRMALARI
#######################################################

# Görev 1: Verilen degerlerin veri yapilarini inceleyiniz
x = 8
y = 3.2
z = 8j + 18
a = "Hello World"
b = True
C = 23 < 22
l = [11, 2, 3, 4]
d = {"Name": "Jake",
     "Age": 27,
     "Adress": "Downtown"}
t = ("Machine Learning", "Data Science")
s = {"Python", "Machine Learning", "Data Science"}

# Görev 2: Verilen string ifadenin tüm harflerini büyük harfe çeviriniz. Virgül ve nokta yerine space koyunuz,
# kelime kelime ayiriniz.

text = "The goal is to turn data into information, and information into insight."

# Çözüm 2:

text_split = text.split()
new_text = []
for word in text_split:
    for letter in word:
        if letter == ",":
            letter.replace(",", " ")
        elif letter == ".":
            letter.replace(".", " ")
    new_text.append(word.upper())
print(new_text)

# Görev 3:  Verilen liste yeaşağıdaki adımları uygulayınız.

lst = ["D", "A", "T", "A", "S", "C", "I", "E", "N", "C", "E"]

## Adım1: Verilen listenin eleman sayısına bakınız.
print(len(lst))
## Adım2: Sıfırıncı ve onuncu indeksteki elemanları çağırınız.
print(f" Sıfırıncı Index: {lst[0]} \n Onuncu Index: {lst[10]}")
## Adım3: Verilen liste üzerinden["D", "A", "T", "A"] listesi oluşturunuz.
data = lst[0:4]
## Adım4: Sekizinci indeksteki elemanı siliniz.
lst.pop(8)
## Adım5: Yeni bir eleman ekleyiniz.
lst.append("Z")
## Adım6: Sekizinci indekse"N" elemanını tekrar ekleyiniz.
lst.insert(8, "N")

# Görev 4:  Verilen sözlük yapısına aşağıdaki adımları uygulayınız.

dct = {'Christian': ["America", 18],
       'Daisy': ["England", 12],
       'Antonio': ["Spain", 22],
       'Dante': ["Italy", 25]}
## Adım1: Key değerlerine erişiniz.
dct.keys()
## Adım2: Value'laraerişiniz.
dct.values()
## Adım3: Daisy key'ine ait 12 değerini 13 olarak güncelleyiniz.
dct["Daisy"][1] = 13
## Adım4: Key değeri Ahmet value değeri [Turkey,24] olan yeni bir değer ekleyiniz.
dct["Ahmet"] = ["Turkey", 24]
## Adım5: Antonio'yu dictionary'den siliniz.
del dct["Antonio"]

# Görev 5:Argüman olarak bir liste alan, listenin içerisindeki tek ve çift sayıları ayrı listelere atayan ve bu
# listeleri return eden fonksiyon yazınız.

sayilar = [12, 13, 18, 93, 221]


def tek_citt_sayi(sayilar):
    tek = []
    cift = []
    for number in sayilar:
        if number % 2 == 0:
            cift.append(number)
        else:
            tek.append(number)
    return tek, cift


tek_citt_sayi(sayilar)

# Görev 6:Aşağıda verilen listede mühendislik ve tıp fakülterinde dereceye giren öğrencilerin isimleri bulunmaktadır.
# Sırasıyla ilk üç öğrenci mühendislik fakültesinin başarı sırasını temsil ederken son üç öğrenci de tıp fakültesi
# öğrenci sırasına aittir. Enumarate kullanarak öğrenci derecelerini fakülte özelinde yazdırınız.

ogrenciler = ["Ali", "Veli", "Ayse", "Talat", "Zeynep", "Ece"]

muh_list = ogrenciler[0:3]
tip_list = ogrenciler[3:]

for index, value in enumerate(ogrenciler):
    if value in muh_list:
        print(f"Mühendislik Fakültesi {index}. öğrenci: {value}")
    else:
        print(f"Tıp Fakültesi {index}. öğrenci: {value}")

# Görev 7:Aşağıda 3 adet liste verilmiştir. Listelerde sırası ile bir dersin kodu, kredisi ve kontenjan bilgileri
# yer almaktadır. Zip kullanarak ders bilgilerini bastırınız.

ders_kodu = ["CMP1005", "PSY1001", "HUK1005", "'SEN2204"]
kredi = [3, 4, 2, 4]
kontenjan = [30, 75, 150, 25]

for ders, puan, kont in zip(ders_kodu, kredi, kontenjan):
    print(f"Kredisi {puan} olan {ders}in kontenjanı {kont} kişidir")

# Görev 8:Aşağıda 2 adet set verilmiştir. Sizden istenilen eğer 1.küme 2.kümeyi kapsiyor ise ortak elemanlarını eğer
# kapsamıyor ise 2.kümenin 1.kümeden farkını yazdıracak fonksiyonu tanımlamanız beklenmektedir.

kume1 = set(["data", "python"])
kume2 = set(["data", "function", "qcut", "lambda", "python", "miuul"])


def kume_karsilastir(kume1, kume2):
    if kume2.issubset(kume1):
        ortak_elemanlar = kume2.intersection(kume1)
        print({ortak_elemanlar})
    else:
        fark = kume2.difference(kume1)
        print(fark)


kume_karsilastir(kume1, kume2)

#######################################################
# List Comprehension Alıştırmalar
#######################################################

# Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini
# büyük harfe çeviriniz ve başına NUM ekleyiniz.

import seaborn as sns

data = sns.load_dataset("car_crashes")
df = data.copy()

df.columns = ["NUM_" + col.upper() if df[col].dtypes in ["int64", "float64"] else col.upper() for col in df.columns]

# Görev 2:  List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin
# isimlerinin sonuna"FLAG" yazınız

[col + "_FLAG" for col in df.columns if "no" not in df.columns]

# Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin
# isimlerini seçiniz ve yeni bir data frame oluşturunuz.

of_list = ["abbrev", "no_previous"]

new_columns = [col for col in df.columns if col not in of_list]
new_df = df[new_columns]
new_df.head()

#######################################################
# PANDAS ALIŞTIRMALAR
#######################################################

# Görev 1: Seaborn kütüphanesi içerisinden Titanicveri setini tanımlayınız
import pandas as pd
import seaborn as sns

data = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = data.copy()

# Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
df["sex"].value_counts()

# Görev3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
for col in df.columns:
    dict = {col: df[col].nunique()}
    print(dict)

# Görev4: pclass değişkeninin unique değerlerinin sayısını bulunuz.
df["pclass"].nunique()

# Görev5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
df[["pclass", "parch"]].nunique()

# Görev6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
df["embarked"].dtypes
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtypes

# Görev7: embarked değeri C olanların tüm bilgelerini gösteriniz.
df[df["embarked"] == "C"].head()

# Görev8: embarked değeri S olmayanların tüm bilgelerini gösteriniz.
df[df["embarked"] != "S"].head()

# Görev9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.
df[(df["age"] < 30) & (df["sex"] == "female")].head()

# Görev10: Fare'i 500'den büyük ve yayaşı 70 den büyük yolcuların bilgilerini gösteriniz.
df[(df["fare"] > 500) & (df["age"] > 70)].head()

# Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
df.isnull().sum()

# Görev 12: who değişkenini dataframe’de nçıkarınız.
df.drop("who", axis=1)

# Görev13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz
df["deck"].fillna(df["deck"].mode().iloc[0])

# Görev14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"].fillna((df["age"].median()))

# Görev15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.
df.groupby(["pclass", "sex"]).agg({"survived": ["sum", "count", "mean"]})

# Görev16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz.
# (apply ve lambda yapılarını kullanınız)

df["age_flag"] = df["age"].apply(lambda x: 1.30 if x < 30 else 0)

# Görev17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.
data = sns.load_dataset("Tips")
df = data.copy()
df.head()

# Görev18: Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını
# min, max ve ortalamasını bulunuz.
df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"]})

# Görev19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
df.groupby("day").agg({"total_bill": ["sum", "min", "max", "mean"]})

# Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip  değerlerinin day'e göre
# toplamını, min, max ve ortalamasını bulunuz
new_df = df[(df["time"] == "Lunch") & (df["sex"] == "Female")]
new_df.groupby("time").agg({"total_bill": ["sum", "min", "max", "mean"],
                            "tip": ["sum", "min", "max", "mean"]})

# Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)
df_average = df[(df["total_bill"] > 10) & (df["size"] < 3)]
df_average["total_bill"].mean()

# Görev22: total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği totalbill ve tip in
# toplamını versin
df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]

# Görev23:  total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi
# yeni bir dataframe'e atayınız.

df = df.sort_values(by = ["total_bill_tip_sum"], ascending = False)
new_df = df.reset_index(drop= True)
new_df[0:30]
