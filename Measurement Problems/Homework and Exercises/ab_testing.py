#####################################################
# AB Testi ile BiddingYöntemlerinin Dönüşümünün Karşılaştırılması
#####################################################

#####################################################
# İş Problemi
#####################################################

# Facebook kısa süre önce mevcut "maximumbidding" adı verilen teklif verme türüne alternatif
# olarak yeni bir teklif türü olan "average bidding"’i tanıttı. Müşterilerimizden biri olan bombabomba.com,
# bu yeni özelliği test etmeye karar verdi veaveragebidding'in maximumbidding'den daha fazla dönüşüm
# getirip getirmediğini anlamak için bir A/B testi yapmak istiyor.A/B testi 1 aydır devam ediyor ve
# bombabomba.com şimdi sizden bu A/B testinin sonuçlarını analiz etmenizi bekliyor.Bombabomba.com için
# nihai başarı ölçütü Purchase'dır. Bu nedenle, istatistiksel testler için Purchasemetriğine odaklanılmalıdır.


#####################################################
# Veri Seti Hikayesi
#####################################################

# Bir firmanın web site bilgilerini içeren bu veri setinde kullanıcıların gördükleri ve tıkladıkları
# reklam sayıları gibi bilgilerin yanı sıra buradan gelen kazanç bilgileri yer almaktadır.Kontrol ve Test
# grubu olmak üzere iki ayrı veri seti vardır. Bu veri setleriab_testing.xlsxexcel’ininayrı sayfalarında yer
# almaktadır. Kontrol grubuna Maximum Bidding, test grubuna AverageBiddinguygulanmıştır.

# impression: Reklam görüntüleme sayısı
# Click: Görüntülenen reklama tıklama sayısı
# Purchase: Tıklanan reklamlar sonrası satın alınan ürün sayısı
# Earning: Satın alınan ürünler sonrası elde edilen kazanç


#####################################################
# Proje Görevleri
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı (shapiro)
#   - 2. Varyans Homojenliği (levene)
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direkt 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


#####################################################
# Görev 1:  Veriyi Hazırlama ve Analiz Etme
#####################################################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr
from statsmodels.stats.proportion import proportions_ztest
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
# Adım 1:  ab_testing_data.xlsx adlı kontrol ve test grubu verilerinden oluşan veri setini okutunuz. Kontrol ve test
# grubu verilerini ayrı değişkenlere atayınız.
data_control_group = pd.read_excel('datasets/ab_testing.xlsx', sheet_name='Control Group')
data_test_group = pd.read_excel('datasets/ab_testing.xlsx', sheet_name='Test Group')

# Adım 2: Kontrol ve test grubu verilerini analiz ediniz.
data_control_group.describe().T
data_test_group.describe().T
# Adım 3: Analiz işleminden sonra concat metodunu kullanarak kontrol ve test grubu verilerini birleştiriniz.
data_control_group.columns = [col + "_controls" for col in data_control_group.columns]
data_test_group.columns = [col + "_tests" for col in data_control_group.columns]
df = pd.concat([data_control_group, data_test_group], axis=1)
#####################################################
# Görev 2:  A/B Testinin Hipotezinin Tanımlanması
#####################################################

# Adım 1: Hipotezi tanımlayınız.
# H0 : M1 = M2 Reklamların görüntülenme ortalamasıyla satış ortalaması arasında bir fark yoktur.
# H1 : M1 != M2 ... vardır.

# Adım 2: Kontrol ve test grubu için purchase(kazanç) ortalamalarını analiz ediniz
df["Purchase_controls"].describe().T
df["Purchase_tests"].describe().T

sns.scatterplot(x="Impression_controls",
                y="Purchase_controls",
                label="Purchase_controls",
                data=df)
sns.scatterplot(x="Impression_tests",
                y="Purchase_tests",
                label="Purchase_controls",
                data=df)

plt.grid(True)
plt.show()

df[(df["Impression_controls"] < 80000) | (df["Impression_tests"] < 80000)]

#####################################################
# GÖREV 3: Hipotez Testinin Gerçekleştirilmesi
#####################################################

######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################


# Adım 1: Hipotez testi yapılmadan önce varsayım kontrollerini yapınız.Bunlar Normallik Varsayımı ve Varyans
# Homojenliğidir.

# Kontrol ve test grubunun normallik varsayımına uyup uymadığını Purchase değişkeni üzerinden ayrı ayrı test ediniz
## Normallik Varsayımı
test_stats, pvalue = shapiro(df["Purchase_controls"])
print(f"Test Stats: {test_stats} and p_value: {pvalue}")

test_stats, pvalue = shapiro(df["Purchase_tests"])
print(f"Test Stats: {test_stats} and p_value: {pvalue}")

## Varyans Homojenliği
test_stats, pvalue = levene(df["Purchase_controls"], df["Purchase_tests"])
print(f"Test Stats: {test_stats} and p_value: {pvalue}")

# Adım 2: Normallik Varsayımı ve Varyans Homojenliği sonuçlarına göre uygun testi seçiniz.
## Varsayım kontrolleri sonucu varsayımlar sağlanmadığı için mannwhitneyu testi yapılacaktır.
test_stats, plavue = mannwhitneyu(df["Purchase_controls"], df["Purchase_tests"])
print(f"Test Stats: {test_stats} & p_value: {plavue}")

# Adım 3: Test sonucunda elde edilen p_value değerini göz önünde bulundurarak kontrol ve test grubu satın alma
# ortalamaları arasında istatistiki olarak anlamlı bir fark olup olmadığını yorumlayınız.
## Yapılan test sonucunda test ve kontrol grubu arasında istatistiki olarak anlamlı bir fark yoktur.

##############################################################
# GÖREV 4 : Sonuçların Analizi
##############################################################

# Adım 1: Hangi testi kullandınız, sebeplerini belirtiniz.
## Çalışmanın akışında mannwithneyu testi kullandım. Kurulan hipotez itibariyle normallik varsayımı sağlansa dahi
## varyans homojemliği varsayımı sağlanmamıştır. Bu sebeple mannwithneyu testi kullanılmıştır.

# Adım 2: Elde ettiğiniz test sonuçlarına göre müşteriye tavsiyede bulununuz.
## Yapılan hipotez testi sonucunda Facebook'un geliştridği yeni average bidding yöntemi şirket için maddi bir yük
## yaratacaksa yeni yönteme geçmemesi tavsiye edilir. Zira Maximum Bidding yöntemiyle Average Bidding yöntemi arasında
## istatistiki bir farklılık yoktur.
