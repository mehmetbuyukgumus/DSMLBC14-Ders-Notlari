############ İş Problemi ############

# Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin edebilecek bir makine öğrenmesi
# modeli geliştirilmesi istenmektedir. Modeli geliştirmeden önce gerekli olan veri analizi ve özellik mühendisliği
# adımlarını gerçekleştirmeniz beklenmektedir.

############ Veri Seti Hikayesi ############

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. Hedef değişken "outcome"
# olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

############ Görev 1 : Keşifçi Veri Analizi ############
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('datasets/diabetes.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = data.copy()
df.head()

# Adım 1: Genel resmi inceleyiniz.
df.describe().T  #
df.info()  # Veri setindeki tüm değişkenler numeric
df.isnull().sum()  # Veri setinde hiç eksik gözlem yok


# Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
for col in num_cols:
    print(df[col].describe())

df.loc[:, cat_cols].info()

################# Yasin'in çözümü
df.select_dtypes(include=['int', 'float']).head()  # Numeric Variables
df.select_dtypes(include=['object'])  # Categorical Variables

################# Ömer Faruk'un çözümü
# def cat_summary(dataframe, col_name, plot=False):
#     """
#
#     Fonksiyon, veri setinde yer alan kategorik, numerik vs... şeklinde gruplandırılan değişkenler için özet bir çıktı
#     sunar.
#
#     Parameters
#     ----------
#     dataframe : Veri setini ifade
#     col_name : Değişken grubunu ifade eder
#     plot : Çıktı olarak bir grafik istenip, istenmediğini ifade eder, defaul olarak "False" gelir
#
#     Returns
#     -------
#     Herhangi bir değer return etmez
#
#     Notes
#     -------
#     Fonksiyonun pandas, seaborn ve matplotlib kütüphanelerine bağımlılığı vardır.
#
#     """
#     print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
#                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
#     print("##########################################")
#     if plot:
#         sns.countplot(x=dataframe[col_name], data=dataframe)
#         plt.show()
# print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
#                         "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
#     print("##########################################")
#     if plot:
#         sns.countplot(x=dataframe[col_name], data=dataframe)
#         plt.show()
#
# cat_summary(df, "Outcome", True)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [.01, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

# Adım 4: Hedef değişken analizi yapınız. (Kategorik değişkenlere göre hedef değişkenin ortalaması, hedef
# değişkene göre numerik değişkenlerin ortalaması)
for col in num_cols:
    target_als_cat_cols = df.groupby(df["Outcome"]).agg({col: "mean"})
    print(target_als_cat_cols)

######## Yasin'in çözümü
categorical_means = df.groupby('Insulin')['Outcome'].mean()
numeric_means = df.groupby('Outcome').mean()
categorical_means
numeric_means

# Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Bir dataframe için verilen ilgili kolondaki aykırı değerleri tespit edebilmek adına üst ve alt limitleri belirlemeyi
    sağlayan fonksiyondur

    Parameters
    ----------
    dataframe: "Dataframe"i ifade eder.
    col_name: Değişkeni ifade eder.
    q1: Veri setinde yer alan birinci çeyreği ifade eder.
    q3: Veri setinde yer alan üçüncü çeyreği ifade eder.

    Returns
    -------
    low_limit, ve up_limit değerlerini return eder
    Notes
    -------
    low, up = outlier_tresholds(df, col_name) şeklinde kullanılır.
    q1 ve q3 ifadeleri yoru açıktır. Aykırı değerle 0.01 ve 0.99 değerleriyle de tespit edilebilir.

    """
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


for cols in num_cols:
    low, up = outlier_thresholds(df, cols)
    print(f"{cols} : {low}, {up}")


def check_outlier(dataframe, col_name):
    """
    Bir dataframein verilen değişkininde aykırı gözlerimerin bulunup bulunmadığını tespit etmeye yardımcı olan
    fonksiyondur.
    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for cols in num_cols:
    print(f"{cols} : {check_outlier(df, cols)}")

# Adım 6: Eksik gözlem analizi yapınız.
df.isnull().sum()  # Bu durumu Adım 1'de gözlemlemiştik.

# Adım 7: Korelasyon analizi yapınız.
heatmap = df.corr()
sns.heatmap(heatmap)
plt.show(block=False)

######### Emre'nin çözümü
cor_matrix = df.corr().abs()

## Görev 2 : Feature Engineering

# Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız. Veri setinde eksik gözlem bulunmamakta ama
# Glikoz, Insulin vb. değişkenlerde 0 değeri içeren gözlem birimleri eksik değeri ifade ediyor olabilir.
# Örneğin; bktıir kişinin glikoz veya insulin değeri 0 olamayacar. Bu durumu dikkate alarak sıfır değerlerini
# ilgili değerlerde NaN olarak atama yapıp sonrasında eksik değerlere işlemleri uygulayabilirsiniz.
df[(df["Glucose"] == 0) | (df["Insulin"] == 0)].shape

df["Glucose"] = df["Glucose"].replace(0, np.nan)
df["Insulin"] = df["Insulin"].replace(0, np.nan)
df["BloodPressure"] = df["BloodPressure"].replace(0, np.nan)
df["SkinThickness"] = df["SkinThickness"].replace(0, np.nan)
df["BMI"] = df["BMI"].replace(0, np.nan)
df[df["Glucose"].isna()]
df[df["Insulin"].isna()]

df["Glucose"] = df["Glucose"].fillna(df.groupby("Outcome")["Glucose"].transform("mean"))
df["Insulin"] = df["Insulin"].fillna(df.groupby("Outcome")["Insulin"].transform("mean"))
df["BloodPressure"] = df["BloodPressure"].fillna(df.groupby("Outcome")["BloodPressure"].transform("mean"))
df["SkinThickness"] = df["SkinThickness"].fillna(df.groupby("Outcome")["SkinThickness"].transform("mean"))
df["BMI"] = df["BMI"].fillna(df.groupby("Outcome")["BMI"].transform("mean"))
df.isnull().sum()

# Adım 2: Yeni değişkenler oluşturunuz.
labes_age = ["Joung", "Adult", "Senior", "Old"]
labels_BMI = ["Thin", "Normal", "Fat", "Obez"]

df["Age_Labels"] = pd.qcut(df["Age"], q=4, labels=labes_age)
df["BMI_Labels"] = pd.qcut(df["BMI"], q=4, labels=labels_BMI)
df = df.drop(["Age_Labels", "BMI_Labels"], axis=1)

########## Can'ın çözümü
df['BMI_Age_Product'] = df['BMI'] * df['Age']
df['BloodPressure_Age_Product'] = df['BloodPressure'] * df['Age']
df['Pregnancies_Age_Difference'] = df['Pregnancies'] - df['Age']
df['Insulin_Glucose_Product'] = df['Insulin'] * df['Glucose']
df['DPF_Age_Sum'] = df['DiabetesPedigreeFunction'] + df['Age']
df['BMI_SkinThickness_Ratio'] = df['BMI'] / df['SkinThickness']

# Adım 3: Encoding işlemlerini gerçekleştiriniz.
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """

    Veri setindeki kategorik değşkenler için one hot encoding işlemini yapar

    Parameters
    ----------
    dataframe : Veri setini ifade eder
    categorical_cols : Kategorik değişkenleri ifade eder
    drop_first : Dummy değişken tuzağına düşmemek için ilk değşşkeni siler

    Returns
    -------
    One-hot encoding işlemi yapılmış bir şekilde "dataframe"i return eder

    Notes
    -------
    Fonksiyonun "pandas" kütüphanesine bağımlılığı bulunmaktadır.
    """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


cat_cols, num_cols, cat_but_car = grab_col_names(df)

cat_cols = [col for col in cat_cols if col not in "Outcome"]

df = one_hot_encoder(df, cat_cols)

# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df[num_cols].head()

# Adım 5: Model oluşturunuz.
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

yeni_kisi = df.iloc[287:288,:]
yeni_kisi = yeni_kisi.reset_index(drop=True)
yeni_kisi.drop("Outcome", axis=1, inplace=True)

rf_model.predict(yeni_kisi)

df.iloc[287:288,:]
