import pandas as pd
import numpy as np
from feature_engeineering.fucs.helper_functions import grab_col_names, outlier_thresholds, check_outlier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_validate, GridSearchCV

data = pd.read_csv("datasets/Telco-Customer-Churn.csv")
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)

## Görev 1 : Keşifçi Veri Analizi

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
df = data.copy()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df["TotalCharges"] = df["TotalCharges"].astype(float)
df["TotalCharges"].replace(" ", np.nan, inplace=True)

## cat_cols, num_cols, cat_but_car = grab_col_names(df) bir daha çalıştırmak gerekiyor.

# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
num_scatter = len(num_cols) / len(df.columns)
cat_scatter = len(cat_cols) / len(df.columns)
car_scatter = len(cat_but_car) / len(df.columns)

print(f"""
Numerik Değişkenler: {num_scatter: .2%}
Kategorik Değişkenler: {cat_scatter: .2%}
Kardinal Değişkenler : {car_scatter: .2%} 
""")

# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
for col in cat_cols:
    print(df.groupby([col, "Churn"]).agg({"Churn": "count"}))

# Adım 5: Aykırı gözlem var mı inceleyiniz.
for col in num_cols:
    low, up = outlier_thresholds(df, col, q1=0.25, q3=0.75)
    print(f"{col} : {low}, {up}")

for col in num_cols:
    print(f"{col} : {check_outlier(df, col)}")

# Adım 6: Eksik gözlem var mı inceleyiniz.
df.isnull().sum()

# Görev 2 : Feature Engineering
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
df = df.dropna()

# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
df.head()
mapping = {"Yes": 1,
           "No": 0}

yes_no_cols = [col for col in df.columns if df[col].isin(["Yes", "No"]).all()]

for col in yes_no_cols:
    df[col] = df[col].map(mapping)

other_cols = [col for col in df.columns if
              not (df[col].isin([0, 1]).all()) and (col != "customerID") and (
                          df[col].dtype not in ["float64", "int64"])]

df = pd.get_dummies(df, columns=other_cols, drop_first=True)
df = df.drop("customerID", axis=1)
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
y = df["Churn"]
X = df.drop("Churn", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=X.columns)

# Görev 3 : Modelleme
# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.

knn_model = KNeighborsClassifier()
knn_model.fit(X, y)
knn_model.get_params()

y_pred = knn_model.predict(X)
y_prop = knn_model.predict_proba(X)

print(classification_report(y, y_pred))

roc_auc_score(y, y_pred)
# 0.76

cv_results = cross_validate(knn_model, X, y, cv=4, scoring=["accuracy", "f1", "roc_auc"])

cv_results["test_accuracy"].mean()
# 0.75
cv_results["test_f1"].mean()
# 0.52
cv_results["test_roc_auc"].mean()
#0.77

# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile
# modeli tekrar kurunuz.
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model, knn_params, cv=4, n_jobs=-1, verbose=1)
knn_gs_best.fit(X,y)

knn_gs_best.best_params_
# 34

knn_final_model = knn_model.set_params(**knn_gs_best.best_params_)
knn_final_model.fit(X,y)

cv_results_final = cross_validate(knn_final_model, X,y, cv=4, scoring=["accuracy", "f1", "roc_auc"])

cv_results_final['test_accuracy'].mean()
# 0.79
cv_results_final['test_f1'].mean()
# 0.58
cv_results_final['test_roc_auc'].mean()
# 0.82

random_user = X.sample(1)
knn_final_model.predict(random_user)
