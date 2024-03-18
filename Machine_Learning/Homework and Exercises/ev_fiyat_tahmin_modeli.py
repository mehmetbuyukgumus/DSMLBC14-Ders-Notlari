# General Importing
import pandas as pd
import numpy as np

# Importing for EDA
from feature_engeineering.fucs.helper_functions import outlier_thresholds, check_outlier, \
    replace_with_thresholds
from sklearn.preprocessing import StandardScaler

# Importing for Modeling
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, make_scorer
from sklearn.model_selection import cross_validate, GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


# Veriyi Okutma
def loading_datas():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 500)
    test = pd.read_csv('datasets/hause_price_test.csv')
    train = pd.read_csv('datasets/hause_price_train.csv')
    return train, test


df, df_test = loading_datas()
datasets = [df, df_test]

# Değişken Analizi
num_cols = list(df.select_dtypes(include=['float', "int"]).columns)
cat_cols = list(df.select_dtypes(include=['O']).columns)
num_cols.remove("Id")
num_cols.remove("SalePrice")
# Eksik ve Aykırı Gözlem Analizi
df[num_cols].describe().T
low, up = outlier_thresholds(df, num_cols, q1=0.05, q3=0.95)
for col in num_cols:
    print(f"{col}: {check_outlier(df, col)}")
# Aykırı Değerleri Baskılama
for dataset in datasets:
    for col in num_cols:
        replace_with_thresholds(dataset, col)

# Eskik Gözlemleri Doldurma
df.isnull().sum()
## Eksik Gözlemleri Bir Üstteki Değer İle Doldurma
for dataset in datasets:
    for col in dataset.columns:
        dataset[col] = dataset[col].fillna(method="ffill")
# Kolerasyon Analizi
corr_matrix = pd.DataFrame(df.corr()["SalePrice"].drop("SalePrice"))
corr_matrix = corr_matrix.sort_values(by="SalePrice", ascending=False)
corr_matrix.index[0:5]

# Encoding
meta_data = pd.concat([df, df_test], axis=0, ignore_index=True)
meta_data = pd.get_dummies(meta_data, cat_cols, drop_first=True)
meta_data_test = meta_data[meta_data["SalePrice"].isnull()]
meta_data = meta_data[~meta_data["SalePrice"].isnull()]
meta_data_test = meta_data_test.drop("SalePrice", axis=1)

# Modelleme
y = meta_data["SalePrice"]
X = meta_data.drop(["SalePrice", "Id"], axis=1)


def err_score(y_true, y_pred):
    return root_mean_squared_error(y_true, y_pred)


rmse_scorer = make_scorer(err_score, greater_is_better=False)

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(),
          "GBM Regressor": GradientBoostingRegressor(),
          "XGB Regression": XGBRegressor(),
          "LGBM": LGBMRegressor(),
          "CatBoost": CatBoostRegressor()}

models_results = {}

for name, model in models.items():
    cv_results = cross_validate(model, X, y, scoring=rmse_scorer, cv=10)
    models_results[name] = np.mean(np.abs(cv_results["test_score"]))

best_model = min(models_results, key=models_results.get)
best_score = models_results[best_model]

print("En iyi model:", best_model)
print("En iyi skor:", best_score)
print(f"Gözlem Başına Hata: {best_score / len(X)}")
# RMSE = 25173.565136917787

# Hiperparametre Optimizasyonu
cat_model = CatBoostRegressor(random_state=17).fit(X, y)
cv_results_model = cross_validate(cat_model, X, y, scoring=rmse_scorer, cv=10)
cv_results_model["test_score"].mean()
# -25438.750213363783
catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}
cat_boost_best_gird = GridSearchCV(cat_model, catboost_params, cv=10, n_jobs=-1, verbose=True).fit(X,y)
cat_boost_best_gird.best_params_

# Final Model
cat_model_final = CatBoostRegressor(**cat_boost_best_gird.best_params_, random_state=17).fit(X,y)
cv_results_final = cross_validate(cat_model_final, X,y, scoring=rmse_scorer, cv=10)
cv_results_final["test_score"].mean()
# -25713.45667944349

# Tahmin
X_final = meta_data_test.drop("Id", axis=1)
cat_model.predict(X_final)
output = pd.DataFrame()
output["Id"] = meta_data_test["Id"]
output["SalePrice"] = cat_model.predict(X_final)
output.to_csv("datasets/output.csv", index=False)
