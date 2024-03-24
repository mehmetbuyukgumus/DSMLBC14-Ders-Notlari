# Çalışanların deneyim yılı ve maaş bilgileri verilmiştir.
# 1-Verilen bias ve weight’e göre doğrusal regresyon model denklemini oluşturunuz. Bias = 275, Weight= 90 (y’ = b+wx)
# 2-Oluşturduğunuz model denklemine göre tablodaki tüm deneyim yılları için maaş tahmini yapınız.
# 3-Modelin başarısını ölçmek için MSE, RMSE, MAE skorlarını hesaplayınız

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
data = pd.read_csv('datasets/maas_verisi.csv', sep= ";")
df_manuel = data.copy()
df_manuel.dropna(inplace=True)

## Model denklemi = y = 275 + 90 * x1

y_pred = []
for value in df_manuel["Deneyim Yılı (x) "]:
    y = 275 + 90 * value
    y_pred.append(y)

df_manuel["Maaş Tahmini (y')"] = pd.Series(y_pred, index=df_manuel.index)
df_manuel["Hata (y-y')"] = df_manuel["Maaş (y) "] - df_manuel["Maaş Tahmini (y')"]
df_manuel["Hata Kareleri"] = np.square(df_manuel["Hata (y-y')"])
df_manuel["Mutlak Hata (|y-y'|)"] = np.abs(df_manuel["Hata (y-y')"])

MSE_manuel = np.mean((df_manuel["Hata Kareleri"]))
RMSE_manuel = np.sqrt(MSE_manuel)
MAE_manuel = np.mean(df_manuel["Mutlak Hata (|y-y'|)"])

#######################################      Sklearn Kullanarak Model Tahmini       ###################################
from sklearn.linear_model import LinearRegression
df_model = data.copy()
df_model.dropna(inplace=True)
X = df_model[["Deneyim Yılı (x) "]]
y = df_model[["Maaş (y) "]]

model = LinearRegression()
model.fit(X,y)
y_pred = model.predict(X)
y_pred = y_pred.flatten()

df_model["Maaş Tahmini (y')"] = pd.Series(y_pred, index=df_model.index)
df_model["Hata (y-y')"] = df_model["Maaş (y) "] - df_model["Maaş Tahmini (y')"]
df_model["Hata Kareleri"] = np.square(df_model["Hata (y-y')"])
df_model["Mutlak Hata (|y-y'|)"] = np.abs(df_model["Hata (y-y')"])

MSE_model = np.mean((df_model["Hata Kareleri"]))
RMSE_model = np.sqrt(MSE_model)
MAE_model = np.mean(df_model["Mutlak Hata (|y-y'|)"])

print(f"""
############# Özet Tablo #############" 
MSE : {MSE_model - MSE_manuel}  
RMSE : {RMSE_model - RMSE_manuel}
MAE : {MAE_model - MAE_manuel}
######################################
""")
