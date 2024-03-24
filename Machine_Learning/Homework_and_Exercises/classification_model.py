### GÖREV 1
# Müşterinin churn olup olmama durumunu tahminleyen bir sınıflandırma modeli oluşturulmuştur.
# 10 test verisi gözleminin gerçek değerleri ve modelin tahmin ettiği olasılık değerleri verilmiştir.
# - Eşikdeğerini 0.5 alarak confusion matrix oluşturunuz. - Accuracy,Recall,Precision,F1Skorlarını hesaplayınız.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

data = pd.read_csv('datasets/class_model.csv', sep=";")
df = data.copy()
df.columns = ['Gerçek Değer', 'Model Olasılık Tahmini']
df["Tahmin"] = df["Model Olasılık Tahmini"].apply(lambda x: 1 if x > 0.50 else 0)

## Confusion Matriksi
y = df['Gerçek Değer']
y_pred = df["Tahmin"]

confusionMatrix = confusion_matrix(df["Gerçek Değer"], df["Tahmin"])


def plot_confusion_matrix(y, y_pred):
    """
    Başarı skorları üzerinden karmaşıklık matrisini görselleştirmeyi sağlayan fonksiyon.
    Parameters
    ----------
    y : Gerçek değerleri ifade eder
    y_pred : Tahmin edilen değerleri ifade eder

    Returns
    -------
    Herhangi bir değer return etmez.

    Notes
    -------
    Fonksiyonun seaborn ve matplotlib, "from sklearn.metrics import accuracy_score, confusion_matrix" kütüphanelerine
    bağlımlılığı vardır.
    """
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()


plot_confusion_matrix(y, y_pred)

## Model Evaluation
print(classification_report(df["Gerçek Değer"], df["Tahmin"]))

## GÖREV 2
# Banka üzerinden yapılan işlemler sırasında dolandırıcılık işlemlerinin yakalanması amacıyla sınıflandırma modeli
# oluşturulmuştur. %90.5 doğruluk oranı elde edilen modelin başarısı yeterli bulunup model canlıya alınmıştır.
# Ancak canlıya alındıktan sonra modelin çıktıları beklendiği gibi olmamış, iş birimi modelin başarısız olduğunu
# iletmiştir. Aşağıda modelin tahmin sonuçlarının karmaşıklık matriksi verilmiştir.

# Buna göre;
# - Accuracy,Recall,Precision,F1 Skorlarını hesaplayınız.
# - Veri Bilimi ekibinin gözden kaçırdığı durum ne olabilir yorumlayınız.

new_confusionMatrix = confusionMatrix
new_confusionMatrix[0, 0] = 5  # True Positif
new_confusionMatrix[0, 1] = 5  # False Nefatif
new_confusionMatrix[1, 0] = 90  # False Positif
new_confusionMatrix[1, 1] = 900  # True Negatif

TP = new_confusionMatrix[0, 0]
FN = new_confusionMatrix[0, 1]
FP = new_confusionMatrix[1, 0]
TN = new_confusionMatrix[1, 1]

accuracy_model = (TP + TN) / 1000
recall_model = TP / (TP + FN)
precision_model = TP / (TP + FP)
f1_score = 2 * (precision_model * recall_model) / (precision_model + recall_model)

print(f"""
############ ÖZET TABLO ############
Accuracy Score: {accuracy_model}
Recall Score: {recall_model}
Precision Score: {precision_model: .4f}
F1 Score: {f1_score: .4f}
####################################
""")
