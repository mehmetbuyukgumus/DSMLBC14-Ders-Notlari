############################################
# Veri Görselleştirme Matplotlib ve Seaborn
############################################

############################################
# Matplotlib
############################################

# Kategorik değişkenler "SÜTUN GRAFİK" ile görselleştirilir. Matplotlib içinde countplot, Seaborn içindeki barplot
# Sayısal değişkenler histogram veya boxplot

# Kategorik değişken görselleştirme

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("titanic")
pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df.head()
df.info()

df["sex"].value_counts().plot(kind="bar")
plt.show()

# Sayısal değişken görselleştirme
plt.hist(df["age"])
plt.show()
plt.boxplot(df["fare"])
plt.show()

############################################
# Matplotlib'in özellikleri
############################################

# plot
x = np.array([1,8])
y = np.array([0,150])

plt.plot(x,y)
plt.show()
plt.plot(x,y, "o")
plt.show()

x = np.array([2, 4, 6, 8, 10])
y = np.array([1, 3, 5, 7, 9])
plt.plot(x,y, "o")
plt.show()

# marker
y = np.array([13,28,11,43])
plt.plot(y, marker = "o")
plt.show()
plt.plot(y, marker = "*")
plt.show()

# line
y = np.array([13,28,31,100,57])
plt.plot(y, linestyle= "dashdot", color = "r")
plt.show()

x = np.array([23, 18, 31, 10])
y = np.array([13, 28, 11, 100])
plt.plot(x)
plt.plot(y)
plt.show()

# Labels
x = np.array([80,85,90,95,100])
y = np.array([240,250,260,270,280])
plt.plot(x,y)
# Başlık ekleme
plt.title("Ana Başlık")
# X label ekleme
plt.xlabel("X Label")
# Y label ekleme
plt.ylabel("Y Label")
# Grid ekleme
plt.grid()
plt.show()

# Subplots
# plot 1
x = np.array([80,85,90,95,100])
y = np.array([240,250,260,270,280])
plt.title("1")
plt.xlabel("X Label")
plt.subplot(1,2,1)
plt.plot(x,y)
plt.show()

# plot 2
x = np.array([80,85,90,95,100])
y = np.array([240,250,260,270,280])
plt.title("1")
plt.xlabel("X Label")
plt.subplot(1,2,2)
plt.plot(x,y)
plt.show()

############################################
# Seaborn ile Veri Görselleştirme
############################################

df = sns.load_dataset("tips")
df.head()

sns.countplot(x = df["sex"],
              data= df)
plt.show()

############################################
# Seaborn ile Sayısal Veri Görselleştirme
############################################

sns.boxplot(x = df["total_bill"],
            data= df)
plt.show()

df["total_bill"].hist()
plt.show()
