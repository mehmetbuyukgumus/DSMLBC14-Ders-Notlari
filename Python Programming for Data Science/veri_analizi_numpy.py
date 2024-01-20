#######################
# Numpy Giriş
#######################

import numpy as np

a = [1,2,3,4]
b = [2, 3, 4, 5]

a = np.array([1, 2, 3, 4])
b = np.array([2, 3, 4, 5])

a*b

###########################
# Numpy Array Oluşturmak
##########################

np.array([1,2,3,4,5])
type(np.array([1,2,3,4,5]))

np.zeros(5)
np.random.random_integers(0,10, 10)
np.random.normal(10,4,(3,4))

###########################
# Numpy Array Özellikleri
##########################

a = np.random.randint(0,10, size=5)

# ndim
a.ndim
# shape
a.shape
# size
a.size
# dtype
a.dtype

###########################
# Reshaping
##########################

a = np.random.randint(0, 10, 9)
a.reshape(3,3)

###########################
# Index seçimi
###########################

a = np.random.randint(10, size=10)
a[0]
a[0:3]
a[0] = 999

m = np.random.randint(0,10, size= (3,3))
m[2,2]
m[:, 2]
m[0:2, 0:2]

###########################
# Fancy index
###########################

v = np.arange(0, 30, 3)
catch = [1,3,9]
v[catch]

###########################
# Numpy Koşullu İşlemler
###########################

v[v > 3]
v[v > 23]
v[v != 27]

###########################
# Numpy ile matematiksel işlemler
###########################

v = np.array([1,2,3,4,5])
v * 2
np.sum(v) # Arraydeki tüm elemanları toplar
np.add(v,9) # verilen değer ile arraydeki tüm elemanları toplar
np.subtract(v,5) # verilen değer ile arraydeki tüm elamanları çıkartır
np.mean(v) # Arrayde yer alan elamanların ortalamsını alır
np.min(v) # Arrayde yer alan en küçük değeri verir
np.max(v) # Arrayde yer alan en büyük değeri verir
np.var(v) # Arrayin varyansını hesaplar

###########################
# Numpy ile denklem çözmek
###########################

# 5*x + y = 12
# x + 3*y = 10

a = np.array([[5,1], [1,3]])
b = np.array([12,10])
np.linalg.solve(a,b)
