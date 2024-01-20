############################
# List Comprehensions
############################
salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(x):
    return x * 20 / 100 + x


for salary in salaries:
    print(new_salary(salary))

maaslar = [salary * 2 for salary in salaries]
print(maaslar)

notlar = [10, 20, 40, 45, 64, 24]

[i * 2 for i in notlar]
[i * 2 for i in notlar if i < 30]

[i * 2 if i < 30 else i * 0 for i in notlar]  # Eğer else yapısı kullanılacaksa for döngüsü sonda yer alır

players = ["Muslera", "Torreira", "Icardi", "Ndombele", "Tete"]
players_no = ["Ndombele", "Tete"]

[player.lower() if player in players_no else player.upper() for player in players]

############################
# Dict Comprehensions
############################

dict = {"a": 1,
        "b": 2,
        "c": 3,
        "d": 4}
dict.keys()
dict.values()
dict.items()

{k: v ** 2 for (k, v) in dict.items()}

############################
# Uygulama
############################

# Amaç: çift sayıların karesini alarak bir sözlüğe kaydetmek

numbers = range(10)
new_dict = {}

for n in numbers:
    if n % 2 == 0:
        new_dict[n] = n ** 2

# İşlemin dictionary comprehension yöntemiyle yapılışı;

{n: n ** 2 for n in numbers if n % 2 == 0}

############################
# Bir veri setindeki değişkenlerin ismini değiştirmek
############################

# veri setindeki tüm başlıkları büyük harf yapmak
import seaborn as sns

df = sns.load_dataset("car_crashes")
df.columns = [col.upper() for col in df.columns]

# "INS" ile başlayan ifadelerin başına "FLAG" ifadesini eklemek

df.columns = ["FLAG_" + col if col.startswith("INS") else "NO_FLAG_" + col for col in df.columns]

