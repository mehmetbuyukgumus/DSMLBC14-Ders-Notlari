##########################
# Fonksiyon Tanımlama
##########################
def calculate(x):
    print(x * 2)


def power(x, y):
    print(x ** y)


calculate(10)
calculate(12)
power(2, 3)
power(99, 99)


##########################
# Docsting
##########################
def summer(x, y):
    """

    Args:
        x: int, float
        y: int, float

    Returns:
        int, float
    """
    print(x + y)


summer(10, 20)
summer(3, 5)


###################################
# Fonksiyonların Gövde Bölümü
###################################

def say_hi():
    print("Hi")
    print("Hello")


say_hi()

list_store = []


def append_list(a, b):
    c = a * b
    list_store.append(c)
    print(list_store)


append_list(2, 4)
append_list(99, 99)


###################################
# Default Parameters
###################################

def say_hi(string="Python"):
    print(f"Hello {string}")


say_hi("John")


def puantaj(maas, saat, gun):
    maas = maas * 1
    saat = saat * 2
    gun = gun / 365
    puan = maas + saat + gun
    return maas, saat, gun, puan


puantaj(1000, 12, 13)


