##############################
# Koşullar (Conditions)
##############################

# if
a = int(input("Lütfen şifrenizi giriniz: "))
if a == 1:
    print("Hoş geldiniz")

# else
a = int(input("Lütfen şifrenizi giriniz: "))
if a == 1:
    print("Hoş geldiniz")
else:
    print("Şifre hatalı")

# elif
def number_check(number):
    if number > 10:
        print("Sayı 10'dan büyük")
    elif number < 10:
        print("Sayı 10'dan küçük")
    else:
        print("Sayı 10'a eşit")

number_check(12)

##############################
# Döngüler (Loops)
##############################

players = ["Muslera", "Nelson", "Toreira", "Mertens", "Icardi"]

for player in players:
    print(player.upper())

##############################
# Uygulama
##############################

# before = "hi my name is John and i am learning python"
# after = "hi mY NaMe iS JoHn aNd i aM LeArNiNg pYtHoN"

def alternating(string):
    new_string = ""
    for string_index in range(len(string)):
        if string_index % 2 == 0:
            new_string += string[string_index].upper()
        else:
            new_string += string[string_index].lower()
    print(new_string)

alternating("hi my name is John and i am learning python")

##############################
# while
##############################

number = 1
while number < 5:
    print(number)
    number += 1

##############################
# Enumarate
##############################

players = ["Muslera", "Davinson", "Torreira", "Mertens", "Icardi"]

A = []
B = []

for index, player in enumerate(players):
    if index % 2 == 0:
        A.append(player)
    else:
        B.append(player)
print(A)
print(B)

##############################
# Enumarate - Uygulama
##############################

GS_Takim = ["Muslera", "Davinson", "Torreira", "Mertens", "Icardi"]
GS_Takim_Divide = [[],[]]


for index, player in enumerate(GS_Takim):
    if index % 2 == 0:
        GS_Takim_Divide[0].append(player)
    else:
        GS_Takim_Divide[1].append(player)
print(GS_Takim_Divide)

##############################
# Zip
##############################

team = ["Galatasaray", "Fenerbahçe", "Beşiktaş"]
players = ["Muslera", "Dzeko", "Gedson"]
position = ["GK","STR","MC"]
new_list = list(zip(team, players, position))
new_list[0]

##################################
# Lambda & Map & Filter & Reduce
##################################

# Lambda
new_number = lambda a,b : a + b
new_number(3,2)

# Map
salaries = [1000, 2000, 3000, 4000, 5000]


def new_salary(salary):
    new_salary = salary * 20 / 100 + salary
    print(new_salary)


new_salary(1000)
list(map(new_salary, salaries))


