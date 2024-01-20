##################################
# VERİ YAPILARI (DATA STRUCTURE)
##################################

# Sayılar (Integer):
x = 46
print(type(x))

# Sayılar (Float):
x = 10.3
print(type(x))

# Sayılar (Complex):
x = 2j + 1
print(type(x))

# String (Karakter Dizileri):
x = "Hello World"
print(type(x))

# Bolean:
x = True
y = False
print(type(x))
print(type(y))

# Listeler:
x = ["a", "b", "c"]
print(type(x))

# Sözlük (Dictionary):
x = {"Name": "Mehmet", "Surname": "Büyükgümüş"}
print(type(x))

# Tuple (Demet):
x = ("a", "b")
print(type(x))

# Set:
x = {"a", "b"}
print(type(x))

########################################
# Sayılar ve veri tiplerini değiştirmek
########################################

number1 = 12
number2 = 23.7
number3 = int(number2)
print(number3)

########################################
# Karakter Dizileri (String)
########################################

print("Mehmet")
name = "Mehmet Büyükgümüş"
print(name)
"Ali"

########################################
# String Methods
########################################

# Len
myname = "Mehmet"
print(len(myname))

# Upper and Lower
myname.upper()
myname.lower()

# Replace
myname.replace("Mehmet", "Ahmet")

# Split
name_and_surname = "Hasan Ali Kaldırım"
name_and_surname.split()

# Strip
name_and_surname.strip()

# Capitilaze
"folik".capitalize()

########################################
# Listeler (Lists)
########################################

notes = [1,2,3,4]
notes_complex = [1,2,3, True, False, None, [1,2,3]]
notes_complex[6] = "John"
print(notes_complex)

# Append
notes.append(5)
print(notes)

# Pop
notes.pop(5)

# Insert
notes.insert(2, 99)

########################################
# Sözlükler (Dictionary)
########################################

dict = {"Name": "John",
        "Surname": "Kenedy"}

# Key Sorgulama
"Age" in dict
"Name" in dict

# Sözlük içinde değer değiştirmek
dict["Name"] = "Bill"

# Tüm key değerlerine erişmek
dict.keys()

# Key ve Value değerlerini güncellemek
dict.update({"Age": 26})

########################################
# Demetler (Tuples)
########################################

tup = ("John", "Kenedy", 12)
tup[0]


