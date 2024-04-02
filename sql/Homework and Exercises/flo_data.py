import mysql.connector

# Connection
mydb = mysql.connector.connect(user='root', password='b-612kpc', host='127.0.0.1', database='FLO')

cursor = mydb.cursor()

# 1. Customers isimli bir veritabanı ve verilen veri setindeki değişkenleri içerecek FLO isimli bir tablo oluşturunuz.
cursor.execute("""
SELECT * FROM flo_data
""")

datas = cursor.fetchall()
for data in datas:
    print(data)

# 2. Kaç farklı müşterinin alışveriş yaptığını gösterecek sorguyu yazınız.
cursor.execute("""
SELECT DISTINCT COUNT(`master_id`) FROM flo_data
""")

datas = cursor.fetchall()
for data in datas:
    print(data)
cursor.close()

# 3. Toplam yapılan alışveriş sayısı ve ciroyu getirecek sorguyu yazınız.
cursor.execute("""
SELECT 
SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`) AS TOPLAM,
COUNT(DISTINCT `master_id`)
FROM flo_data
""")

datas = cursor.fetchall()
for data in datas:
    print(data)

# 4. Alışveriş başına ortalama ciroyu getirecek sorguyu yazınız.
cursor.execute("""
SELECT 
(SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`)) / COUNT(DISTINCT `master_id`)
FROM flo_data
""")

datas = cursor.fetchall()
for data in datas:
    print(data)

# 5. En son alışveriş yapılan kanal (last_order_channel) üzerinden yapılan alışverişlerin toplam ciro ve alışveriş
# sayılarını getirecek sorguyu yazınız.
cursor.execute("""
    SELECT (SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`)) AS TOTAL, last_order_channel
    FROM flo_data 
    GROUP BY last_order_channel
""")

datas = cursor.fetchall()
for data in datas:
    print(data)

# 6. Store type kırılımında elde edilen toplam ciroyu getiren sorguyu yazınız.
# 7. Yıl kırılımında alışveriş sayılarını getirecek sorguyu yazınız (Yıl olarak müşterinin ilk alışveriş tarihi
# (first_order_date) yılını baz alınız.
cursor.execute("""
SELECT (SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`)) AS TOTAL, YEAR(`first_order_date`) 
FROM flo_data
GROUP BY YEAR(`first_order_date`)
""")

datas = cursor.fetchall()
for data in datas:
    print(data)

# 8. En son alışveriş yapılan kanal kırılımında alışveriş başına ortalama ciroyu hesaplayacak sorguyu yazınız.
cursor.execute("""
SELECT
SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`) AS TOTAL,last_order_channel, 
YEAR(`last_order_date`)   
FROM flo_data
GROUP BY last_order_channel, YEAR(`last_order_date`)
""")

datas = cursor.fetchall()
for data in datas:
    print(data)

# 9. Son 12 ayda en çok ilgi gören kategoriyi getiren sorguyu yazınız.
cursor.execute("""
SELECT DISTINCT `interested_in_categories_12`
FROM flo_data
WHERE YEAR(last_order_date) = (SELECT MAX(YEAR(last_order_date)) FROM flo_data) 
AND (SELECT MAX(YEAR(last_order_date)) -1 FROM flo_data) ;
""")
datas = cursor.fetchall()
for data in datas:
    print(data)

# 10. En çok tercih edilen store_type bilgisini getiren sorguyu yazınız.

# 11. En son alışveriş yapılan kanal (last_order_channel) bazında, en çok ilgi gören kategoriyi ve bu kategoriden ne
# kadarlık alışveriş yapıldığını getiren sorguyu yazınız.
cursor.execute("""
SELECT SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`) AS TOTAL, `interested_in_categories_12`
FROM flo_data
GROUP BY `interested_in_categories_12`
ORDER BY TOTAL DESC
LIMIT 5
""")
datas = cursor.fetchall()
for data in datas:
    print(data)

# 12. En çok alışveriş yapan kişinin ID’ sini getiren sorguyu yazınız.
cursor.execute("""
SELECT `master_id` 
FROM flo_data
GROUP BY `master_id`
ORDER BY  SUM(`order_num_total_ever_online`) + SUM(`order_num_total_ever_offline`) DESC
LIMIT 1
""")
datas = cursor.fetchall()
for data in datas:
    print(data)

# 13. En çok alışveriş yapan kişinin alışveriş başına ortalama cirosunu ve alışveriş yapma gün ortalamasını
# (alışveriş sıklığını) getiren sorguyu yazınız.
cursor.execute("""
SELECT 
SUM(`order_num_total_ever_offline`) + SUM(`order_num_total_ever_online`) AS TOTAL,
DATEDIFF((MAX(`last_order_date`)),(MIN(`first_order_date`))) AS DATEDIFF
FROM flo_data
WHERE master_id = '5d1c466a-9cfd-11e9-9897-000d3a38a36f';
""")
datas = cursor.fetchall()
for data in datas:
    print(data)

# 14. En çok alışveriş yapan (ciro bazında) ilk 100 kişinin alışveriş yapma gün ortalamasını (alışveriş sıklığını)
# getiren sorguyu yazınız.
cursor.execute("""
SELECT
DATEDIFF(`last_order_date`, `first_order_date`) AS DATEDIFF
FROM flo_data
GROUP BY `master_id`,last_order_date, first_order_date
ORDER BY SUM(`order_num_total_ever_offline` + `order_num_total_ever_online`) DESC
LIMIT 100;
""")
datas = cursor.fetchall()
for data in datas:
    print(data)

# 15. En son alışveriş yapılan kanal (last_order_channel) kırılımında en çok alışveriş yapan müşteriyi getiren sorguyu
# yazınız.
cursor.execute("""
SELECT `master_id`, `last_order_channel`
FROM flo_data
GROUP BY `master_id`, `last_order_channel`
ORDER BY SUM(`customer_value_total_ever_offline` + `customer_value_total_ever_online`) DESC 
LIMIT 1
""")
datas = cursor.fetchall()
for data in datas:
    print(data)

# 16. En son alışveriş yapan kişinin ID’ sini getiren sorguyu yazınız. (Max son tarihte birden fazla alışveriş yapan
# ID bulunmakta. Bunları da getiriniz.)
cursor.execute("""
SELECT master_id 
FROM flo_data
WHERE `last_order_date` = (SELECT MAX(last_order_date) FROM flo_data)
""")
datas = cursor.fetchall()
for data in datas:
    print(data)
