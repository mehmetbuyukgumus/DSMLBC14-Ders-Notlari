SELECT * FROM CUSTOMERS;
UPDATE CUSTOMERS SET COUNTRY = "TÜRKİYE";
UPDATE CUSTOMERS SET AGE = (YEAR(CURDATE()) - YEAR(BIRTHDATE));
UPDATE CUSTOMERS SET `AGEGROUP` = "GENÇ" WHERE AGE BETWEEN 20 and 40;
UPDATE CUSTOMERS SET `AGEGROUP` = "ORTA YAŞ" WHERE AGE BETWEEN 40 and 50;
UPDATE CUSTOMERS SET `AGEGROUP` = "YAŞLI" WHERE AGE > 50;
SELECT * FROM CUSTOMERS WHERE AGEGROUP = "ORTA YAŞ";
