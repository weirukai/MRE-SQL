

BIRD_EXAMPLES="""
###Example 1 -- Question: Among the books ordered by Lucas Wyldbore, what is the percentage of those books over $13?
###Example 1 -- Evidence: books over $13 refers to price > 13; percentage = Divide (Sum (order_id where price > 13), Count (order_id)) * 100,
###Example 1 -- GOLD_SQL: SELECT CAST(SUM(CASE WHEN T1.price > 13 THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(*) FROM order_line AS T1 INNER JOIN cust_order AS T2 ON T2.order_id = T1.order_id INNER JOIN customer AS T3 ON T3.customer_id = T2.customer_id WHERE T3.first_name = 'Lucas' AND T3.last_name = 'Wyldbore'

###Example 2 -- Question: Which mountain is the highest in an independent country?,
###Example 2 -- Evidence: "",
###Example 2 -- GOLD_SQL: "SELECT T4.Name FROM country AS T1 INNER JOIN politics AS T2 ON T1.Code = T2.Country INNER JOIN geo_mountain AS T3 ON T3.Country = T2.Country INNER JOIN mountain AS T4 ON T4.Name = T3.Mountain WHERE T2.Independence IS NOT NULL ORDER BY T4.Height DESC LIMIT 1",


###Example 3 -- Question: State 10 emails of UK Sales Rep who have the lowest credit limit?
###Example 3 -- Evidence: UK is a country; Sales Rep is a job title;
###Example 3 -- Gold SQL: SELECT DISTINCT T2.email FROM customers AS T1 INNER JOIN employees AS T2 ON T1.salesRepEmployeeNumber = T2.employeeNumber WHERE T2.jobTitle = 'Sales Rep' AND T1.country = 'UK' ORDER BY T1.creditLimit LIMIT 10

###Example 4 -- Question: How many Sales Rep who are working in Tokyo? List out email and full name of those employees.
###Example 4 -- Evidence: Sales Rep is a job title; Tokyo is a city; full name = firstName+lastName;
###Example 4 -- Gold SQL: SELECT T1.firstName, T1.lastName, T1.email FROM employees AS T1 INNER JOIN offices AS T2 ON T1.officeCode = T2.officeCode WHERE T2.city = 'Tokyo' AND T1.jobTitle = 'Sales Rep'

###Example 5 -- Question: Among the schools whose donators are teachers, what is the percentage of schools that are in Brooklyn?
###Example 5 -- Evidence: donors are teachers refers to is_teacher_acct = ’t’; Brooklyn is school_city; percentage = Divide(Count(school_city-’Brooklyn’),Count(school_city))*100
###Example 5 -- Gold SQL: SELECT CAST(SUM(CASE WHEN T1.school_city LIKE ’Brooklyn’ THEN 1 ELSE 0 END) AS REAL) * 100 / COUNT(T1.teacher_acctid) FROM projects AS T1 INNER JOIN donations AS T2 ON T1.projectid = T2.projectid WHERE T2.is_teacher_acct = ’t’
"""

Spider_EXAMPLES = """
###Example 1 -- Question: Find the id, forename and number of races of all drivers who have at least participated in two races?
###Example 1 -- GOLD_SQL: SELECT T1.driverid ,  T1.forename ,  count(*) FROM drivers AS T1 JOIN results AS T2 ON T1.driverid = T2.driverid JOIN races AS T3 ON T2.raceid = T3.raceid GROUP BY T1.driverid HAVING count(*)  >=  2

###Example 2 -- Question: Find the names of the campus which has more faculties in 2002 than every campus in Orange county.
###Example 2 -- GOLD_SQL: SELECT T1.campus FROM campuses AS T1 JOIN faculty AS T2 ON T1.id  =  T2.campus WHERE T2.year  =  2002 AND faculty  >  (SELECT max(faculty) FROM campuses AS T1 JOIN faculty AS T2 ON T1.id  =  T2.campus WHERE T2.year  =  2002 AND T1.county  =  "Orange")

###Example 3 -- Question: What are the lot details of lots associated with transactions whose share count is bigger than 100 and whose type code is "PUR"?
###Example 3 -- Gold SQL: SELECT T1.lot_details FROM LOTS AS T1 JOIN TRANSACTIONS_LOTS AS T2 ON  T1.lot_id  =  T2.transaction_id JOIN TRANSACTIONS AS T3 ON T2.transaction_id  =  T3.transaction_id WHERE T3.share_count  >  100 AND T3.transaction_type_code  =  "PUR"

###Example 4 -- Question: Which engineers have never visited to maintain the assets? List the engineer first name and last name.
###Example 4 -- Gold SQL: SELECT first_name ,  last_name FROM Maintenance_Engineers WHERE engineer_id NOT IN (SELECT engineer_id FROM Engineer_Visits)

###Example 5 -- Question: What is the title of the course that is a prerequisite for Mobile Computing?
###Example 5 -- Gold SQL: SELECT title FROM course WHERE course_id IN (SELECT T1.prereq_id FROM prereq AS T1 JOIN course AS T2 ON T1.course_id  =  T2.course_id WHERE T2.title  =  'Mobile Computing')
"""