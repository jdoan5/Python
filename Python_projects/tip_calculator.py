# How to calculate tip and percentage for each person
# Author: John Doan
# June 2023

# User input
print("Tip Calculator")
bill = float(input("Please enter the total bill: \n$"))
total_people = int(input("How many people to split the bill: \n"))
percentage_tip = float(input("What percentage tip would you like to give? 10%, 12%, or 15%\n"))

# Calculate
tip_amount = percentage_tip / 100
total_tip_amount = tip_amount * bill
total_bill = bill + total_tip_amount
bill_per_person = total_bill / total_people
final_result = bill_per_person

# Print result
print(f"The bill costs ${bill} plus {percentage_tip}% tip gratuity and there are {total_people} people.")
print(f"Therefore, each person should pay: ${final_result}")