# Creating a guessing number with "while loops"
# John Doan
# May 2023

import random

number1 = random.randint(0,100)

print("Games Rules")
print("User picks an number between 1 and 200")

guesses = [0]

#while loop
while True:
    guess = int(input("Enter a number between 1 to 200: "))
    if guess < 1 or guess > 200:
        print('Invalid number')
        continue
    break

#Compare user's answer against the set 
while True:
    guess = int(input("Enter a number between 1 to 200: "))

    if guess < 1 or guess > 200:
        print("You guessed wrong, try again")
        continue
    if guess == num:
        print('Good job after you have guessed {len(guesses)}')
        break

    guesses.append(guess)
      if guesses[-2]:
        
            