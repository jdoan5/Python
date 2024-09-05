def display_menu():
    print("\n--- Welcome to the Python Prompt Application ---")
    print("Please choose one of the following options:")
    print("1. Say Hello")
    print("2. Perform a Calculation")
    print("3. Exit the Application")

def say_hello():
    name = input("Enter your name: ")
    print(f"Hello, {name}! Welcome to the application.")

def perform_calculation():
    try:
        num1 = float(input("Enter the first number: "))
        num2 = float(input("Enter the second number: "))
        operation = input("Enter the operation (+, -, *, /): ")

        if operation == '+':
            result = num1 + num2
        elif operation == '-':
            result = num1 - num2
        elif operation == '*':
            result = num1 * num2
        elif operation == '/':
            if num2 != 0:
                result = num1 / num2
            else:
                print("Error: Cannot divide by zero.")
                return
        else:
            print("Invalid operation. Please try again.")
            return

        print(f"The result of {num1} {operation} {num2} is: {result}")
    except ValueError:
        print("Invalid input. Please enter numeric values.")

def main():
    while True:
        display_menu()
        choice = input("Enter your choice (1/2/3): ")

        if choice == '1':
            say_hello()
        elif choice == '2':
            perform_calculation()
        elif choice == '3':
            print("Exiting the application. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
