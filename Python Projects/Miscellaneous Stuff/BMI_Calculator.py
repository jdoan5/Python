# A program to calculate the Body Mass Index (BMI)


def calculate_bmi(weight_kg: float, height_m: float) -> float:
    """BMI = weight in kilograms divided by height in metres squared."""
    if height_m <= 0:
        raise ValueError("Height must be greater than zero.")
    return weight_kg / (height_m ** 2)


def classify_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Normal weight"
    if bmi < 30:
        return "Overweight"
    return "Obese"


def main() -> None:
    weight = float(input("Enter your weight in kilograms: "))
    height = float(input("Enter your height in meters: "))
    bmi = calculate_bmi(weight, height)
    print(f"BMI: {bmi:.1f} ({classify_bmi(bmi)})")


if __name__ == "__main__":
    main()
