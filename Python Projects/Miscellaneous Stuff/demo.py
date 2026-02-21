import pandas as pd
import matplotlib.pyplot as plt
import calendar
from datetime import datetime

# Load PTO Data from CSV (Format: Employee, PTO_Date)
file_path = "employee_pto.csv"
df = pd.read_csv(file_path, parse_dates=["PTO_Date"])

# Extract year and month from the data
df["Year"] = df["PTO_Date"].dt.year
df["Month"] = df["PTO_Date"].dt.month
df["Day"] = df["PTO_Date"].dt.day

# Get unique years and months
years = df["Year"].unique()
months = df["Month"].unique()

# Create a calendar heatmap for PTO
for year in years:
    for month in months:
        # Filter data for the specific month and year
        pto_days = df[(df["Year"] == year) & (df["Month"] == month)]["Day"].tolist()

        # Generate a blank calendar
        _, ax = plt.subplots(figsize=(8, 6))
        cal = calendar.monthcalendar(year, month)
        ax.set_xticks(range(7))
        ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        ax.set_yticks(range(len(cal)))
        ax.set_yticklabels(["Week " + str(i + 1) for i in range(len(cal))])

        # Highlight PTO days
        for week_idx, week in enumerate(cal):
            for day_idx, day in enumerate(week):
                if day == 0:
                    continue  # Skip empty days
                color = "red" if day in pto_days else "lightgray"
                ax.text(day_idx, week_idx, str(day), ha="center", va="center", fontsize=12,
                        bbox=dict(facecolor=color, edgecolor="black", boxstyle="round,pad=0.3"))

        # Customize plot
        plt.title(f"PTO Calendar - {calendar.month_name[month]} {year}", fontsize=14)
        plt.gca().invert_yaxis()  # Align top to bottom
        plt.axis("off")
        plt.show()