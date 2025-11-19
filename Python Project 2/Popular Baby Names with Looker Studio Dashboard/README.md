# NYC Popular Baby Names – Trend Dashboard

This project explores the **NYC Popular Baby Names** dataset from [https://catalog.data.gov/dataset/popular-baby-names/resource/02e8f55e-2157-4cb2-961a-2aabb75cbc8b]  and turns it into an interactive **trend dashboard** built with **Python (PyCharm)** and **Google Looker Studio**.

The goal is to show:

- Which baby names are most popular
- How popularity changes over time
- Which names are **New**, **Rising**, **Falling**, or **Stable**

---

## 1. Project Overview

**Pipeline**

1. **Data source**  
   - Public dataset: *NYC Popular Baby Names* (CSV from https://catalog.data.gov/dataset/popular-baby-names/resource/02e8f55e-2157-4cb2-961a-2aabb75cbc8b).

2. **Data preparation (Python)**  
   - Script: `prepare_baby_names.py`
   - Tasks:
     - Load the raw `Popular_Baby_Names.csv`
     - Clean and rename columns
     - Aggregate duplicate rows (by Year, Gender, Ethnicity, FirstName)
     - Calculate **year-over-year change** in counts for each name
     - Create a **Trend** label:
       - `New` – first year the name appears
       - `Rising` – more babies than previous year
       - `Falling` – fewer babies than previous year
       - `Stable` – same number as previous year
     - Save the enriched data as `baby_names_trend.csv`

3. **Visualization (Looker Studio)**  
   - Connect `baby_names_trend.csv` (via Google Sheets)
   - Build an interactive dashboard with:
     - KPI cards (total babies, distinct names)
     - Table of names with Year, Gender, Ethnicity, Trend, Count, Rank
     - Bar chart of top names
     - Pie chart of Trend distribution
     - Filters for Year, Gender, Ethnicity, and Trend

---

## 2. Tech Stack

- **Python 3.x**
  - `pandas`
- **PyCharm** (for development, optional)
- **Google Sheets**
- **Google Looker Studio** (dashboard)

---

## 3. Repository Structure

```text
.
├── Popular_Baby_Names.csv      # Raw dataset from data.gov
├── prepare_baby_names.py       # Data cleaning & feature engineering script
├── baby_names_trend.csv        # Output file used by Looker Studio (generated)
└── README.md                   # This file