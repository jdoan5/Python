import pandas as pd

INPUT_FILE = "Popular_Baby_Names.csv"
OUTPUT_FILE = "baby_names_trend.csv"

def main():
    # 1. Load raw data
    df = pd.read_csv(INPUT_FILE)

    # Optional: rename columns to simpler names
    df = df.rename(columns={
        "Year of Birth": "Year",
        "Child's First Name": "FirstName"
    })

    # 2. Combine duplicate rows (same year/gender/ethnicity/name)
    df_agg = (
        df.groupby(["Year", "Gender", "Ethnicity", "FirstName"], as_index=False)
          .agg({
              "Count": "sum",   # total babies with that name
              "Rank": "min"     # best (lowest) rank
          })
    )

    # 3. Compute trend vs previous year for each name (within gender+ethnicity)
    df_agg = df_agg.sort_values(["Gender", "Ethnicity", "FirstName", "Year"])
    df_agg["Prev_Count"] = (
        df_agg
        .groupby(["Gender", "Ethnicity", "FirstName"])["Count"]
        .shift(1)
    )
    df_agg["Change"] = df_agg["Count"] - df_agg["Prev_Count"]

    def label_trend(x):
        if pd.isna(x):
            return "New"
        if x > 0:
            return "Rising"
        if x < 0:
            return "Falling"
        return "Stable"

    df_agg["Trend"] = df_agg["Change"].apply(label_trend)

    # 4. Save clean dataset for Looker Studio
    df_agg.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {OUTPUT_FILE} with {len(df_agg)} rows.")

if __name__ == "__main__":
    main()