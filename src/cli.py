# src/cli.py
import pandas as pd
from config import DATA_FILE, OUTPUT_FILE, REQUIRED_COLUMNS

def canonical_text(row):
    """
    Create a text-based version of each CPI record.
    """
    return (
        f"CPI_RECORD | date: {row['Month'].strftime('%Y-%m-%d')} | "
        f"country: {row['Country']} | province: {row['Province']} | "
        f"city: {row['City']} | item: {row['Item']} | unit: {row['Unit']} | "
        f"price: {row['Price']}"
    )

def main():
    # Step 1: Load the dataset
    df = pd.read_excel(DATA_FILE)
    print(f"✅ Loaded dataset with {len(df)} rows")

    # Step 2: Check columns
    print("\nColumns found:", list(df.columns))
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        print(f"⚠️ Missing columns: {missing}")
        return
    else:
        print("✅ All expected columns are present")

    # Step 3: Create textual representation for each row
    df['text_record'] = df.apply(canonical_text, axis=1)

    # Step 4: Show preview
    print("\n🔹 Sample Text Records:")
    for i in range(3):
        print(df.loc[i, 'text_record'])

    # Step 5: (Optional) Save to file
    df[['text_record']].to_parquet(OUTPUT_FILE, index=False)
    print(f"\n💾 Text records saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
