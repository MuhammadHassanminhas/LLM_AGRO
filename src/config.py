# src/config.py
from pathlib import Path

# Base project path
ROOT = Path(__file__).resolve().parents[1]

# Path to your dataset
DATA_FILE = ROOT / "Data" / "Updated_CPI_with_Province_and_Country.xlsx"

# Optional: output file for text records
OUTPUT_FILE = ROOT / "Data" / "CPI_Text_Records.parquet"

# Expected columns in your dataset
REQUIRED_COLUMNS = [
    "City", "Item", "Unit", "Month", "Price", "Province", "Country"
]
