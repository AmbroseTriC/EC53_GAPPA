# Expenditure Cleaning (2019–2025)

This cleaning script extracts selected household expenditure categories from
Statistics Canada table 11-10-0223 (input file
`Expenditure-11100223/11100223.xlsx`), limited to specific regions and income
quintiles.

## Output File
- `Cleaning/Expenditure/expenditure_cleaned.xlsx`

## Coverage
- Years: 2019–2025 (inclusive)
- Regions: Canada, Ontario, Manitoba, Saskatchewan, Alberta, Yukon
- Categories:
  - Natural gas for principal accommodation
  - Gas and other fuels (all vehicles and tools)
- Income groups:
  - 1st, 2nd, 3rd, 4th, 5th quintiles
  - All quintiles

## Processing Summary
- Filters the input dataset to the above regions, years, categories, and
  quintiles.
- Maps quintile labels from the source to `1st`–`5th` plus `All quintiles`.
- Preserves units (no unit conversion).

## Columns
- `REF_DATE`: Year
- `GEO`: Region
- `Income_Quintile`: Mapped quintile label
- `Household expenditures, summary-level categories`: Expenditure category
- `UOM`: Unit of measurement
- `VALUE`: Expenditure value

## How It’s Produced
Run:

```bash
python Cleaning/Expenditure/expenditure_cleaning.py
```
