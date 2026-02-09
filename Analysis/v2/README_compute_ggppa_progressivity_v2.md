# GGPPA Fuel Charge Progressivity (v2)

## What this package produces
Running `compute_ggppa_progressivity_v2.py` generates three files:

1. **`ggppa_progressivity_results_v2.csv`**  
   Province × income quintile × policy year (2019–2025) incidence measures:
   - **direct_tax**: modeled direct household fuel-charge payments (gasoline + light fuel oil + natural gas)
   - **gross_cost**: direct tax plus *indirect / non-direct* wedge calibrated to Finance Canada averages
   - **net_*_(base|rural)**: net cost after subtracting CCR/CAI schedules for representative household types  
   - **etr_***: effective tax rates relative to total expenditure

2. **`ggppa_kakwani_indices_v2.csv`**  
   Kakwani indices for selected measures (direct, gross, net-single), using total expenditure as the ranking variable.

3. **`ggppa_finance_crosswalk_v2.csv`**  
   Diagnostics showing the calibration that anchors the indirect wedge to Finance Canada “average cost impact per household.”

## Inputs expected in the same folder
- `11100223.xlsx` — Statistics Canada Table 11-10-0223-01 extract (household spending by quintile)
- `18100001.xlsx` — Statistics Canada Table 18-10-0001-01 extract (retail gasoline & heating fuel prices)
- `36100101.xlsx` — Statistics Canada Table 36-10-0101-01 extract (household counts by quintile)
- `Tax Rate Items.xlsx` — provided extract of federal fuel-charge rates (2019–2022)

## How to run
```bash
python compute_ggppa_progressivity_v2.py
```

## Key modeling choices (important)
- **Policy year alignment:** `Year = 2019` corresponds to the April 2019–March 2020 fuel-charge year.  
- **Missing expenditure years:** the provided spending extract contains 2019, 2021, 2023; 2020 and 2022 are interpolated linearly; 2024–2025 are held at 2023 values.  
- **Vehicle fuel aggregation:** the spending category “Gas and other fuels (all vehicles and tools)” is treated as gasoline for quantity inference.
- **Natural gas conversion:** natural-gas expenditures include fixed charges; physical gigajoules are approximated as `GJ = φ_r × dollars` using province-specific reduced-form factors.
- **Indirect effects:** modeled as a multiplicative wedge `gross = (1 + κ_rt) × direct`, calibrated to Finance Canada average household cost impacts for 2021 and 2024 and interpolated in the statutory carbon price.
- **CCR schedules:** implemented as annual base amounts (first adult, second adult, each child) plus a rural top-up of 10% (through 2023) and 20% (2024). Alberta’s 2020 CAI uses the 15‑month adjustment stated by Finance Canada.
- **2025 abolition:** fuel charge (and CCR) are set to zero in 2025.

## Outputs dictionary (selected fields)
- `direct_tax` — direct fuel-charge burden (dollars/household/year)
- `gross_cost` — direct + indirect wedge (dollars/household/year)
- `CCR_*_base`, `CCR_*_rural` — rebate schedules (dollars/household/year)
- `net_*_base`, `net_*_rural` — net cost (gross − rebate)
- `etr_*` — effective tax rate (cost divided by total expenditure)

