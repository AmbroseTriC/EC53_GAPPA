#!/usr/bin/env python3
"""Compute distributional incidence of Canada's federal fuel charge (GGPPA Part I)
and the Canada Carbon Rebate (CCR) for selected provinces (ON, MB, SK, AB), 2019-2025.

Inputs (expected in same folder as this script, or edit paths):
- 11100223.xlsx : Survey of Household Spending (SHS) expenditures by income quintile and province.
- 18100001.xlsx : Annual average gasoline and household heating fuel prices by province.
- Tax Rate Items.xlsx : Fuel charge rates by fuel type and policy year (2019-2022 in file).
Outputs:
- ggppa_progressivity_results.csv
- ggppa_kakwani_gross_tax.csv

Assumptions are documented inline; see accompanying PDF report for discussion.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

PROVINCES = ['Ontario','Manitoba','Saskatchewan','Alberta']
QUINTILES = ['Lowest quintile','Second quintile','Third quintile','Fourth quintile','Highest quintile']
YEARS = list(range(2019, 2026))

# Carbon price schedule for the federal fuel charge policy years (start year).
# 2025 set to 0 (fuel charge abolished).
CARBON_PRICE = {2019:20, 2020:30, 2021:40, 2022:50, 2023:65, 2024:80, 2025:0}

# CCR 2024-25 amounts for a family of four, by province (Finance Canada backgrounder).
CCR_2024_FAMILY4 = {'Alberta':1800, 'Ontario':1120, 'Manitoba':1200, 'Saskatchewan':1504}

# CER (2015) - share of homes with natural gas as primary heating, and avg annual natural gas use (GJ) among gas homes.
CER_SHARE_NG = {'Ontario':0.67,'Manitoba':0.51,'Saskatchewan':0.72,'Alberta':0.79}
CER_AVG_GJ_IF_NG = {'Ontario':90.8,'Manitoba':81.7,'Saskatchewan':88.7,'Alberta':100.3}

# Conversion (assumption): 1 GJ ~= 26.3 m3 of natural gas (typical higher heating value).
M3_PER_GJ = 26.3

def fill_series(obs_years, obs_vals, years=YEARS, default=0.0):
    """Linear interpolate internal missing and extrapolate 2024-2025 using CAGR from 2021->2023."""
    s = pd.Series(index=years, dtype=float)
    for y, v in zip(obs_years, obs_vals):
        s.loc[y] = v
    if s.dropna().empty:
        return pd.Series([default]*len(years), index=years, dtype=float)
    s = s.interpolate(method='linear')
    obs = s.dropna()
    g = 0.0
    base_year = obs.index.max()
    base_val = obs.loc[base_year]
    if 2021 in obs.index and 2023 in obs.index and obs.loc[2021] > 0 and obs.loc[2023] > 0:
        g = (obs.loc[2023]/obs.loc[2021])**(1/2) - 1
        base_year, base_val = 2023, obs.loc[2023]
    else:
        if len(obs) >= 2:
            y2, y1 = obs.index[-1], obs.index[-2]
            if obs.loc[y1] > 0 and obs.loc[y2] > 0:
                g = (obs.loc[y2]/obs.loc[y1])**(1/(y2-y1)) - 1
                base_year, base_val = y2, obs.loc[y2]
    for y in [2024, 2025]:
        if pd.isna(s.loc[y]) and y > base_year:
            steps = y - base_year
            s.loc[y] = base_val*((1+g)**steps)
    return s.fillna(method='ffill').fillna(default)

def gini_from_group(means, weights):
    total = np.sum(means*weights)
    if total <= 0:
        return np.nan
    shares = means*weights/total
    cum_pop = np.cumsum(weights)
    cum_share = np.cumsum(shares)
    X = np.concatenate([[0], cum_pop])
    Y = np.concatenate([[0], cum_share])
    area = np.trapz(Y, X)
    return 1 - 2*area

def concentration_from_group(tax_means, weights):
    total_tax = np.sum(tax_means*weights)
    if total_tax == 0:
        return np.nan
    shares = tax_means*weights/total_tax
    cum_pop = np.cumsum(weights)
    cum_share = np.cumsum(shares)
    X = np.concatenate([[0], cum_pop])
    Y = np.concatenate([[0], cum_share])
    area = np.trapz(Y, X)
    return 1 - 2*area

def main():
    # -------- Expenditures (SHS) --------
    df_exp = pd.read_excel('11100223.xlsx', sheet_name='11100223')
    df_exp = df_exp[df_exp['GEO'].isin(PROVINCES)]
    df_exp = df_exp[df_exp['REF_DATE'].between(2019, 2023)]
    df_exp = df_exp[df_exp['Before-tax household income quintile'].isin(QUINTILES + ['All quintiles'])]
    use_cats = ['Gas and other fuels (all vehicles and tools)',
                'Natural gas for principal accommodation',
                'Other fuel for principal accommodation',
                'Total expenditure']
    df_exp = df_exp[df_exp['Household expenditures, summary-level categories'].isin(use_cats)]
    df_exp = df_exp[['GEO','REF_DATE','Before-tax household income quintile',
                     'Household expenditures, summary-level categories','VALUE']].copy()
    df_exp.columns = ['Province','Year','Quintile','Category','Expenditure']
    exp_wide = df_exp.pivot_table(index=['Province','Year','Quintile'], columns='Category', values='Expenditure', aggfunc='first').reset_index()
    exp_wide.columns.name = None

    grid = pd.MultiIndex.from_product([PROVINCES, YEARS, exp_wide['Quintile'].unique()],
                                      names=['Province','Year','Quintile']).to_frame(index=False)
    exp_full = grid.merge(exp_wide, on=['Province','Year','Quintile'], how='left').sort_values(['Province','Quintile','Year'])

    filled_rows = []
    for (prov, quint), g in exp_full.groupby(['Province','Quintile']):
        filled = pd.DataFrame({'Year':YEARS})
        filled['Province'] = prov
        filled['Quintile'] = quint
        for cat in use_cats:
            obs = g.dropna(subset=[cat])
            filled[cat] = fill_series(obs['Year'].tolist(), obs[cat].tolist()).values
        filled_rows.append(filled)
    exp_filled = pd.concat(filled_rows, ignore_index=True)

    # -------- Prices --------
    df_price = pd.read_excel('18100001.xlsx', sheet_name='18100001')
    df_price = df_price[df_price['GEO'].isin(PROVINCES)]
    df_price = df_price[df_price['Type of fuel'].isin(['Regular unleaded gasoline at self service filling stations','Household heating fuel'])]
    df_price['year'] = df_price['REF_DATE'].str[:4].astype(int)
    annual = df_price.groupby(['GEO','year','Type of fuel'])['VALUE'].mean().reset_index()
    annual.columns = ['Province','Year','Fuel','price_cents_per_litre']
    price_pivot = annual.pivot_table(index=['Province','Year'], columns='Fuel', values='price_cents_per_litre').reset_index()
    price_pivot.columns.name = None
    price_pivot['gasoline_price'] = price_pivot['Regular unleaded gasoline at self service filling stations']/100
    price_pivot['heating_oil_price'] = price_pivot['Household heating fuel']/100
    price_pivot = price_pivot[['Province','Year','gasoline_price','heating_oil_price']]

    # -------- Fuel charge rates --------
    rates = pd.read_excel('Tax Rate Items.xlsx')
    for col in ['Period_Start','Period_End','Version_Date']:
        rates[col] = pd.to_datetime(rates[col])
    fuels = ['Gasoline','Marketable natural gas','Light fuel oil']
    rates = rates[(rates['Province'].isin(PROVINCES)) & (rates['Fuel_Type'].isin(fuels))]
    rates['Year'] = rates['Period_Start'].dt.year
    base = (rates[rates['Year'].between(2019, 2022)]
            .sort_values('Version_Date')
            .groupby(['Province','Fuel_Type','Year'], as_index=False)
            .tail(1))[['Province','Fuel_Type','Year','Rate']]

    # Extend using proportional scaling for 2023-2024, and 0 for 2025.
    rows = []
    for prov in PROVINCES:
        for fuel in fuels:
            base2022 = base[(base['Province']==prov)&(base['Fuel_Type']==fuel)&(base['Year']==2022)]
            if base2022.empty:
                continue
            r2022 = float(base2022['Rate'].iloc[0])
            for y in YEARS:
                if y <= 2022:
                    r = base[(base['Province']==prov)&(base['Fuel_Type']==fuel)&(base['Year']==y)]
                    rate = float(r['Rate'].iloc[0]) if not r.empty else np.nan
                elif y == 2023:
                    rate = r2022*(CARBON_PRICE[2023]/CARBON_PRICE[2022])
                elif y == 2024:
                    rate = r2022*(CARBON_PRICE[2024]/CARBON_PRICE[2022])
                else:
                    rate = 0.0
                if prov=='Alberta' and y==2019:
                    rate = 0.0
                rows.append({'Province':prov,'Year':y,'Fuel_Type':fuel,'Rate':rate})
    rates_ext = pd.DataFrame(rows)
    rate_gas = rates_ext[rates_ext['Fuel_Type']=='Gasoline'][['Province','Year','Rate']].rename(columns={'Rate':'rate_gasoline'})
    rate_ng = rates_ext[rates_ext['Fuel_Type']=='Marketable natural gas'][['Province','Year','Rate']].rename(columns={'Rate':'rate_natural_gas'})
    rate_oil = rates_ext[rates_ext['Fuel_Type']=='Light fuel oil'][['Province','Year','Rate']].rename(columns={'Rate':'rate_light_fuel_oil'})

    # -------- Combine and compute incidence --------
    df = exp_filled.merge(price_pivot, on=['Province','Year'], how='left')
    df = df.merge(rate_gas, on=['Province','Year'], how='left').merge(rate_ng, on=['Province','Year'], how='left').merge(rate_oil, on=['Province','Year'], how='left')

    # Ratio to "All quintiles" for natural gas allocation
    base_ng_exp = df[df['Quintile']=='All quintiles'][['Province','Year','Natural gas for principal accommodation']].rename(columns={'Natural gas for principal accommodation':'ng_exp_all'})
    df = df.merge(base_ng_exp, on=['Province','Year'], how='left')
    df['ng_ratio'] = np.where(df['ng_exp_all']>0, df['Natural gas for principal accommodation']/df['ng_exp_all'], 0.0)

    # Gasoline
    df['litres_gasoline'] = np.where(df['gasoline_price']>0, df['Gas and other fuels (all vehicles and tools)']/df['gasoline_price'], np.nan)
    df['tax_gasoline'] = df['litres_gasoline']*df['rate_gasoline']

    # Heating oil proxy
    df['litres_heating_oil'] = np.where(df['heating_oil_price']>0, df['Other fuel for principal accommodation']/df['heating_oil_price'], np.nan)
    df['tax_heating_oil'] = df['litres_heating_oil']*df['rate_light_fuel_oil']
    df.loc[df['heating_oil_price'].isna(), 'tax_heating_oil'] = 0.0

    # Natural gas (consumption-based)
    uncond_m3 = {p: CER_SHARE_NG[p]*CER_AVG_GJ_IF_NG[p]*M3_PER_GJ for p in PROVINCES}
    df['m3_natural_gas'] = df.apply(lambda r: uncond_m3.get(r['Province'],0.0)*r['ng_ratio'], axis=1)
    df['tax_natural_gas'] = df['m3_natural_gas']*df['rate_natural_gas']

    df['gross_tax'] = df[['tax_gasoline','tax_heating_oil','tax_natural_gas']].sum(axis=1)
    df['gross_etr'] = np.where(df['Total expenditure']>0, df['gross_tax']/df['Total expenditure'], np.nan)

    # -------- CCR (household types) --------
    base_adult_2024 = {p: CCR_2024_FAMILY4[p]/2 for p in CCR_2024_FAMILY4}
    def ccr_amount(prov, year, household_type='single'):
        if year==2025:
            return 0.0
        if prov=='Alberta' and year==2019:
            return 0.0
        b = base_adult_2024.get(prov, np.nan)
        if pd.isna(b):
            return np.nan
        if household_type=='single':
            amt_2024 = b
        elif household_type=='couple':
            amt_2024 = 1.5*b
        elif household_type=='family4':
            amt_2024 = 2.0*b
        else:
            raise ValueError
        return amt_2024*(CARBON_PRICE[year]/CARBON_PRICE[2024])

    for ht in ['single','couple','family4']:
        df[f'ccr_{ht}'] = df.apply(lambda r: ccr_amount(r['Province'], r['Year'], ht), axis=1)
        df[f'net_tax_{ht}'] = df['gross_tax'] - df[f'ccr_{ht}']
        df[f'net_etr_{ht}'] = np.where(df['Total expenditure']>0, df[f'net_tax_{ht}']/df['Total expenditure'], np.nan)

    # Keep quintiles only (exclude All quintiles) for distributional results
    out = df[df['Quintile'].isin(QUINTILES)].copy()
    out.to_csv('ggppa_progressivity_results.csv', index=False)

    # Kakwani (gross only)
    weights = np.array([0.2]*5)
    kak_rows = []
    out['Quintile'] = pd.Categorical(out['Quintile'], categories=QUINTILES, ordered=True)
    for (prov, year), g in out.groupby(['Province','Year']):
        g = g.sort_values('Quintile')
        y = g['Total expenditure'].values
        t = g['gross_tax'].values
        G = gini_from_group(y, weights)
        C = concentration_from_group(t, weights)
        kak_rows.append({'Province':prov,'Year':year,'Gini_expenditure':G,'Concentration_gross_tax':C,'Kakwani_gross_tax':C-G})
    pd.DataFrame(kak_rows).to_csv('ggppa_kakwani_gross_tax.csv', index=False)

if __name__ == '__main__':
    main()
