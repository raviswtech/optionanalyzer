import streamlit as st
import pandas as pd
import io
import re
from datetime import date, datetime, timedelta
from py_vollib.black_scholes_merton.greeks import analytical
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# --- DEFAULT CSV DATA ---
DEFAULT_CSV_DATA = """CALLS,,PUTS
,OI,CHNG IN OI,VOLUME,IV,LTP,CHNG,BID QTY,BID,ASK,ASK QTY,STRIKE,BID QTY,BID,ASK,ASK QTY,CHNG,LTP,IV,VOLUME,CHNG IN OI,OI,
,102,-,1,15.0,"2,466.75",-63.25,750,"2,522.50","2,558.00",150,"22,250.00","17,72,550",0.65,0.70,"15,85,200",-0.30,30.29,30.29,"2,18,116","30,004","54,625",
,51,-,4,17.5,"2,556.25",17.55,150,"2,482.30","2,505.95",525,"22,300.00","9,46,950",0.65,0.70,"7,95,075",-0.30,29.72,29.72,"87,761","13,796","35,214",
,7,-,-,18.0,-,-,750,"2,399.20","2,507.80",750,"22,350.00","1,67,325",0.70,0.75,"64,800",-0.25,29.35,29.35,"11,352",231,"6,811",
,"1,15,564","61,391","9,48,859",16.34,203.60,-25.85,450,203.50,203.90,"1,350","24,800.00","5,550",210.00,210.25,975,33.55,16.42,16.42,"11,76,610","38,954","85,475",
,"9,798","2,335","37,508",16.51,386.15,-37.85,225,386.20,387.10,75,"24,500.00","1,125",93.10,93.30,675,22.35,16.57,16.57,"5,63,128","18,450","56,173",
,"1,26,784","52,688","7,40,357",16.33,120.10,-23.30,"1,125",120.20,120.40,150,"25,000.00",300,326.25,326.95,150,37.95,16.41,16.41,"1,42,681","3,092","30,611"
"""

HOW_TO_GET_DATA_NSE = """
**Steps to get Option Chain data from NSE Website:**
1.  **Visit NSE India Website:** Go to `https://www.nseindia.com/option-chain`.
2.  **Navigate to Option Chain:** Market Data -> Derivatives -> Option Chain (Equity/Index).
3.  **Select Instrument & Expiry:** Choose Index/Stock and the correct Expiry Date.
4.  **Export Data:** Look for a "Download CSV" button and download the file.
5.  **Copy Data:** Open the CSV, select all (`Ctrl+A`), copy (`Ctrl+C`).
6.  **Paste in Tool:** Select "Paste CSV" option here and paste the data.
7.  **Analyze:** Fill other inputs and click "🚀 Analyze...".
**Notes:** Copy entire relevant data range including NSE headers. For "Upload CSV", use the direct NSE download.
"""

# Session state initialization 
if 'custom_lot_size_active' not in st.session_state:
    st.session_state.custom_lot_size_active = False
if 'custom_lot_size_value' not in st.session_state: # Initialize based on lot_size_input
    st.session_state.custom_lot_size_value = st.session_state.get('lot_size_input', 75)

# --- Helper Functions ---
def clean_number(value_str):
    if isinstance(value_str, (int, float)): return value_str
    if not isinstance(value_str, str): return None
    value_str = value_str.strip()
    if value_str == '-' or value_str == '': return None
    try:
        cleaned_val = value_str.strip('"')
        return float(cleaned_val.replace(',', ''))
    except ValueError:
        try: return float(value_str)
        except ValueError: return None

def parse_option_chain_csv(csv_data_string):
    lines = csv_data_string.strip().splitlines()
    data_rows = []
    if not lines: return pd.DataFrame()
    for i, line_raw in enumerate(lines):
        line = line_raw.strip();
        if not line or i < 2: continue
        columns = re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)
        if len(columns) < 22: continue
        try:
            strike_price = clean_number(columns[11])
            if strike_price is None: continue
            def get_col_val(idx): return columns[idx] if idx < len(columns) else ""
            row_data = {
                'CALLS_OI': clean_number(get_col_val(1)), 'CALLS_CHNG_IN_OI': clean_number(get_col_val(2)),
                'CALLS_VOLUME': clean_number(get_col_val(3)), 'CALLS_IV': clean_number(get_col_val(4)),
                'CALLS_LTP': clean_number(get_col_val(5)), 'CALLS_CHNG': clean_number(get_col_val(6)),
                'CALLS_BID_QTY': clean_number(get_col_val(7)), 'CALLS_BID': clean_number(get_col_val(8)),
                'CALLS_ASK': clean_number(get_col_val(9)), 'CALLS_ASK_QTY': clean_number(get_col_val(10)),
                'STRIKE': strike_price,
                'PUTS_BID_QTY': clean_number(get_col_val(12)), 'PUTS_BID': clean_number(get_col_val(13)),
                'PUTS_ASK': clean_number(get_col_val(14)), 'PUTS_ASK_QTY': clean_number(get_col_val(15)),
                'PUTS_CHNG': clean_number(get_col_val(16)), 'PUTS_LTP': clean_number(get_col_val(17)),
                'PUTS_IV': clean_number(get_col_val(18)), 'PUTS_VOLUME': clean_number(get_col_val(19)),
                'PUTS_CHNG_IN_OI': clean_number(get_col_val(20)), 'PUTS_OI': clean_number(get_col_val(21)),
            }
            data_rows.append(row_data)
        except Exception: continue
    return pd.DataFrame(data_rows)

def calculate_greeks(row, option_type, S, K_val, t, r, q, iv_col_name):
    iv = row.get(iv_col_name)
    condition_nan = pd.isna(iv) or iv <= 0 or pd.isna(S) or pd.isna(K_val) or pd.isna(t) or t <=0 or pd.isna(r) or pd.isna(q)
    if condition_nan: return pd.Series({'Delta': np.nan, 'Gamma': np.nan, 'Theta': np.nan, 'Vega': np.nan})
    flag = 'c' if option_type == 'call' else 'p'
    iv_decimal, r_decimal, q_decimal = iv / 100.0, r / 100.0, q / 100.0
    try:
        delta = analytical.delta(flag, S, K_val, t, r_decimal, iv_decimal, q_decimal)
        gamma = analytical.gamma(flag, S, K_val, t, r_decimal, iv_decimal, q_decimal)
        theta = analytical.theta(flag, S, K_val, t, r_decimal, iv_decimal, q_decimal) / 365.0
        vega = analytical.vega(flag, S, K_val, t, r_decimal, iv_decimal, q_decimal) / 100.0 
    except Exception: return pd.Series({'Delta': np.nan, 'Gamma': np.nan, 'Theta': np.nan, 'Vega': np.nan})
    return pd.Series({'Delta': delta, 'Gamma': gamma, 'Theta': theta, 'Vega': vega})

def analyze_option_data_extended(df, cmp, vix, lot_size, interest_rate, dividend_yield_perc, dte):
    if df.empty: return {"error": "No data to analyze."}, pd.DataFrame()
    analysis = {}
    df_calc = df.copy() 
    analysis['total_call_oi'] = df_calc['CALLS_OI'].fillna(0).sum()
    analysis['total_put_oi'] = df_calc['PUTS_OI'].fillna(0).sum()
    analysis['pcr_oi'] = (analysis['total_put_oi'] / analysis['total_call_oi']) if analysis['total_call_oi'] else 0
    oi_threshold_factor = 1.5 
    valid_call_oi = df_calc['CALLS_OI'].dropna(); valid_call_oi = valid_call_oi[valid_call_oi > 0]
    median_call_oi = valid_call_oi.median() if not valid_call_oi.empty else 0
    valid_put_oi = df_calc['PUTS_OI'].dropna(); valid_put_oi = valid_put_oi[valid_put_oi > 0]
    median_put_oi = valid_put_oi.median() if not valid_put_oi.empty else 0
    call_walls_df = df_calc[df_calc['CALLS_OI'].fillna(0) > median_call_oi * oi_threshold_factor].sort_values(by='CALLS_OI', ascending=False) if median_call_oi > 0 else pd.DataFrame()
    put_walls_df = df_calc[df_calc['PUTS_OI'].fillna(0) > median_put_oi * oi_threshold_factor].sort_values(by='PUTS_OI', ascending=False) if median_put_oi > 0 else pd.DataFrame()
    analysis['max_call_oi_strike'] = call_walls_df['STRIKE'].iloc[0] if not call_walls_df.empty else (df_calc.loc[df_calc['CALLS_OI'].fillna(0).idxmax()]['STRIKE'] if df_calc['CALLS_OI'].fillna(0).sum() > 0 else 0)
    analysis['max_put_oi_strike'] = put_walls_df['STRIKE'].iloc[0] if not put_walls_df.empty else (df_calc.loc[df_calc['PUTS_OI'].fillna(0).idxmax()]['STRIKE'] if df_calc['PUTS_OI'].fillna(0).sum() > 0 else 0)
    analysis['call_walls_detail'] = call_walls_df[['STRIKE', 'CALLS_OI']].head(3)
    analysis['put_walls_detail'] = put_walls_df[['STRIKE', 'PUTS_OI']].head(3)
    df_calc['CALLS_OI_Cash'] = df_calc['CALLS_LTP'].fillna(0) * df_calc['CALLS_OI'].fillna(0) * lot_size
    df_calc['PUTS_OI_Cash'] = df_calc['PUTS_LTP'].fillna(0) * df_calc['PUTS_OI'].fillna(0) * lot_size
    analysis['total_call_oi_cash'] = df_calc['CALLS_OI_Cash'].sum()
    analysis['total_put_oi_cash'] = df_calc['PUTS_OI_Cash'].sum()
    df_calc['CALLS_OI_Contracts'] = df_calc['CALLS_OI'].fillna(0) 
    df_calc['PUTS_OI_Contracts'] = df_calc['PUTS_OI'].fillna(0)
    required_greek_cols_check = ['CALLS_Delta', 'PUTS_Delta', 'CALLS_Gamma', 'PUTS_Gamma'] 
    if not all(col in df_calc.columns for col in required_greek_cols_check):
        analysis['cumulative_greeks_error'] = f"Greek cols missing. Expected e.g. {required_greek_cols_check[0]}. Avail: {df_calc.columns.tolist()}"
        for key in ['total_delta_exposure', 'total_gamma_exposure', 'total_theta_decay_daily', 'total_vega_exposure', 
                    'total_call_gamma_exposure_notional', 'total_put_gamma_exposure_notional', 'net_market_gamma_exposure_notional',
                    'gamma_flip_level', 'call_gamma_wall_strike', 'put_gamma_wall_strike']: analysis[key] = 0
        analysis['gamma_exposure_error'] = "Gamma columns not found for exposure calculation."
    else:
        analysis['total_delta_exposure'] = (df_calc['CALLS_Delta'] * df_calc['CALLS_OI_Contracts'] * lot_size).sum(skipna=True) + (df_calc['PUTS_Delta'] * df_calc['PUTS_OI_Contracts'] * lot_size).sum(skipna=True)
        analysis['total_gamma_exposure'] = (df_calc['CALLS_Gamma'] * df_calc['CALLS_OI_Contracts'] * lot_size).sum(skipna=True) + (df_calc['PUTS_Gamma'] * df_calc['PUTS_OI_Contracts'] * lot_size).sum(skipna=True)
        analysis['total_theta_decay_daily'] = (df_calc['CALLS_Theta'] * df_calc['CALLS_OI_Contracts'] * lot_size).sum(skipna=True) + (df_calc['PUTS_Theta'] * df_calc['PUTS_OI_Contracts'] * lot_size).sum(skipna=True)
        analysis['total_vega_exposure'] = (df_calc['CALLS_Vega'] * df_calc['CALLS_OI_Contracts'] * lot_size).sum(skipna=True) + (df_calc['PUTS_Vega'] * df_calc['PUTS_OI_Contracts'] * lot_size).sum(skipna=True)
        df_calc['CALLS_Gamma_Exposure_Notional'] = df_calc['CALLS_Gamma'].fillna(0) * df_calc['CALLS_OI_Contracts'].fillna(0) * lot_size * cmp 
        df_calc['PUTS_Gamma_Exposure_Notional'] = df_calc['PUTS_Gamma'].fillna(0) * df_calc['PUTS_OI_Contracts'].fillna(0) * lot_size * cmp
        df_calc['Net_Gamma_Exposure_At_Strike'] = df_calc['CALLS_Gamma_Exposure_Notional'] - df_calc['PUTS_Gamma_Exposure_Notional']
        analysis['total_call_gamma_exposure_notional'] = df_calc['CALLS_Gamma_Exposure_Notional'].sum()
        analysis['total_put_gamma_exposure_notional'] = df_calc['PUTS_Gamma_Exposure_Notional'].sum()
        analysis['net_market_gamma_exposure_notional'] = analysis['total_call_gamma_exposure_notional'] - analysis['total_put_gamma_exposure_notional']
        strike_filter_percentage = 0.10 
        min_relevant_strike = cmp * (1 - strike_filter_percentage)
        max_relevant_strike = cmp * (1 + strike_filter_percentage)
        df_relevant_strikes = df_calc[(df_calc['STRIKE'] >= min_relevant_strike) & (df_calc['STRIKE'] <= max_relevant_strike)]
        if not df_relevant_strikes.empty:
            if 'Net_Gamma_Exposure_At_Strike' in df_relevant_strikes.columns and not df_relevant_strikes['Net_Gamma_Exposure_At_Strike'].dropna().empty:
                gamma_flip_strike_iloc = (df_relevant_strikes['Net_Gamma_Exposure_At_Strike'].abs()).idxmin() # idxmin on absolute values
                if not pd.isna(gamma_flip_strike_iloc) and gamma_flip_strike_iloc in df_relevant_strikes.index: 
                    analysis['gamma_flip_level'] = df_relevant_strikes.loc[gamma_flip_strike_iloc]['STRIKE']
                else: # Fallback if idxmin problematic
                    analysis['gamma_flip_level'] = 0
            else: analysis['gamma_flip_level'] = 0 # No valid net gamma exposure in range
            otm_calls_relevant_df = df_relevant_strikes[df_relevant_strikes['STRIKE'] > cmp]
            if not otm_calls_relevant_df.empty and 'CALLS_Gamma_Exposure_Notional' in otm_calls_relevant_df.columns and otm_calls_relevant_df['CALLS_Gamma_Exposure_Notional'].fillna(0).max() > 0:
                analysis['call_gamma_wall_strike'] = otm_calls_relevant_df.loc[otm_calls_relevant_df['CALLS_Gamma_Exposure_Notional'].fillna(0).idxmax()]['STRIKE']
            elif 'CALLS_Gamma_Exposure_Notional' in df_relevant_strikes.columns and df_relevant_strikes['CALLS_Gamma_Exposure_Notional'].fillna(0).max() > 0:
                analysis['call_gamma_wall_strike'] = df_relevant_strikes.loc[df_relevant_strikes['CALLS_Gamma_Exposure_Notional'].fillna(0).idxmax()]['STRIKE']
            else: analysis['call_gamma_wall_strike'] = 0
            otm_puts_relevant_df = df_relevant_strikes[df_relevant_strikes['STRIKE'] < cmp]
            if not otm_puts_relevant_df.empty and 'PUTS_Gamma_Exposure_Notional' in otm_puts_relevant_df.columns and otm_puts_relevant_df['PUTS_Gamma_Exposure_Notional'].fillna(0).max() > 0:
                analysis['put_gamma_wall_strike'] = otm_puts_relevant_df.loc[otm_puts_relevant_df['PUTS_Gamma_Exposure_Notional'].fillna(0).idxmax()]['STRIKE']
            elif 'PUTS_Gamma_Exposure_Notional' in df_relevant_strikes.columns and df_relevant_strikes['PUTS_Gamma_Exposure_Notional'].fillna(0).max() > 0:
                analysis['put_gamma_wall_strike'] = df_relevant_strikes.loc[df_relevant_strikes['PUTS_Gamma_Exposure_Notional'].fillna(0).idxmax()]['STRIKE']
            else: analysis['put_gamma_wall_strike'] = 0
        else: 
            analysis['gamma_flip_level'], analysis['call_gamma_wall_strike'], analysis['put_gamma_wall_strike'] = 0,0,0
            if 'gamma_exposure_error' not in analysis: analysis['gamma_exposure_error'] = "Not enough data in relevant strike range for Gamma Flip/Walls."
    market_view = "Neutral / Indecisive" 
    if analysis['pcr_oi'] > 1.1 and analysis['max_put_oi_strike'] <= cmp: market_view = "Moderately Bullish"
    elif analysis['pcr_oi'] < 0.9 and analysis['max_call_oi_strike'] >= cmp: market_view = "Moderately Bearish"
    atm_call_iv_series = df_calc.iloc[(df_calc['STRIKE'] - cmp).abs().argsort()[:1]]['CALLS_IV'] if not df_calc.empty else pd.Series()
    atm_put_iv_series = df_calc.iloc[(df_calc['STRIKE'] - cmp).abs().argsort()[:1]]['PUTS_IV'] if not df_calc.empty else pd.Series()
    avg_atm_iv_val = np.nan
    if not atm_call_iv_series.empty and not pd.isna(atm_call_iv_series.iloc[0]) and \
       not atm_put_iv_series.empty and not pd.isna(atm_put_iv_series.iloc[0]):
        avg_atm_iv_val = (atm_call_iv_series.iloc[0] + atm_put_iv_series.iloc[0]) / 2.0
    elif vix is not None and vix > 0 : avg_atm_iv_val = vix
    volatility_outlook = "Moderate Volatility Expected"
    if avg_atm_iv_val is not None and not pd.isna(avg_atm_iv_val):
        if avg_atm_iv_val > 20 : volatility_outlook = "High Volatility Expected / Priced In"
        elif avg_atm_iv_val < 12 : volatility_outlook = "Low Volatility Expected / Priced In"
    analysis['market_view'] = market_view
    analysis['volatility_outlook'] = volatility_outlook
    analysis['dte_used'] = dte
    analysis['interest_rate_used'] = interest_rate
    analysis['dividend_yield_used'] = dividend_yield_perc
    analysis['avg_atm_iv'] = avg_atm_iv_val if (avg_atm_iv_val is not None and not pd.isna(avg_atm_iv_val)) else None
    analysis['lot_size_used'] = lot_size
    return analysis, df_calc

def style_option_chain(df, cmp):
    if df.empty: return df
    atm_bg, itm_bg, otm_bg, text_dark, strike_bg, header_bg = '#FFFACD', '#E0F2F7', '#FFEBEE', '#212121', '#E8EAF6', '#3F51B5'
    cmp_strike_rounded = round(cmp / 50) * 50
    def apply_styles(row):
        styles, row_strike = [''] * len(row), row['STRIKE']
        base_bg = atm_bg if row_strike == cmp_strike_rounded else ''
        for i, col_name in enumerate(row.index):
            cell_style, current_bg = f'color: {text_dark};', base_bg
            is_call_col = col_name.startswith('CALLS_') or ('CALLS_' in col_name and any(g in col_name for g in ['Delta', 'Gamma', 'Theta', 'Vega']))
            is_put_col = col_name.startswith('PUTS_') or ('PUTS_' in col_name and any(g in col_name for g in ['Delta', 'Gamma', 'Theta', 'Vega']))
            if not current_bg:
                if is_call_col: current_bg = itm_bg if row_strike < cmp_strike_rounded else (otm_bg if row_strike > cmp_strike_rounded else '')
                elif is_put_col: current_bg = itm_bg if row_strike > cmp_strike_rounded else (otm_bg if row_strike < cmp_strike_rounded else '')
            if current_bg: cell_style += f'background-color: {current_bg};'
            if col_name == 'STRIKE':
                cell_style = f'background-color: {strike_bg}; color: {text_dark}; font-weight: bold;'
                if row_strike == cmp_strike_rounded: cell_style += 'border: 2px solid orange;'
            styles[i] = cell_style
        return styles
    format_dict = {col: '{:,.0f}' for col in df.columns if any(k in col for k in ['OI', 'VOLUME', 'QTY', 'STRIKE', 'Cash', 'Exposure_Notional'])}
    format_dict.update({col: '{:,.2f}' for col in df.columns if any(k in col for k in ['LTP', 'CHNG', 'IV', 'BID', 'ASK']) and 'STRIKE' not in col})
    format_dict.update({col: '{:,.4f}' for col in df.columns if any(g in col for g in ['Delta', 'Gamma', 'Theta', 'Vega'])})
    styler = df.style.apply(apply_styles, axis=1).format(format_dict, na_rep='-')
    return styler.set_table_styles([{'selector': 'th', 'props': [('background-color', header_bg), ('color', 'white'), ('font-weight', 'bold')]}])

def get_strategy_suggestions(market_view, volatility_outlook, avg_atm_iv_val, analysis_res=None):
    suggestions = []
    is_high_iv = "High Volatility" in volatility_outlook or (avg_atm_iv_val is not None and not pd.isna(avg_atm_iv_val) and avg_atm_iv_val > 18)
    is_low_iv = "Low Volatility" in volatility_outlook or (avg_atm_iv_val is not None and not pd.isna(avg_atm_iv_val) and avg_atm_iv_val < 14)
    general_strategies = [
        ("Long Call", "Bullish", "Any (better if low/rising)", "Limited to Premium", "Unlimited", "Directional upside bet."),
        ("Long Put", "Bearish", "Any (better if low/rising)", "Limited to Premium", "Unlimited", "Directional downside bet."),
        ("Bull Call Spread", "Moderately Bullish", "Any", "Limited to Net Debit", "Limited", "Cost-effective bullish view."),
        ("Bear Put Spread", "Moderately Bearish", "Any", "Limited to Net Debit", "Limited", "Cost-effective bearish view."),
        ("Long Straddle", "Neutral - Expecting Volatility", "Low/Rising", "Net Premium", "High if large move", "Profits from large price swing, regardless of direction."),
        ("Long Strangle", "Neutral - Expecting Volatility", "Low/Rising", "Net Premium", "High if very large move", "Cheaper than straddle, needs bigger move."),
        ("Short Put", "Bullish/Neutral", "High/Falling", "Substantial", "Limited to Premium", "Collect premium if stock stays above strike. High IV helps."),
        ("Short Call", "Bearish/Neutral", "High/Falling", "Unlimited", "Limited to Premium", "Collect premium if stock stays below strike. High IV helps."),
    ]
    for name, s_view, s_iv_pref, s_risk, s_reward, s_notes in general_strategies:
        view_match, iv_pref_match = False, False
        if any(term in market_view for term in s_view.split('/')) or (s_view == "Neutral - Expecting Volatility" and "Expecting Volatility" in market_view): view_match = True
        if "Any" in s_iv_pref or ("low/rising" in s_iv_pref.lower() and is_low_iv) or ("high/falling" in s_iv_pref.lower() and is_high_iv): iv_pref_match = True
        if view_match and iv_pref_match:
            suggestions.append({"Category": "General View/IV", "Strategy": name, "Suitable View": s_view, "IV Pref.": s_iv_pref, "Max Risk": s_risk, "Max Reward": s_reward, "Notes & Risks": s_notes})
    
    if analysis_res: # For Gamma/IV specific suggestions if full analysis_res is passed
        g_flip = analysis_res.get('gamma_flip_level', 0)
        cmp_val = st.session_state.cmp_for_analysis 
        net_mkt_gexp_notional = analysis_res.get('net_market_gamma_exposure_notional', 0)

        # Gamma Based
        if g_flip != 0:
            if cmp_val < g_flip and net_mkt_gexp_notional < -100_000_000: 
                 suggestions.append({"Category": "Gamma Exposure", "Strategy": "Consider Long Gamma (e.g., Straddle/Strangle, Directional Buys)", "Condition": f"CMP < G-Flip ({g_flip}), Sig. Negative Net GExp", "Rationale": "Dealers might amplify moves. Long gamma profits from volatility.", "Max Risk": "Premium Paid", "Max Reward": "High"})
            elif cmp_val > g_flip and net_mkt_gexp_notional > 100_000_000:
                 suggestions.append({"Category": "Gamma Exposure", "Strategy": "Consider Short Gamma (e.g., Iron Condor, Spreads)", "Condition": f"CMP > G-Flip ({g_flip}), Sig. Positive Net GExp", "Rationale": "Dealers might dampen moves. Short gamma profits from time decay/low vol. High Risk.", "Max Risk": "Varies", "Max Reward": "Limited"})
        
        # IV Skew Based
        atm_iv_info = analysis_res.get('avg_atm_iv_info', {}) 
        call_atm_iv = atm_iv_info.get('call', np.nan) # Default to NaN if not found
        put_atm_iv = atm_iv_info.get('put', np.nan)

        if not pd.isna(call_atm_iv) and not pd.isna(put_atm_iv):
            if put_atm_iv > call_atm_iv * 1.05: 
                suggestions.append({"Category": "IV Skew", "Strategy": "Consider Bearish or Hedging (e.g. Bear Put Spread)", "Condition": "Put IV > Call IV (Put Skew)", "Rationale": "Market pricing in higher cost of downside protection.", "Max Risk": "Varies", "Max Reward": "Varies"})
            elif call_atm_iv > put_atm_iv * 1.05:
                suggestions.append({"Category": "IV Skew", "Strategy": "Consider Bullish (e.g. Bull Call Spread)", "Condition": "Call IV > Put IV (Call Skew)", "Rationale": "Market pricing in higher cost for upside participation.", "Max Risk": "Varies", "Max Reward": "Varies"})
    return suggestions

# Add this near other helper functions or in the UI section for the new tab
def generate_llm_analysis_prompt(cmp, vix, lot_size, dte, interest_rate, dividend_yield, 
                                 df_atm_strikes, analysis_results):
    prompt_parts = []

    prompt_parts.append("## Market Context & Option Chain Data Analysis Prompt\n")
    prompt_parts.append("Please perform a detailed options analysis based on the following data. Provide insights, potential strategies (with risk considerations), and identify key levels or anomalies.")
    prompt_parts.append("\n**I. Current Market & Contract Parameters:**")
    prompt_parts.append(f"- Underlying Spot Price (CMP): {cmp:,.2f}")
    prompt_parts.append(f"- India VIX: {vix:.2f}%")
    prompt_parts.append(f"- Days to Expiry (DTE): {dte} days")
    prompt_parts.append(f"- Annual Risk-Free Interest Rate: {interest_rate:.2f}%")
    prompt_parts.append(f"- Annual Dividend Yield (q): {dividend_yield:.2f}%")
    prompt_parts.append(f"- Lot Size: {lot_size}")

    if not df_atm_strikes.empty:
        prompt_parts.append("\n**II. Option Chain Data (Strikes around ATM):**")
        # Header for the table-like structure in the prompt
        header = "| Strike | Call OI | Call Chg OI | Call Vol | Call IV | Call LTP | C_Delta | C_Gamma | C_Theta | C_Vega | Put OI  | Put Chg OI  | Put Vol  | Put IV  | Put LTP  | P_Delta | P_Gamma | P_Theta | P_Vega |"
        separator = "|--------|---------|-------------|----------|---------|----------|---------|---------|---------|--------|---------|-------------|----------|---------|----------|---------|---------|---------|--------|"
        prompt_parts.append(header)
        prompt_parts.append(separator)

        for _, row in df_atm_strikes.iterrows():
            # Ensure all keys exist, use .get(key, 'N/A') or similar for safety if some greeks might be missing
            row_str = f"| {row.get('STRIKE', 'N/A'):<6,.0f} " \
                      f"| {row.get('CALLS_OI', 0):<7,.0f} " \
                      f"| {row.get('CALLS_CHNG_IN_OI', 0):<11,.0f} " \
                      f"| {row.get('CALLS_VOLUME', 0):<8,.0f} " \
                      f"| {row.get('CALLS_IV', 0):<7.2f} " \
                      f"| {row.get('CALLS_LTP', 0):<8.2f} " \
                      f"| {row.get('CALLS_Delta', 0):<7.4f} " \
                      f"| {row.get('CALLS_Gamma', 0):<7.4f} " \
                      f"| {row.get('CALLS_Theta', 0):<7.4f} " \
                      f"| {row.get('CALLS_Vega', 0):<6.4f} " \
                      f"| {row.get('PUTS_OI', 0):<7,.0f} " \
                      f"| {row.get('PUTS_CHNG_IN_OI', 0):<11,.0f} " \
                      f"| {row.get('PUTS_VOLUME', 0):<8,.0f} " \
                      f"| {row.get('PUTS_IV', 0):<7.2f} " \
                      f"| {row.get('PUTS_LTP', 0):<8.2f} " \
                      f"| {row.get('PUTS_Delta', 0):<7.4f} " \
                      f"| {row.get('PUTS_Gamma', 0):<7.4f} " \
                      f"| {row.get('PUTS_Theta', 0):<7.4f} " \
                      f"| {row.get('PUTS_Vega', 0):<6.4f} |"
            prompt_parts.append(row_str.replace('nan', 'N/A').replace('0.0000', '0.000')) # Basic NaN and zero formatting
    else:
        prompt_parts.append("\n**II. Option Chain Data (Strikes around ATM):** Data not available or not enough strikes.")

    if analysis_results and isinstance(analysis_results, dict):
        prompt_parts.append("\n**III. Key Derived Metrics & Observations:**")
        prompt_parts.append(f"- PCR (OI): {analysis_results.get('pcr_oi', 0):.2f}")
        prompt_parts.append(f"- Max Call OI Strike (OI Wall): {analysis_results.get('max_call_oi_strike', 0):,.0f}")
        prompt_parts.append(f"- Max Put OI Strike (OI Wall): {analysis_results.get('max_put_oi_strike', 0):,.0f}")
        prompt_parts.append(f"- Average ATM IV: {analysis_results.get('avg_atm_iv', 'N/A'):.2f}%" if analysis_results.get('avg_atm_iv') is not None else "- Average ATM IV: N/A")
        prompt_parts.append(f"- Derived Market View: {analysis_results.get('market_view', 'N/A')}")
        prompt_parts.append(f"- Derived Volatility Outlook: {analysis_results.get('volatility_outlook', 'N/A')}")
        
        prompt_parts.append("\n  **Gamma Exposure Metrics:**")
        prompt_parts.append(f"  - Gamma Flip Level (Strike): {analysis_results.get('gamma_flip_level', 0):,.0f}")
        prompt_parts.append(f"  - Call Gamma Wall (Strike, Exposure-based): {analysis_results.get('call_gamma_wall_strike', 0):,.0f}")
        prompt_parts.append(f"  - Put Gamma Wall (Strike, Exposure-based): {analysis_results.get('put_gamma_wall_strike', 0):,.0f}")
        prompt_parts.append(f"  - Net Market Gamma Exposure (Notional): ₹{analysis_results.get('net_market_gamma_exposure_notional', 0):,.0f}")
        
        prompt_parts.append("\n  **Cumulative Greeks (Overall Market):**")
        prompt_parts.append(f"  - Total Delta Exposure (Shares Equiv.): {analysis_results.get('total_delta_exposure', 0):,.0f}")
        prompt_parts.append(f"  - Total Gamma Exposure (Delta Chg per point for total OI): {analysis_results.get('total_gamma_exposure', 0):,.2f}") # Clarified unit
        prompt_parts.append(f"  - Total Theta Decay (₹ daily): ₹{analysis_results.get('total_theta_decay_daily', 0):,.0f}")
        prompt_parts.append(f"  - Total Vega Exposure (₹ per 1% IV chg): ₹{analysis_results.get('total_vega_exposure', 0):,.0f}")

    prompt_parts.append("\n**IV. Analysis Request for LLM:**")
    prompt_parts.append("1.  Interpret the overall market sentiment (bullish, bearish, neutral, volatile, range-bound) based on the data.")
    prompt_parts.append("2.  Identify key support and resistance levels using OI Walls and Gamma Walls. Explain their significance.")
    prompt_parts.append("3.  Analyze the IV Skew (if discernible from ATM data) and its implications.")
    prompt_parts.append("4.  Discuss the meaning of the Gamma Flip level and current CMP's position relative to it.")
    prompt_parts.append("5.  Based on the data, suggest 2-3 potential option strategies for a retail trader with limited capital (e.g., up to ₹50,000 - ₹2,00,000). For each strategy, specify:")
    prompt_parts.append("    -   Strategy Name (e.g., Bull Call Spread, Long Straddle).")
    prompt_parts.append("    -   Specific strikes to consider (if applicable).")
    prompt_parts.append("    -   Rationale based on the provided data (IV, OI, Gamma levels, market view).")
    prompt_parts.append("    -   Maximum Risk and Maximum Reward potential (qualitative or approximate).")
    prompt_parts.append("    -   Key risks or conditions under which the strategy might fail.")
    prompt_parts.append("6.  Are there any anomalies or particularly interesting patterns in the data (e.g., unusually high IV for a specific strike, large OI change)?")
    prompt_parts.append("7.  Provide a concise summary of the market outlook for the next few days based on this snapshot.")

    return "\n".join(prompt_parts)

BSM_FORMULAS_MD = """
**Footnotes: Option Pricing & Greeks (Simplified Black-Scholes-Merton)**
*   **Delta (Δ):** Change in option price per $1 change in underlying. `Δ_call ≈ N(d1)`, `Δ_put ≈ N(d1) - 1`.
*   **Gamma (Γ):** Change in Delta per $1 change in underlying. `Γ = N'(d1) / (S * σ * sqrt(T))`.
*   **Theta (Θ):** Option price change per day (time decay). Reported here as daily.
*   **Vega (ν):** Option price change per 1% change in Implied Volatility. `ν = S * N'(d1) * sqrt(T) / 100`.
*   _S=Spot, K=Strike, T=Time (yrs), r=Rate, q=Dividend Yield, σ=IV, N(x)=CDF, N'(x)=PDF of std. normal dist._
*   _Actual calculations use `py_vollib`. This is a conceptual guide._
"""

st.set_page_config(layout="wide")
st.title("📈 NIFTY Advanced Option Chain Analyzer")

default_expiry = date.today() + timedelta(days=5)
if (default_expiry - date.today()).days < 1: default_expiry = date.today() + timedelta(days=1)
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = {}
if 'df_with_all_calcs' not in st.session_state: st.session_state.df_with_all_calcs = pd.DataFrame()
if 'parsed_df_with_greeks' not in st.session_state: st.session_state.parsed_df_with_greeks = pd.DataFrame()
if 'cmp_for_analysis' not in st.session_state: st.session_state.cmp_for_analysis = 24750.0
if 'expiry_date_for_analysis' not in st.session_state: st.session_state.expiry_date_for_analysis = default_expiry
if 'interest_rate_for_analysis' not in st.session_state: st.session_state.interest_rate_for_analysis = 7.0
if 'dividend_yield_for_analysis' not in st.session_state: st.session_state.dividend_yield_for_analysis = 1.2 
if 'india_vix_input' not in st.session_state: st.session_state.india_vix_input = 16.5
if 'lot_size_input' not in st.session_state: st.session_state.lot_size_input = 75
if 'general_strategy_df' not in st.session_state: st.session_state.general_strategy_df = pd.DataFrame()
if 'iv_strategy_df' not in st.session_state: st.session_state.iv_strategy_df = pd.DataFrame()
if 'gamma_strategy_df' not in st.session_state: st.session_state.gamma_strategy_df = pd.DataFrame()


LOT_SIZE_OPTIONS = [25, 40, 50, 75, 500,1000]

# Add the new tab name to st.tabs
tab_names = [
    "⚙️ Input Data", "⛓️ Option Chain & Greeks", "📊 Summary & OI Walls", 
    "🧐 Greeks & Strategy", "💧 IV Analysis & Skew", "☢️ Gamma Exposure", "💹 OI Charts",
    "🤖 Analysis Prompt", "💡 Final Recommendations" # Added Analysis Prompt
]
tabs = st.tabs(tab_names)

input_tab = tabs[0]
option_chain_tab = tabs[1]
summary_tab = tabs[2]
greeks_strat_tab = tabs[3]
iv_analysis_tab = tabs[4]
gamma_exposure_tab = tabs[5]
oi_charts_tab = tabs[6]
analysis_prompt_tab = tabs[7] 
final_recommendations_tab = tabs[8]



with input_tab:
    st.header("Provide Market and Option Data")
    st.markdown("##### **Step 1: Market & Contract Parameters**")
    
    col_cmp, col_vix, col_lot_container = st.columns(3) # Use a container for lot size group
    with col_cmp: 
        nifty_cmp = st.number_input(
            "NIFTY CMP:", 
            min_value=1.0, 
            value=st.session_state.cmp_for_analysis, 
            step=0.05, 
            format="%.2f",
            help="Enter the current market price of the underlying (e.g., NIFTY spot price)."
        )
    with col_vix: 
        india_vix_val = st.number_input(
            "India VIX (%):", 
            min_value=1.0, 
            value=st.session_state.india_vix_input, 
            step=0.01, 
            format="%.2f", 
            key="india_vix_widget",
            help="Enter the current India VIX value (e.g., 16.5 for 16.5%)."
        )
    with col_lot_container: # Container for lot size widgets
        st.write("Contract Lot Size:") # Label for the group
        LOT_SIZE_PRESETS = [25, 40, 50, 75, 100]
        
        lot_size_choice_type = st.radio(
            "Lot Size Type:", # Shortened label for radio
            options=["Preset", "Custom"],
            horizontal=True,
            index=1 if st.session_state.custom_lot_size_active else 0, # Set index based on active state
            key="lot_size_type_radio",
            label_visibility="collapsed" # Hide the "Lot Size Type:" label if st.write is used above
        )

        if lot_size_choice_type == "Preset":
            st.session_state.custom_lot_size_active = False
            # If switching back from custom, try to find current lot_size_input in presets, or default
            current_preset_val = st.session_state.lot_size_input
            if current_preset_val not in LOT_SIZE_PRESETS:
                # If current lot_size_input (possibly from a previous custom entry) is not a preset,
                # default to the first preset or a common one like 75.
                current_preset_val = st.session_state.lot_size_input if st.session_state.lot_size_input in LOT_SIZE_PRESETS else 75
            
            try:
                preset_index = LOT_SIZE_PRESETS.index(current_preset_val)
            except ValueError:
                preset_index = LOT_SIZE_PRESETS.index(75) # Default to 75 if not found

            selected_preset_lot_size = st.selectbox(
                "Select Preset:", # Shortened label
                options=LOT_SIZE_PRESETS,
                index=preset_index,
                key="lot_size_preset_select",
                help="Select a common lot size."
            )
            lot_size_val = selected_preset_lot_size
            st.session_state.lot_size_input = lot_size_val 
            st.session_state.custom_lot_size_value = lot_size_val # Sync custom value display

        else: # Custom
            st.session_state.custom_lot_size_active = True
            custom_lot_size = st.number_input(
                "Enter Custom Lot Size:",
                min_value=1,
                value=int(st.session_state.custom_lot_size_value), # Ensure it's int for number_input
                step=1,
                key="lot_size_custom_input",
                help="Enter a specific lot size if not in presets."
            )
            lot_size_val = custom_lot_size
            st.session_state.lot_size_input = lot_size_val 
            st.session_state.custom_lot_size_value = lot_size_val

    col_exp, col_r, col_q = st.columns(3)
    with col_exp: 
        expiry_date = st.date_input(
            "Option Expiry Date:", 
            value=st.session_state.expiry_date_for_analysis, 
            min_value=date.today() + timedelta(days=1),
            help="Select the expiry date of the options you are analyzing."
        )
    with col_r: 
        interest_rate = st.number_input(
            "Annual Interest Rate (%):", 
            min_value=0.0, max_value=20.0, 
            value=st.session_state.interest_rate_for_analysis, 
            step=0.1, format="%.1f",
            help="Risk-free interest rate (e.g., 7 for 7%). Used in Black-Scholes."
        )
    with col_q: 
        dividend_yield = st.number_input(
            "Annual Dividend Yield (q, %):", 
            min_value=0.0, max_value=10.0, 
            value=st.session_state.dividend_yield_for_analysis, 
            step=0.05, format="%.2f",
            help="Continuous annual dividend yield of the underlying (e.g., 1.2 for 1.2%). Use 0 if none."
        )

    dte_val = (expiry_date - date.today()).days
    st.info(f"Days to Expiry (DTE): {dte_val} days (Time in years for BSM: {dte_val/365.25:.4f})")

    st.markdown("---"); st.markdown("##### **Step 2: Option Chain Data**")
    with st.expander("ℹ️ Click for instructions on getting Option Chain Data from NSE", expanded=False): st.markdown(HOW_TO_GET_DATA_NSE)
    data_source = st.radio("Choose data source:", ("Paste CSV", "Upload CSV"), horizontal=True, index=0, key="data_source_radio", help="Select data provision method.")
    csv_input_data = ""
    if data_source == "Paste CSV": csv_input_data = st.text_area("Paste Data:", value=DEFAULT_CSV_DATA, height=200, key="csv_paste_area", help="Paste entire content from NSE CSV.")
    else:
        uploaded_file = st.file_uploader("Upload NSE Option Chain CSV:", type=['csv'], key="csv_upload_widget", help="Upload .csv from NSE.")
        if uploaded_file: 
            try: csv_input_data = io.StringIO(uploaded_file.getvalue().decode("utf-8")).read(); st.success("File uploaded!")
            except Exception as e: st.error(f"Error reading file: {e}"); csv_input_data = ""
    
    st.markdown("---"); st.markdown("##### **Step 3: Analyze**")
    if st.button("🚀 Analyze Option Chain, Calculate Greeks & Generate Insights", type="primary", use_container_width=True, key="analyze_button"):
        if not nifty_cmp or dte_val <= 0 or not csv_input_data.strip(): st.error("Valid CMP, DTE > 0, & Option Data required.")
        else:
            # Ensure lot_size_val is correctly captured from the logic above before this button press
            # The variable 'lot_size_val' should be correctly set by the radio/selectbox/number_input logic
            st.session_state.india_vix_input = india_vix_val 
            st.session_state.dividend_yield_for_analysis = dividend_yield
            st.session_state.lot_size_input = lot_size_val # This line now correctly uses the resolved lot_size_val
            
            with st.spinner("Processing... Please wait."):
                # ... (rest of the button logic: parse_option_chain_csv, calculate_greeks, analyze_option_data_extended, etc.)
                # Make sure to pass the final `lot_size_val` to `analyze_option_data_extended`
                parsed_df = parse_option_chain_csv(csv_input_data)
                if not parsed_df.empty:
                    time_to_expiry_years = dte_val / 365.25
                    call_greeks = parsed_df.apply(lambda r: calculate_greeks(r, 'call', nifty_cmp, r['STRIKE'], time_to_expiry_years, interest_rate, dividend_yield, 'CALLS_IV'), axis=1).rename(columns=lambda c: f'CALLS_{c.capitalize()}')
                    put_greeks = parsed_df.apply(lambda r: calculate_greeks(r, 'put', nifty_cmp, r['STRIKE'], time_to_expiry_years, interest_rate, dividend_yield, 'PUTS_IV'), axis=1).rename(columns=lambda c: f'PUTS_{c.capitalize()}')
                    df_with_greeks = pd.concat([parsed_df, call_greeks, put_greeks], axis=1)
                    st.session_state.parsed_df_with_greeks = df_with_greeks
                    
                    # Pass the final lot_size_val to the analysis function
                    analysis_results_dict, df_processed_for_analysis = analyze_option_data_extended(df_with_greeks, nifty_cmp, india_vix_val, lot_size_val, interest_rate, dividend_yield, dte_val) 
                    
                    st.session_state.analysis_results = analysis_results_dict
                    st.session_state.df_with_all_calcs = df_processed_for_analysis
                    st.session_state.cmp_for_analysis = nifty_cmp
                    st.session_state.expiry_date_for_analysis = expiry_date
                    st.session_state.interest_rate_for_analysis = interest_rate
                    
                    # Generate and store all suggestion tables after main analysis
                    res = st.session_state.analysis_results
                    avg_atm_iv_for_strat = res.get('avg_atm_iv', st.session_state.india_vix_input) 
                    if pd.isna(avg_atm_iv_for_strat): avg_atm_iv_for_strat = st.session_state.india_vix_input

                    st.session_state.general_strategy_df = pd.DataFrame(get_strategy_suggestions(
                        res.get('market_view', 'Neutral'), res.get('volatility_outlook', 'Moderate'), 
                        avg_atm_iv_for_strat, analysis_res=None 
                    ))
                    iv_suggestions_list_final = []
                    vix_current_local = st.session_state.india_vix_input # Use the one from state
                    if not pd.isna(avg_atm_iv_for_strat): # Use resolved avg_atm_iv_for_strat
                        if avg_atm_iv_for_strat > 18 and avg_atm_iv_for_strat > vix_current_local: 
                            iv_suggestions_list_final.append({"Category": "IV Profile", "Strategy": "Consider Option Selling", "Condition": "High IV > VIX", "Rationale": "Premiums rich.", "Max Risk": "Varies", "Max Reward": "Limited"})
                        elif avg_atm_iv_for_strat < 14 and avg_atm_iv_for_strat < vix_current_local:
                            iv_suggestions_list_final.append({"Category": "IV Profile", "Strategy": "Consider Option Buying", "Condition": "Low IV < VIX", "Rationale": "Premiums cheap.", "Max Risk": "Premium", "Max Reward": "High"})
                    st.session_state.iv_strategy_df = pd.DataFrame(iv_suggestions_list_final)

                    gamma_suggestions_list_final = []
                    g_flip_val_local = res.get('gamma_flip_level',0)
                    net_mkt_gexp_notional_val_local = res.get('net_market_gamma_exposure_notional',0)
                    if g_flip_val_local != 0:
                        if nifty_cmp < g_flip_val_local and net_mkt_gexp_notional_val_local < -100_000_000: 
                             gamma_suggestions_list_final.append({"Category": "Gamma Env.", "Strategy Type": "Long Gamma", "Condition": f"CMP < G-Flip ({g_flip_val_local}), Sig. Neg. GExp", "Consider": "Straddle/Strangle, Directional Buys"})
                        elif nifty_cmp > g_flip_val_local and net_mkt_gexp_notional_val_local > 100_000_000:
                             gamma_suggestions_list_final.append({"Category": "Gamma Env.", "Strategy Type": "Short Gamma", "Condition": f"CMP > G-Flip ({g_flip_val_local}), Sig. Pos. GExp", "Consider": "Iron Condors, Spreads"})
                    st.session_state.gamma_strategy_df = pd.DataFrame(gamma_suggestions_list_final)

                    st.success("Analysis Complete! Check other tabs for details.")
                else: st.error("Failed to parse option chain data.")
                
with option_chain_tab:
    st.header("Processed Option Chain with Greeks")
    if not st.session_state.parsed_df_with_greeks.empty:
        df_display_full = st.session_state.parsed_df_with_greeks.copy()
        cmp_used = st.session_state.cmp_for_analysis
        call_group1 = ['CALLS_OI', 'CALLS_VOLUME', 'CALLS_IV', 'CALLS_LTP']
        call_greeks_group = ['CALLS_Delta', 'CALLS_Gamma', 'CALLS_Theta', 'CALLS_Vega']
        call_group2 = ['CALLS_CHNG_IN_OI','CALLS_CHNG', 'CALLS_BID_QTY', 'CALLS_BID', 'CALLS_ASK', 'CALLS_ASK_QTY']
        put_group1 = ['PUTS_LTP', 'PUTS_IV', 'PUTS_VOLUME', 'PUTS_OI']
        put_greeks_group = ['PUTS_Delta', 'PUTS_Gamma', 'PUTS_Theta', 'PUTS_Vega']
        put_group2 = ['PUTS_CHNG_IN_OI', 'PUTS_CHNG', 'PUTS_BID_QTY', 'PUTS_BID', 'PUTS_ASK', 'PUTS_ASK_QTY']
        ordered_cols = \
            [c for c in call_group1 if c in df_display_full.columns] + \
            [c for c in call_greeks_group if c in df_display_full.columns] + \
            [c for c in call_group2 if c in df_display_full.columns] + \
            (['STRIKE'] if 'STRIKE' in df_display_full.columns else []) + \
            [c for c in put_group2 if c in df_display_full.columns] + \
            [c for c in put_greeks_group if c in df_display_full.columns] + \
            [c for c in put_group1 if c in df_display_full.columns]
        display_df_reordered = df_display_full.sort_values(by='STRIKE').reset_index(drop=True)
        final_display_cols = [col for col in ordered_cols if col in display_df_reordered.columns]
        if not final_display_cols : final_display_cols = display_df_reordered.columns.tolist()
        df_to_show = display_df_reordered[final_display_cols]
        styled_df = style_option_chain(df_to_show, cmp_used)
        st.dataframe(styled_df, height=600, use_container_width=True)
        st.markdown("---")
        with st.expander("Formulas & Definitions Used (Simplified)"): st.markdown(BSM_FORMULAS_MD)
    else: st.info("Analyze data first.")

with summary_tab:
    st.header("Key Market Summary & OI Walls")
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        cmp = st.session_state.cmp_for_analysis
        st.subheader(f"Overall Market View (CMP: {cmp:,.2f})")
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Derived Market View", res.get('market_view', 'N/A'))
        with col2: st.metric("PCR (OI)", f"{res.get('pcr_oi',0):.2f}")
        with col3: st.metric("Volatility Outlook", res.get('volatility_outlook', 'N/A'))
        col_iv, col_dte, col_rates = st.columns(3)
        with col_iv:
            avg_atm_iv_display = res.get('avg_atm_iv')
            st.metric("Avg ATM IV", f"{avg_atm_iv_display:.2f}%" if avg_atm_iv_display is not None and not pd.isna(avg_atm_iv_display) else "N/A")
        with col_dte: st.metric("DTE Used", f"{res.get('dte_used','N/A')} days")
        with col_rates: st.metric("Rates (r% | q% | Lot)", f"{res.get('interest_rate_used','N/A')}% | {res.get('dividend_yield_used','N/A')}% | {res.get('lot_size_used','N/A')}")
        st.subheader("Potential OI Walls (Support/Resistance)")
        st.write("Call Walls (Potential Resistance):")
        call_walls_df = res.get('call_walls_detail', pd.DataFrame())
        if not call_walls_df.empty: st.dataframe(call_walls_df.set_index('STRIKE'), use_container_width=True)
        else: st.write("No significant call walls identified.")
        st.write("Put Walls (Potential Support):")
        put_walls_df = res.get('put_walls_detail', pd.DataFrame())
        if not put_walls_df.empty: st.dataframe(put_walls_df.set_index('STRIKE'), use_container_width=True)
        else: st.write("No significant put walls identified.")
    else: st.info("Analyze data first.")

with greeks_strat_tab:
    st.header("Cumulative Greeks Exposure & General Strategy Ideas")
    if st.session_state.analysis_results:
        res = st.session_state.analysis_results
        st.subheader("Cumulative Greeks Exposure (OI Weighted * Lot Size)")
        if 'cumulative_greeks_error' in res: st.warning(res['cumulative_greeks_error'])
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Delta Exposure (Shares Equiv.)", f"{res.get('total_delta_exposure', 0):,.0f}")
                st.metric("Total Theta Decay (Value daily)", f"₹{res.get('total_theta_decay_daily', 0):,.0f}")
            with col2:
                st.metric("Total Gamma Exposure (Delta Chg per point)", f"{res.get('total_gamma_exposure', 0):,.2f}")
                st.metric("Total Vega Exposure (Value per 1% IV chg)", f"₹{res.get('total_vega_exposure', 0):,.0f}")
        st.markdown("---")
        st.subheader("General Strategy Ideas (View & IV based)")
        st.warning("""**DISCLAIMER:** EDUCATIONAL. NOT FINANCIAL ADVICE. RISKY. DYOR & CONSULT ADVISOR.""")
        if not st.session_state.general_strategy_df.empty:
            st.dataframe(st.session_state.general_strategy_df, hide_index=True, use_container_width=True, height=(len(st.session_state.general_strategy_df) + 1) * 35 + 3)
        else: st.info("No general strategy suggestions were generated or available.")
    else: st.info("Analyze data first.")

with iv_analysis_tab:
    st.header("IV Analysis, Skew & OI in Cash")
    if not st.session_state.df_with_all_calcs.empty and st.session_state.analysis_results:
        df_iv_analysis_source = st.session_state.df_with_all_calcs.copy()
        cmp_atm = st.session_state.cmp_for_analysis
        vix_current = st.session_state.india_vix_input
        atm_strike_rounded = round(cmp_atm / 50) * 50
        strike_step = 50 
        num_strikes_side = st.slider("Number of Strikes from ATM for IV Analysis:", 3, 15, st.session_state.get("iv_analysis_num_strikes_val",7) , key="iv_analysis_strikes_slider_val")
        st.session_state.iv_analysis_num_strikes_val = num_strikes_side
        min_strike_iv = atm_strike_rounded - (num_strikes_side * strike_step)
        max_strike_iv = atm_strike_rounded + (num_strikes_side * strike_step)
        df_iv_atm = df_iv_analysis_source[(df_iv_analysis_source['STRIKE'] >= min_strike_iv) & (df_iv_analysis_source['STRIKE'] <= max_strike_iv)].sort_values(by="STRIKE").reset_index(drop=True)
        if not df_iv_atm.empty:
            iv_data = pd.DataFrame()
            iv_data['STRIKE'] = df_iv_atm['STRIKE']
            iv_data['CALLS_IV'] = df_iv_atm['CALLS_IV'].fillna(0); iv_data['PUTS_IV'] = df_iv_atm['PUTS_IV'].fillna(0)
            iv_data['IV_Diff (C-P)'] = iv_data['CALLS_IV'] - iv_data['PUTS_IV']
            iv_data['IV_Sum (C+P)'] = iv_data['CALLS_IV'] + iv_data['PUTS_IV']
            iv_data['IV_Ratio (P/C)'] = (iv_data['PUTS_IV'] / iv_data['CALLS_IV']).replace([np.inf, -np.inf], np.nan).fillna(0)
            st.subheader("IV Data around ATM"); st.dataframe(iv_data.set_index('STRIKE').style.format("{:.2f}", na_rep='-'), use_container_width=True)
            fig_iv_skew = go.Figure()
            fig_iv_skew.add_trace(go.Scatter(x=iv_data['STRIKE'], y=iv_data['CALLS_IV'], mode='lines+markers', name='Call IV', line=dict(color='skyblue')))
            fig_iv_skew.add_trace(go.Scatter(x=iv_data['STRIKE'], y=iv_data['PUTS_IV'], mode='lines+markers', name='Put IV', line=dict(color='orange')))
            fig_iv_skew.add_hline(y=vix_current, line_dash="dot", annotation_text=f"India VIX: {vix_current}%", annotation_position="bottom right", line_color="green")
            fig_iv_skew.update_layout(title='Volatility Smile/Skew vs Strikes', xaxis_title='Strike Price', yaxis_title='Implied Volatility (%)'); st.plotly_chart(fig_iv_skew, use_container_width=True)
            st.subheader("OI vs OI in Cash (₹) around ATM")
            oi_cash_data = pd.DataFrame({'STRIKE': df_iv_atm['STRIKE'], 'CALLS_OI': df_iv_atm['CALLS_OI'].fillna(0), 'CALLS_OI_Cash': df_iv_atm['CALLS_OI_Cash'].fillna(0), 'PUTS_OI': df_iv_atm['PUTS_OI'].fillna(0), 'PUTS_OI_Cash': df_iv_atm['PUTS_OI_Cash'].fillna(0)})
            st.dataframe(oi_cash_data.set_index('STRIKE').style.format("{:,.0f}", na_rep='-'), use_container_width=True)
            st.subheader("IV Observations & Potential Strategy Insights")
            avg_call_iv_atm = iv_data['CALLS_IV'][iv_data['CALLS_IV'] > 0].mean(); avg_put_iv_atm = iv_data['PUTS_IV'][iv_data['PUTS_IV'] > 0].mean()
            obs_text = f"Current India VIX: **{vix_current:.2f}%**.\n"; overall_atm_iv = np.nan
            if not pd.isna(avg_call_iv_atm) and not pd.isna(avg_put_iv_atm):
                overall_atm_iv = (avg_call_iv_atm + avg_put_iv_atm) / 2.0
                obs_text += f"Avg Call IV (near ATM): **{avg_call_iv_atm:.2f}%**. Avg Put IV (near ATM): **{avg_put_iv_atm:.2f}%**.\n"
                if overall_atm_iv > vix_current * 1.1: obs_text += "Option IVs near ATM trading at premium to VIX.\n"
                elif overall_atm_iv < vix_current * 0.9: obs_text += "Option IVs near ATM trading at discount to VIX.\n"
                else: obs_text += "Option IVs near ATM roughly in line with VIX.\n"
                atm_iv_row = iv_data.iloc[(iv_data['STRIKE'] - atm_strike_rounded).abs().argsort()[:1]]
                if not atm_iv_row.empty:
                    atm_put_iv, atm_call_iv = atm_iv_row['PUTS_IV'].iloc[0], atm_iv_row['CALLS_IV'].iloc[0]
                    if atm_put_iv > atm_call_iv * 1.05: obs_text += "Put Skew observed.\n"
                    elif atm_call_iv > atm_put_iv * 1.05: obs_text += "Call Skew observed.\n"
                    else: obs_text += "Call/Put IVs at ATM balanced.\n"
            else: obs_text += "Could not calculate average ATM IVs.\n"
            st.markdown(obs_text)
            iv_strategy_suggestion = "Based on IV profile:\n"
            if not pd.isna(overall_atm_iv):
                if overall_atm_iv > 18 and overall_atm_iv > vix_current: iv_strategy_suggestion += "- Consider Option Selling (high IV)."
                elif overall_atm_iv < 14 and overall_atm_iv < vix_current: iv_strategy_suggestion += "- Consider Option Buying (low IV)."
                else: iv_strategy_suggestion += "- IV moderate; focus on direction."
            else: iv_strategy_suggestion += "- IV profile unclear."
            st.info(iv_strategy_suggestion); st.caption("Disclaimer: Generalized, not financial advice.")
        else: st.info("No option data for selected IV strikes.")
    else: st.info("Analyze data first.")

with gamma_exposure_tab:
    st.header("☢️ Gamma Exposure Analysis & Key Levels")
    if st.session_state.analysis_results and not st.session_state.df_with_all_calcs.empty:
        res = st.session_state.analysis_results
        df_gamma_analysis_full = st.session_state.df_with_all_calcs.copy()
        cmp_val = st.session_state.cmp_for_analysis
        atm_strike_rounded_gamma = round(cmp_val / 50) * 50
        min_s_gamma = int(df_gamma_analysis_full['STRIKE'].min()) if not df_gamma_analysis_full.empty and not df_gamma_analysis_full['STRIKE'].dropna().empty else int(atm_strike_rounded_gamma - 1000)
        max_s_gamma = int(df_gamma_analysis_full['STRIKE'].max()) if not df_gamma_analysis_full.empty and not df_gamma_analysis_full['STRIKE'].dropna().empty else int(atm_strike_rounded_gamma + 1000)
        if min_s_gamma >= max_s_gamma : min_s_gamma = max_s_gamma - 500; 
        if min_s_gamma < 0 : min_s_gamma = 0
        default_range_min_g = max(min_s_gamma, int(atm_strike_rounded_gamma - 750))
        default_range_max_g = min(max_s_gamma, int(atm_strike_rounded_gamma + 750))
        if default_range_min_g >= default_range_max_g: default_range_min_g = min_s_gamma; default_range_max_g = max_s_gamma
        strike_range_gamma = st.slider("Strike Range for Gamma Exposure Chart:", min_s_gamma, max_s_gamma, (default_range_min_g, default_range_max_g), step=50, key="gamma_exp_strike_slider")
        df_gamma_analysis_filtered = df_gamma_analysis_full[(df_gamma_analysis_full['STRIKE'] >= strike_range_gamma[0]) & (df_gamma_analysis_full['STRIKE'] <= strike_range_gamma[1])]
        st.subheader("Key Gamma Exposure Metrics (Notional)")
        if res.get('gamma_exposure_error'): st.warning(res['gamma_exposure_error'])
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Call G-Exp", f"₹{res.get('total_call_gamma_exposure_notional', 0):,.0f}")
            col2.metric("Total Put G-Exp", f"₹{res.get('total_put_gamma_exposure_notional', 0):,.0f}")
            col3.metric("Net Market G-Exp", f"₹{res.get('net_market_gamma_exposure_notional', 0):,.0f}")
            col_flip, col_cwall, col_pwall = st.columns(3)
            col_flip.metric("G-Flip Level", f"{res.get('gamma_flip_level', 0):,.0f}")
            col_cwall.metric("Call G-Wall", f"{res.get('call_gamma_wall_strike', 0):,.0f}")
            col_pwall.metric("Put G-Wall", f"{res.get('put_gamma_wall_strike', 0):,.0f}")
        if not df_gamma_analysis_filtered.empty and 'CALLS_Gamma_Exposure_Notional' in df_gamma_analysis_filtered.columns:
            st.markdown("---"); st.subheader("Gamma Exposure Charts (Filtered Range)")
            fig_exp_strike = go.Figure()
            fig_exp_strike.add_trace(go.Bar(x=df_gamma_analysis_filtered['STRIKE'], y=df_gamma_analysis_filtered['CALLS_Gamma_Exposure_Notional'], name='Call G-Exp', marker_color='green'))
            fig_exp_strike.add_trace(go.Bar(x=df_gamma_analysis_filtered['STRIKE'], y=df_gamma_analysis_filtered['PUTS_Gamma_Exposure_Notional'], name='Put G-Exp', marker_color='red'))
            fig_exp_strike.update_layout(title_text='Notional Gamma Exposure by Strike', xaxis_title='Strike', yaxis_title='Notional G-Exp (₹)', barmode='group')
            st.plotly_chart(fig_exp_strike, use_container_width=True)
            if 'Net_Gamma_Exposure_At_Strike' in df_gamma_analysis_filtered.columns:
                fig_net_exp_strike = px.bar(df_gamma_analysis_filtered, x='STRIKE', y='Net_Gamma_Exposure_At_Strike', title='Net Notional Gamma Exposure by Strike', color='Net_Gamma_Exposure_At_Strike', color_continuous_scale=px.colors.diverging.RdYlGn_r)
                fig_net_exp_strike.add_hline(y=0, line_dash="dash", line_color="gray")
                gamma_flip_val = res.get('gamma_flip_level', 0)
                if gamma_flip_val != 0 and gamma_flip_val >= strike_range_gamma[0] and gamma_flip_val <= strike_range_gamma[1]: fig_net_exp_strike.add_vline(x=gamma_flip_val, line_dash="dot", annotation_text=f"G-Flip: {gamma_flip_val}", annotation_position="top left", line_color="purple")
                st.plotly_chart(fig_net_exp_strike, use_container_width=True)
            st.markdown("---"); st.subheader("Gamma Exposure Interpretations & Strategy Insights")
            g_flip, c_gwall, p_gwall, net_mkt_gexp = res.get('gamma_flip_level', 0), res.get('call_gamma_wall_strike', 0), res.get('put_gamma_wall_strike', 0), res.get('net_market_gamma_exposure_notional',0)
            st.markdown(f"**CMP:** `{cmp_val:,.2f}` | **G-Flip:** `{g_flip:,.0f}` | **Call G-Wall:** `{c_gwall:,.0f}` | **Put G-Wall:** `{p_gwall:,.0f}`")
            interpretation = ""
            if g_flip !=0: 
                if cmp_val > g_flip : interpretation += "- CMP > G-Flip: Dealers may dampen volatility.\n"
                elif cmp_val < g_flip : interpretation += "- CMP < G-Flip: Dealers may amplify moves.\n"
            if net_mkt_gexp > 0: interpretation += f"- Net Market G-Exp Positive (`₹{net_mkt_gexp:,.0f}`): Suggests Call sellers dominant/dealers net long gamma.\n"
            elif net_mkt_gexp < 0: interpretation += f"- Net Market G-Exp Negative (`₹{net_mkt_gexp:,.0f}`): Suggests Put sellers dominant/dealers net short gamma.\n"
            st.markdown(interpretation if interpretation else "Gamma Flip/Net Exposure inconclusive.")
            gamma_strat_list = []
            if g_flip != 0:
                if cmp_val < g_flip and net_mkt_gexp < -1e8: gamma_strat_list.append({"Category": "Gamma Env.", "Strategy": "Long Gamma (Straddle/Strangle)", "Rationale": "Potential for amplified moves."})
                elif cmp_val > g_flip and net_mkt_gexp > 1e8: gamma_strat_list.append({"Category": "Gamma Env.", "Strategy": "Short Gamma (Iron Condor - High Risk)", "Rationale": "Potential for dampened moves."})
            df_gamma_strat = pd.DataFrame(gamma_strat_list)
            if not df_gamma_strat.empty: st.dataframe(df_gamma_strat, hide_index=True, use_container_width=True)
            else: st.info("No strong Gamma-specific strategy signals based on current thresholds.")
            st.session_state.gamma_strategy_df = df_gamma_strat
        else: st.warning("Gamma Exposure data not calculated/available for charts in selected range.")
    else: st.info("Analyze data first.")

with oi_charts_tab:
    st.header("Open Interest Distribution")
    if not st.session_state.df_with_all_calcs.empty and st.session_state.analysis_results:
        df_chart_data = st.session_state.df_with_all_calcs.copy().sort_values(by='STRIKE')
        analysis_res = st.session_state.analysis_results
        cmp_used = st.session_state.cmp_for_analysis
        atm_strike_rounded = round(cmp_used / 50) * 50
        max_call_oi_strike_val = analysis_res.get('max_call_oi_strike')
        max_put_oi_strike_val = analysis_res.get('max_put_oi_strike')
        min_s_val = int(df_chart_data['STRIKE'].min()) if not df_chart_data.empty and not df_chart_data['STRIKE'].dropna().empty else int(atm_strike_rounded - 1000)
        max_s_val = int(df_chart_data['STRIKE'].max()) if not df_chart_data.empty and not df_chart_data['STRIKE'].dropna().empty else int(atm_strike_rounded + 1000)
        if min_s_val >= max_s_val : min_s_val = max_s_val - 500; 
        if min_s_val < 0 : min_s_val = 0
        default_range_min = max(min_s_val, int(atm_strike_rounded - 1000))
        default_range_max = min(max_s_val, int(atm_strike_rounded + 1000))
        if default_range_min >= default_range_max: default_range_min = min_s_val; default_range_max = max_s_val
        strike_range_oi = st.slider("Strike Range for OI Chart:", min_s_val, max_s_val, (default_range_min, default_range_max), step=50, key="oi_strike_slider_main")
        df_chart_filtered = df_chart_data[(df_chart_data['STRIKE'] >= strike_range_oi[0]) & (df_chart_data['STRIKE'] <= strike_range_oi[1])]
        if not df_chart_filtered.empty:
            fig_oi = go.Figure()
            fig_oi.add_trace(go.Bar(x=df_chart_filtered['STRIKE'], y=df_chart_filtered['CALLS_OI'].fillna(0), name='Call OI', marker_color='lightgreen'))
            fig_oi.add_trace(go.Bar(x=df_chart_filtered['STRIKE'], y=df_chart_filtered['PUTS_OI'].fillna(0), name='Put OI', marker_color='salmon'))
            annotations = []
            if max_call_oi_strike_val and max_call_oi_strike_val >= strike_range_oi[0] and max_call_oi_strike_val <= strike_range_oi[1]:
                max_call_oi_val_series = df_chart_filtered[df_chart_filtered['STRIKE'] == max_call_oi_strike_val]['CALLS_OI']
                if not max_call_oi_val_series.empty:
                    max_call_oi_val = max_call_oi_val_series.iloc[0]
                    if max_call_oi_val > 0 : annotations.append(dict(x=max_call_oi_strike_val, y=max_call_oi_val, text="Max CE OI", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="darkgreen")))
            if max_put_oi_strike_val and max_put_oi_strike_val >= strike_range_oi[0] and max_put_oi_strike_val <= strike_range_oi[1]:
                max_put_oi_val_series = df_chart_filtered[df_chart_filtered['STRIKE'] == max_put_oi_strike_val]['PUTS_OI']
                if not max_put_oi_val_series.empty:
                    max_put_oi_val = max_put_oi_val_series.iloc[0]
                    if max_put_oi_val > 0: annotations.append(dict(x=max_put_oi_strike_val, y=max_put_oi_val, text="Max PE OI", showarrow=True, arrowhead=2, ax=0, ay=-30, font=dict(color="darkred")))
            fig_oi.update_layout(barmode='group', title_text='Open Interest Distribution (Calls vs Puts)', xaxis_title='Strike Price', yaxis_title='Open Interest (Contracts)', annotations=annotations)
            st.plotly_chart(fig_oi, use_container_width=True)
        else: st.info("No data in selected strike range for OI charts.")
    else: st.info("Analyze data first.")

with analysis_prompt_tab:
    st.header("🤖 Generate Analysis Prompt for LLMs")
    st.info("""
    This section compiles key data from your current analysis into a structured prompt. 
    You can copy this prompt and use it with Large Language Models (LLMs) like ChatGPT, Claude, Gemini, etc., 
    for further in-depth analysis, alternative interpretations, or specific queries.
    """)

    if st.session_state.analysis_results and isinstance(st.session_state.analysis_results, dict) and \
       not st.session_state.df_with_all_calcs.empty:

        res = st.session_state.analysis_results
        df_full_chain = st.session_state.df_with_all_calcs.copy()
        
        cmp_current = st.session_state.cmp_for_analysis
        vix_current = st.session_state.india_vix_input
        lot_size_current = st.session_state.lot_size_input
        dte_current = res.get('dte_used', 0)
        interest_rate_current = res.get('interest_rate_used', 0)
        dividend_yield_current = res.get('dividend_yield_used', 0)

        # Select ~10 strikes around ATM for the prompt (+/- 5 from ATM rounded strike)
        atm_strike_rounded = round(cmp_current / 50) * 50
        strike_step = 50 
        num_strikes_side_prompt = 5 # 5 on each side + ATM = 11 strikes approx.

        min_strike_prompt = atm_strike_rounded - (num_strikes_side_prompt * strike_step)
        max_strike_prompt = atm_strike_rounded + (num_strikes_side_prompt * strike_step)
        
        df_atm_for_prompt = df_full_chain[
            (df_full_chain['STRIKE'] >= min_strike_prompt) & 
            (df_full_chain['STRIKE'] <= max_strike_prompt)
        ].sort_values(by="STRIKE").reset_index(drop=True)
        
        # Ensure necessary Greek columns are present for the prompt generation
        # The generate_llm_analysis_prompt function uses .get() for safety
        
        generated_prompt = generate_llm_analysis_prompt(
            cmp=cmp_current,
            vix=vix_current,
            lot_size=lot_size_current,
            dte=dte_current,
            interest_rate=interest_rate_current,
            dividend_yield=dividend_yield_current,
            df_atm_strikes=df_atm_for_prompt,
            analysis_results=res
        )
        
        st.subheader("Generated Prompt:")
        st.text_area("Copy the prompt below:", value=generated_prompt, height=600, key="llm_prompt_text_area")
        
        # A simple copy button (Streamlit doesn't have a native one, this is a common workaround)
        # For a real copy-to-clipboard button, you'd typically need streamlit_js_eval or similar
        st.markdown("Right-click on the text area and select 'Copy', or use Ctrl+A then Ctrl+C.")
        st.download_button(
            label="Download Prompt as .txt",
            data=generated_prompt,
            file_name="llm_option_analysis_prompt.txt",
            mime="text/plain"
        )

    else:
        st.warning("Please analyze option chain data first to generate a prompt.")

with final_recommendations_tab:
    st.header("💡 Consolidated Strategy Insights")
    st.warning("""**VERY IMPORTANT DISCLAIMER:** All suggestions provided are for **EDUCATIONAL & INFORMATIONAL PURPOSES ONLY**. 
    They are generated algorithmically based on data patterns and simplified models, and **DO NOT CONSTITUTE FINANCIAL ADVICE**.
    Option trading involves substantial risk of loss and may not be suitable for all investors. 
    Market conditions are dynamic. **Always conduct your own thorough research (DYOR), understand the specific risks of each strategy, manage your capital wisely, and consult with a qualified financial advisor before making any trading decisions.**
    Past performance or model-based suggestions are not indicative of future results.
    """)

    if st.session_state.analysis_results and isinstance(st.session_state.analysis_results, dict): # Check if results exist and is a dict
        st.markdown("---")
        st.subheader("1. General View & IV Based Strategy Ideas")
        if 'general_strategy_df' in st.session_state and not st.session_state.general_strategy_df.empty:
            st.dataframe(st.session_state.general_strategy_df, hide_index=True, use_container_width=True)
        else:
            st.info("No general strategy ideas were generated (check 'Greeks & Strategy' tab or re-analyze).")

        st.markdown("---")
        st.subheader("2. IV Profile Based Strategy Insights")
        if 'iv_strategy_df' in st.session_state and not st.session_state.iv_strategy_df.empty:
            st.dataframe(st.session_state.iv_strategy_df, hide_index=True, use_container_width=True)
        else:
            st.info("No specific IV profile based insights were generated (check 'IV Analysis' tab or re-analyze).")

        st.markdown("---")
        st.subheader("3. Gamma Exposure Environment Insights")
        if 'gamma_strategy_df' in st.session_state and not st.session_state.gamma_strategy_df.empty:
            st.dataframe(st.session_state.gamma_strategy_df, hide_index=True, use_container_width=True)
        else:
            st.info("No Gamma exposure specific insights were generated (check 'Gamma Exposure' tab or re-analyze).")
        
        st.markdown("---")
        st.info("Consider these insights together, along with your own market view and risk appetite. The 'best' strategy often depends on multiple converging factors.")
    else:
        st.info("Please analyze option chain data first to see consolidated recommendations.")