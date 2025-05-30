import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# --- Default Values & Configuration ---
DEFAULT_INTEREST_RATE = 7.0
DTE_OPTIONS = list(range(0, 31))
DEFAULT_DTE = 7
N_RECOMMENDATIONS = 2 # Number of strikes to recommend

# --- Helper Functions ---
def parse_options_data(data_string):
    """
    Parses the pasted options data string into a Pandas DataFrame.
    Expected 10-column order (Calls on Left, Puts on Right):
    Gamma_Call, Volume_Call, OI_Lakh_Call, LTP_Call, Strike, IV_Call, LTP_Put, OI_Lakh_Put, Volume_Put, Gamma_Put
    """
    if not data_string.strip():
        return pd.DataFrame()
    lines = data_string.strip().split('\n')
    data_rows = []
    col_names = [
        'Gamma_Call', 'Volume_Call', 'OI_Lakh_Call', 'LTP_Call', 
        'Strike', 
        'IV_Call', 
        'LTP_Put', 'OI_Lakh_Put', 'Volume_Put', 'Gamma_Put'
    ]
    for line in lines:
        parts = line.split() 
        if len(parts) == 10: 
            try:
                row = [float(p) for p in parts]
                data_rows.append(row)
            except ValueError:
                pass 
    if not data_rows:
        st.error("No valid data rows found. Ensure 10 numeric columns per line and correct order (Calls on left, Puts on right).")
        return pd.DataFrame()
    
    df = pd.DataFrame(data_rows, columns=col_names)
    int_cols = ['Strike', 'Volume_Call', 'Volume_Put']
    for col in int_cols:
        if col in df.columns: df[col] = df[col].astype(int)
    df['OI_Call'] = df['OI_Lakh_Call'] * 100000
    df['OI_Put'] = df['OI_Lakh_Put'] * 100000
    return df

def calculate_metrics(df, market_price):
    if df.empty or market_price <= 0: return df, 0, 0, 0, 0
    df['Exposure_Put'] = df['OI_Put'] * (market_price**2) * df['Gamma_Put'] * 0.01
    df['Exposure_Call'] = df['OI_Call'] * (market_price**2) * df['Gamma_Call'] * 0.01
    df['Net_Call_Exposure_At_Strike'] = df['Exposure_Call'] - df['Exposure_Put'] 
    total_gamma = df['Gamma_Put'].sum() + df['Gamma_Call'].sum()
    total_exposure_put = df['Exposure_Put'].sum()
    total_exposure_call = df['Exposure_Call'].sum()
    net_market_exposure = total_exposure_call - total_exposure_put
    return df, total_gamma, total_exposure_put, total_exposure_call, net_market_exposure

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title("📈 Option Chain Gamma Exposure Analyzer")

for key, default_val in [
    ('calculated_df', pd.DataFrame()), ('total_gamma', 0),
    ('total_exposure_put', 0), ('total_exposure_call', 0),
    ('net_market_exposure', 0), ('market_price', 80000.0),
    ('interest_rate', DEFAULT_INTEREST_RATE), ('dte', DEFAULT_DTE),
    ('gamma_flip_level', 0), ('call_wall_strike', 0), ('put_wall_strike', 0) 
]:
    if key not in st.session_state: st.session_state[key] = default_val

with st.sidebar:
    st.header("1. Input Parameters")
    st.session_state.market_price = st.number_input("Current Market Price (CMP) of Underlying", value=st.session_state.market_price, min_value=0.01, format="%.2f")
    st.session_state.interest_rate = st.number_input("Interest Rate (%)", value=st.session_state.interest_rate, min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
    st.session_state.dte = st.selectbox("Days To Expiry (DTE)", options=DTE_OPTIONS, index=DTE_OPTIONS.index(st.session_state.dte))
    st.subheader("Paste Option Chain Data")
    options_data_paste = st.text_area(
        "Data Format (10 columns, space/tab separated):", height=250,
        placeholder="Paste 10 columns per line in this EXACT order:\nGamma_CALL  Vol_CALL  OI_Lakh_CALL  LTP_CALL  STRIKE  IV_CALL  LTP_PUT  OI_Lakh_PUT  Vol_PUT  Gamma_PUT\n\nExample:\n0.0002 11 1.1 58.85 79400 16.9 2281.10 0.0 0 0.0001"
    )
    st.caption("Order: CALLS data | STRIKE | IV_CALL | PUTS data.")
    if st.button("Analyze Option Chain"):
        if not options_data_paste: st.error("Please paste options data.")
        elif st.session_state.market_price <= 0: st.error("Please enter a valid market price.")
        else:
            raw_df = parse_options_data(options_data_paste)
            if not raw_df.empty:
                st.session_state.calculated_df, st.session_state.total_gamma, \
                st.session_state.total_exposure_put, st.session_state.total_exposure_call, \
                st.session_state.net_market_exposure = calculate_metrics(raw_df.copy(), st.session_state.market_price)
                
                # Calculate Gamma Flip, Call Wall, Put Wall
                if 'Net_Call_Exposure_At_Strike' in st.session_state.calculated_df.columns and not st.session_state.calculated_df.empty:
                    st.session_state.gamma_flip_level = st.session_state.calculated_df.loc[st.session_state.calculated_df['Net_Call_Exposure_At_Strike'].abs().idxmin()]['Strike']
                else:
                    st.session_state.gamma_flip_level = 0

                # CORRECTED Call Wall Calculation
                if 'Exposure_Call' in st.session_state.calculated_df.columns and not st.session_state.calculated_df.empty:
                    otm_calls_df = st.session_state.calculated_df[st.session_state.calculated_df['Strike'] > st.session_state.market_price]
                    if not otm_calls_df.empty and otm_calls_df['Exposure_Call'].max() > 0 :
                        st.session_state.call_wall_strike = otm_calls_df.loc[otm_calls_df['Exposure_Call'].idxmax()]['Strike']
                    elif st.session_state.calculated_df['Exposure_Call'].max() > 0: # Fallback
                        st.session_state.call_wall_strike = st.session_state.calculated_df.loc[st.session_state.calculated_df['Exposure_Call'].idxmax()]['Strike']
                    else:
                        st.session_state.call_wall_strike = 0
                else:
                    st.session_state.call_wall_strike = 0

                # CORRECTED Put Wall Calculation
                if 'Exposure_Put' in st.session_state.calculated_df.columns and not st.session_state.calculated_df.empty:
                    otm_puts_df = st.session_state.calculated_df[st.session_state.calculated_df['Strike'] < st.session_state.market_price]
                    if not otm_puts_df.empty and otm_puts_df['Exposure_Put'].max() > 0:
                        st.session_state.put_wall_strike = otm_puts_df.loc[otm_puts_df['Exposure_Put'].idxmax()]['Strike']
                    elif st.session_state.calculated_df['Exposure_Put'].max() > 0: # Fallback
                        st.session_state.put_wall_strike = st.session_state.calculated_df.loc[st.session_state.calculated_df['Exposure_Put'].idxmax()]['Strike']
                    else:
                        st.session_state.put_wall_strike = 0
                else:
                    st.session_state.put_wall_strike = 0

                st.success("Analysis Complete!")
            else: 
                st.session_state.calculated_df = pd.DataFrame()
                st.session_state.gamma_flip_level = 0
                st.session_state.call_wall_strike = 0
                st.session_state.put_wall_strike = 0

tab1, tab2, tab3 = st.tabs(["📊 Charts & Key Metrics", "💡 Trading Insights & Recommendations", "📋 Detailed Data Table & Formulas"])

with tab1:
    st.header("2. Charts & Key Metrics")
    if not st.session_state.calculated_df.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Market Gamma", f"{st.session_state.total_gamma:.4f}")
        col2.metric("Total Put Gamma Exposure", f"{st.session_state.total_exposure_put:,.0f}")
        col3.metric("Total Call Gamma Exposure", f"{st.session_state.total_exposure_call:,.0f}")
        col4.metric("Net Market Gamma Exposure (Call - Put)", f"{st.session_state.net_market_exposure:,.0f}")
        st.markdown("---")
        st.subheader("Visual Analysis and Interpretation")

        st.markdown("""
        **Chart 1: Gamma Exposure at Each Strike (Call vs. Put)**
        - **What it shows:** This chart displays the calculated Gamma Exposure for call options (typically green) and put options (typically red) at each strike price. Gamma Exposure is a measure of how much dealers/market makers might need to buy or sell the underlying asset to re-hedge their positions if the price moves.
        - **How to interpret:**
            - **Tall Green Bars (Call Exposure):** Indicate strikes with significant Call option activity. These levels might act as **resistance**. If the price approaches these strikes, market makers who sold these calls may need to sell the underlying to hedge their positions, potentially pushing the price down.
            - **Tall Red Bars (Put Exposure):** Indicate strikes with significant Put option activity. These levels might act as **support**. If the price approaches these strikes, market makers who sold these puts may need to buy the underlying to hedge, potentially pushing the price up.
            - The taller the bar, the more significant the potential support or resistance from dealer hedging.
        """)
        fig_exposure_strike = go.Figure()
        fig_exposure_strike.add_trace(go.Bar(x=st.session_state.calculated_df['Strike'], y=st.session_state.calculated_df['Exposure_Call'], name='Call Exposure', marker_color='green'))
        fig_exposure_strike.add_trace(go.Bar(x=st.session_state.calculated_df['Strike'], y=st.session_state.calculated_df['Exposure_Put'], name='Put Exposure', marker_color='red'))
        fig_exposure_strike.update_layout(title_text='Gamma Exposure by Strike (Call vs. Put)', xaxis_title='Strike Price', yaxis_title='Gamma Exposure Amount', barmode='group')
        st.plotly_chart(fig_exposure_strike, use_container_width=True)

        st.markdown("""
        **Chart 2: Net Call Exposure at Each Strike (Call Exp. - Put Exp.)**
        - **What it shows:** This chart illustrates the difference between Call Gamma Exposure and Put Gamma Exposure at each strike.
        - **How to interpret:**
            - **Positive Bars:** Call Exposure is greater than Put Exposure at that strike. This implies dealers might be net short call gamma. These areas can act as strong resistance.
            - **Negative Bars:** Put Exposure is greater than Call Exposure. This implies dealers might be net short put gamma. These areas can act as strong support.
            - **Zero Line (or bars close to it):** This level is often called "Gamma Neutral" or is related to "Max Pain." It's where Call and Put exposures are relatively balanced. The market price might gravitate towards these levels, especially as options approach expiry, as this could be where option sellers (often market makers) have the least hedging pressure or potential maximum profit.
        """)
        fig_net_exp_strike = px.bar(st.session_state.calculated_df, x='Strike', y='Net_Call_Exposure_At_Strike', title='Net Call Exposure by Strike (Call Exp. - Put Exp.)', labels={'Net_Call_Exposure_At_Strike': 'Net Call Exposure Amount'}, color='Net_Call_Exposure_At_Strike', color_continuous_scale=px.colors.diverging.RdYlGn)
        fig_net_exp_strike.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_net_exp_strike, use_container_width=True)
        
        st.markdown("""
        **Chart 3: Open Interest (OI) Profile by Strike**
        - **What it shows:** The total number of outstanding (not yet settled) Call contracts (often light green) and Put contracts (often salmon/light red) at each strike.
        - **How to interpret:**
            - **High Call OI:** Indicates a strike with many open Call contracts. This often acts as **resistance**, as many participants (potentially sellers) want the price to stay below this level.
            - **High Put OI:** Indicates a strike with many open Put contracts. This often acts as **support**, as many participants (potentially sellers) want the price to stay above this level.
            - OI helps confirm the significance of strikes identified through gamma exposure.
        """)
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Bar(x=st.session_state.calculated_df['Strike'], y=st.session_state.calculated_df['OI_Call'], name='Call OI', marker_color='lightgreen'))
        fig_oi.add_trace(go.Bar(x=st.session_state.calculated_df['Strike'], y=st.session_state.calculated_df['OI_Put'], name='Put OI', marker_color='salmon'))
        fig_oi.update_layout(title_text='Open Interest (OI) by Strike', xaxis_title='Strike Price', yaxis_title='Open Interest (Number of Contracts)', barmode='group')
        st.plotly_chart(fig_oi, use_container_width=True)

        st.markdown("""
        **Chart 4: Gamma Value Profile by Strike**
        - **What it shows:** The actual Gamma value for Call options (often sky blue) and Put options (often orange) at each strike. Gamma measures how much an option's Delta (its price sensitivity to the underlying) is expected to change if the underlying price moves by 1 point.
        - **How to interpret:**
            - **High Gamma Values:** Mean the option's Delta is very sensitive to price changes. This is typically highest for At-The-Money (ATM) options.
            - **Impact on Dealer Hedging:** Market Makers who are short options with high gamma must adjust their hedges more frequently and aggressively. If they are net short gamma (a common scenario), their hedging activities can *amplify* market moves. For example, if they sold calls and the price rises, their delta becomes more negative, forcing them to buy more of the underlying to stay delta-neutral, which can push the price even higher.
            - **Volatility & Pinning:** High gamma areas can lead to increased volatility. Near expiry, very high gamma at a specific strike where market makers have large short positions can sometimes lead to "price pinning" at that strike.
        """)
        fig_gamma = go.Figure()
        fig_gamma.add_trace(go.Bar(x=st.session_state.calculated_df['Strike'], y=st.session_state.calculated_df['Gamma_Call'], name='Call Gamma Value', marker_color='skyblue'))
        fig_gamma.add_trace(go.Bar(x=st.session_state.calculated_df['Strike'], y=st.session_state.calculated_df['Gamma_Put'], name='Put Gamma Value', marker_color='orange'))
        fig_gamma.update_layout(title_text='Gamma Value by Strike', xaxis_title='Strike Price', yaxis_title='Gamma Value', barmode='group')
        st.plotly_chart(fig_gamma, use_container_width=True)
    else: st.info("Input data & click 'Analyze Option Chain' for charts.")

with tab2:
    st.header("3. Trading Insights & Recommendations")
    if not st.session_state.calculated_df.empty:
        cmp = st.session_state.market_price
        df = st.session_state.calculated_df
        net_market_exp_val = st.session_state.net_market_exposure
        total_gamma_val = st.session_state.total_gamma
        dte_val = st.session_state.dte
        gamma_flip = st.session_state.gamma_flip_level
        call_wall = st.session_state.call_wall_strike
        put_wall = st.session_state.put_wall_strike

        st.subheader("Overall Market Sentiment & Key Gamma Levels")
        sentiment_color = "green" if net_market_exp_val > 0 else "red" if net_market_exp_val < 0 else "gray"
        sentiment_text = "Potentially Bullish Tilt (Dealers Net Short Call Gamma)" if net_market_exp_val > 0 else "Potentially Bearish Tilt (Dealers Net Short Put Gamma)" if net_market_exp_val < 0 else "Neutral Dealer Positioning"
        
        st.markdown(f"**Current Market Price (CMP):** `{cmp:,.2f}`")
        st.markdown(f"**Net Market Gamma Exposure:** <font color='{sentiment_color}'>`{net_market_exp_val:,.0f}`</font> ({sentiment_text})", unsafe_allow_html=True)
        st.markdown(f"**Total Market Gamma:** `{total_gamma_val:.4f}`")
        st.markdown(f"**Identified Gamma Flip Level (approx.):** `{gamma_flip:,.0f}` (Strike where Net Call Exposure is near zero)")
        st.markdown(f"**Identified Call Wall (major resistance):** `{call_wall:,.0f}` (Highest OTM or nearest Call Exposure)")
        st.markdown(f"**Identified Put Wall (major support):** `{put_wall:,.0f}` (Highest OTM or nearest Put Exposure)")
        st.markdown(f"**Days To Expiry (DTE):** `{dte_val}`")

        st.markdown("---")
        st.markdown("**Interpreting Key Gamma Levels:**")
        st.markdown(f"- **Gamma Flip Level (`{gamma_flip:,.0f}`):** This is a crucial pivot. If CMP is **above** this level, dealers are often in 'positive gamma' (hedging dampens volatility). If CMP is **below** this level, dealers are often in 'negative gamma' (hedging amplifies volatility). Negative gamma regimes can lead to faster, more sustained moves, which option buyers might seek.")
        st.markdown(f"- **Call Wall (`{call_wall:,.0f}`):** A significant resistance area due to high Call OI and Exposure. A decisive break above could trigger buying from dealers covering short calls (gamma squeeze).")
        st.markdown(f"- **Put Wall (`{put_wall:,.0f}`):** A significant support area due to high Put OI and Exposure. A decisive break below could trigger selling from dealers hedging short puts (gamma squeeze).")
        
        num_strikes = len(df.index) if not df.empty else 1 
        if total_gamma_val > 0.005 * num_strikes : 
             st.markdown("A **High Total Market Gamma** suggests dealer hedging could **amplify price moves**. Volatility may increase, especially near ATM strikes. Risky for sellers, potentially good for buyers if directional.")
        elif total_gamma_val < 0.001 * num_strikes: 
             st.markdown("A **Low Total Market Gamma** suggests dealer hedging might have **less impact**. Price action may be more stable. Potentially favors sellers (lower premiums for buyers).")
        else:
             st.markdown("A **Moderate Total Market Gamma** suggests a balanced potential for volatility from dealer hedging.")
        
        # --- NAKED OPTION SELLING ---
        st.markdown("---")
        st.subheader(f"Naked Option Selling Opportunities")
        if all(col in df.columns for col in ['Strike', 'Exposure_Put', 'OI_Put', 'Gamma_Put']):
            put_candidates_df = df[(df['Strike'] < cmp) & (df['Strike'] != 0)] 
            if not put_candidates_df.empty:
                put_candidates = put_candidates_df.sort_values(by=['Exposure_Put', 'OI_Put'], ascending=[False, False]).head(N_RECOMMENDATIONS)
            else: put_candidates = pd.DataFrame()
        else:
            put_candidates = pd.DataFrame()
            st.warning("Cols for Put selling recommendations missing.")
        if not put_candidates.empty:
            for i, row in put_candidates.iterrows():
                strike = int(row['Strike']); put_oi = row['OI_Put']; put_exp = row['Exposure_Put']; put_gamma = row['Gamma_Put']
                st.markdown(f"**Consider Selling Naked Put at Strike: {strike}** (Price expected > {strike})")
                st.info(f"- **Reasoning:** OTM ({strike} < CMP {cmp:,.0f}). High Put OI ({put_oi:,.0f}) & Exposure ({put_exp:,.0f}) suggest support. Gamma: {put_gamma:.4f}. Theta decay ({dte_val} DTE) favorable. Best if Net Mkt Exp is bullish/neutral. **Risk:** Substantial loss if price drops below {strike}.")
        else: st.markdown("No strong OTM Put selling candidates found.")
        
        if all(col in df.columns for col in ['Strike', 'Exposure_Call', 'OI_Call', 'Gamma_Call']):
            call_candidates_df = df[(df['Strike'] > cmp) & (df['Strike'] != 0)]
            if not call_candidates_df.empty:
                call_candidates = call_candidates_df.sort_values(by=['Exposure_Call', 'OI_Call'], ascending=[False, False]).head(N_RECOMMENDATIONS)
            else: call_candidates = pd.DataFrame()
        else:
            call_candidates = pd.DataFrame()
            st.warning("Cols for Call selling recommendations missing.")
        if not call_candidates.empty:
            for i, row in call_candidates.iterrows():
                strike = int(row['Strike']); call_oi = row['OI_Call']; call_exp = row['Exposure_Call']; call_gamma = row['Gamma_Call']
                st.markdown(f"**Consider Selling Naked Call at Strike: {strike}** (Price expected < {strike})")
                st.warning(f"- **Reasoning:** OTM ({strike} > CMP {cmp:,.0f}). High Call OI ({call_oi:,.0f}) & Exposure ({call_exp:,.0f}) suggest resistance. Gamma: {call_gamma:.4f}. Theta decay ({dte_val} DTE) favorable. Best if Net Mkt Exp is bearish/neutral. **Risk:** Substantial loss if price rallies above {strike}.")
        else: st.markdown("No strong OTM Call selling candidates found.")

        # --- NAKED OPTION BUYING ---
        st.markdown("---")
        st.subheader(f"Naked Option Buying Opportunities")
        st.markdown("Option buyers are **long gamma** (benefit from large moves & increased volatility) but fight **theta decay** (time decay, especially with short DTE). Max loss is the premium paid.")

        st.markdown("**Potential Call Buys (Bullish Outlook):**")
        if 'Strike' in df.columns and call_wall != 0 and put_wall != 0: # Ensure df and walls are valid
            call_buy_candidates = df[(df['Strike'] >= cmp) & (df['Strike'] <= cmp * 1.05) & (df['Strike'] != 0) & (df['Gamma_Call'] > 0)] 
            if not call_buy_candidates.empty:
                # Heuristic: Prioritize strikes near walls or with high potential for squeeze
                call_buy_candidates_sorted_wall_break = call_buy_candidates[call_buy_candidates['Strike'] >= call_wall].sort_values(by='Gamma_Call', ascending=False)
                call_buy_candidates_sorted_wall_bounce = call_buy_candidates[call_buy_candidates['Strike'] >= put_wall].sort_values(by='Gamma_Call', ascending=False)
                
                combined_calls_list = []
                if not call_buy_candidates_sorted_wall_break.empty:
                    combined_calls_list.append(call_buy_candidates_sorted_wall_break.head(1)) # Top breakout candidate
                if not call_buy_candidates_sorted_wall_bounce.empty:
                     combined_calls_list.append(call_buy_candidates_sorted_wall_bounce.head(1)) # Top bounce candidate
                
                if combined_calls_list:
                    combined_calls = pd.concat(combined_calls_list).drop_duplicates().head(N_RECOMMENDATIONS)
                    for i, row in combined_calls.iterrows():
                        strike = int(row['Strike']); call_oi = row['OI_Call']; call_exp = row['Exposure_Call']; call_gamma_val = row['Gamma_Call']; net_exp_strike = row['Net_Call_Exposure_At_Strike']
                        reason = "General ATM/Slightly OTM call."
                        if strike >= call_wall and (strike - call_wall) < (cmp*0.02): # Breakout scenario
                            reason = f"Potential breakout play above Call Wall ({call_wall:,.0f}). High Call Exposure here could lead to squeeze if breached."
                        elif strike >= put_wall and (strike - put_wall) < (cmp*0.02) and strike < call_wall : # Bounce scenario
                            reason = f"Potential bounce play from Put Wall ({put_wall:,.0f})."
                        
                        st.markdown(f"**Consider Buying Call at Strike: {strike}**")
                        st.info(f"- **Scenario:** {reason}\n"
                                f"- **Strike Details:** Call OI ({call_oi:,.0f}), Call Exp ({call_exp:,.0f}), Gamma ({call_gamma_val:.4f}), Net Exp @Strike ({net_exp_strike:,.0f}).\n"
                                f"- **Context:** CMP relative to Gamma Flip (`{gamma_flip:,.0f}`). A move into positive dealer gamma (CMP > Flip) or squeeze if CMP < Flip (negative dealer gamma) could amplify gains. High Gamma means faster price change.\n"
                                f"- **Objective:** Profit from significant upward price movement. Requires price to move above {strike} + premium paid.\n"
                                f"- **Risk:** Max loss is premium paid. Theta decay is adverse."
                               )
                else:
                    st.markdown("No specific Call buying candidates matching wall-based scenarios currently.")
            else:
                st.markdown("No ATM/Slightly OTM Call options with positive gamma found for buying consideration.")
        else:
            st.markdown("Data for Call buying analysis is insufficient (e.g., key walls not identified).")


        st.markdown("**Potential Put Buys (Bearish Outlook):**")
        if 'Strike' in df.columns and call_wall != 0 and put_wall != 0:
            put_buy_candidates = df[(df['Strike'] <= cmp) & (df['Strike'] >= cmp * 0.95) & (df['Strike'] != 0) & (df['Gamma_Put'] > 0)]
            if not put_buy_candidates.empty:
                put_buy_candidates_sorted_wall_break = put_buy_candidates[put_buy_candidates['Strike'] <= put_wall].sort_values(by='Gamma_Put', ascending=False)
                put_buy_candidates_sorted_wall_rejection = put_buy_candidates[put_buy_candidates['Strike'] <= call_wall].sort_values(by='Gamma_Put', ascending=False)

                combined_puts_list = []
                if not put_buy_candidates_sorted_wall_break.empty:
                    combined_puts_list.append(put_buy_candidates_sorted_wall_break.head(1))
                if not put_buy_candidates_sorted_wall_rejection.empty:
                    combined_puts_list.append(put_buy_candidates_sorted_wall_rejection.head(1))

                if combined_puts_list:
                    combined_puts = pd.concat(combined_puts_list).drop_duplicates().head(N_RECOMMENDATIONS)
                    for i, row in combined_puts.iterrows():
                        strike = int(row['Strike']); put_oi = row['OI_Put']; put_exp = row['Exposure_Put']; put_gamma_val = row['Gamma_Put']; net_exp_strike = row['Net_Call_Exposure_At_Strike']
                        reason = "General ATM/Slightly OTM put."
                        if strike <= put_wall and (put_wall - strike) < (cmp*0.02): # Breakdown
                            reason = f"Potential breakdown play below Put Wall ({put_wall:,.0f}). High Put Exposure here could lead to squeeze if breached."
                        elif strike <= call_wall and (call_wall - strike) < (cmp*0.02) and strike > put_wall: # Rejection
                            reason = f"Potential rejection play from Call Wall ({call_wall:,.0f})."

                        st.markdown(f"**Consider Buying Put at Strike: {strike}**")
                        st.warning(f"- **Scenario:** {reason}\n"
                                   f"- **Strike Details:** Put OI ({put_oi:,.0f}), Put Exp ({put_exp:,.0f}), Gamma ({put_gamma_val:.4f}), Net Exp @Strike ({net_exp_strike:,.0f}).\n"
                                   f"- **Context:** CMP relative to Gamma Flip (`{gamma_flip:,.0f}`). A breakdown while in negative dealer gamma (CMP < Flip) could amplify decline. High Gamma means faster price change.\n"
                                   f"- **Objective:** Profit from significant downward price movement. Requires price to move below {strike} - premium paid.\n"
                                   f"- **Risk:** Max loss is premium paid. Theta decay is adverse."
                                  )
                else:
                    st.markdown("No specific Put buying candidates matching wall-based scenarios currently.")
            else:
                st.markdown("No ATM/Slightly OTM Put options with positive gamma found for buying consideration.")
        else:
            st.markdown("Data for Put buying analysis is insufficient (e.g., key walls not identified).")


        st.markdown("---")
        st.error("""⚠️ **IMPORTANT DISCLAIMER & RISK WARNING:** 
        The information, analyses, and potential trading ideas provided by this tool are for **informational and educational purposes ONLY**. They are generated based on the options data you provide and predefined formulas. They do **NOT** constitute financial advice, investment recommendations, or a solicitation to buy or sell any securities.
        - **Naked option selling involves substantial risk of loss.** Losses can significantly exceed the premium received and may not be suitable for all investors. Option buying has a maximum loss of the premium paid, but this can still be 100% of the capital allocated to that trade.
        - **Always conduct your own thorough research (DYOR).** Do not rely solely on this tool for making trading decisions.
        - **Consider all relevant factors:** This includes, but is not limited to, Implied Volatility (IV) levels, overall market conditions, economic news, company-specific events, liquidity of the options, and your personal financial situation and risk tolerance.
        - **Consult with a qualified financial advisor** if you are unsure about the risks or suitability of any trading strategy.
        """)
    else: st.info("Input data & click 'Analyze Option Chain' for insights.")

with tab3:
    st.header("4. Detailed Data Table & Formulas Used")
    if not st.session_state.calculated_df.empty:
        df_to_display = st.session_state.calculated_df.copy()
        display_columns_ordered_for_table = [
            'Strike',
            'Gamma_Call', 'OI_Lakh_Call', 'OI_Call', 'Exposure_Call', 'LTP_Call', 'Volume_Call', 'IV_Call',
            'Gamma_Put', 'OI_Lakh_Put', 'OI_Put', 'Exposure_Put', 'LTP_Put', 'Volume_Put',
            'Net_Call_Exposure_At_Strike'
        ]
        actual_display_columns = [col for col in display_columns_ordered_for_table if col in df_to_display.columns]
        if not actual_display_columns: st.warning("No relevant data columns for table display.")
        else:
            display_df = df_to_display[actual_display_columns]
            formatters = {}
            for col in ['OI_Put', 'OI_Call', 'Exposure_Put', 'Exposure_Call', 'Net_Call_Exposure_At_Strike', 'Volume_Call', 'Volume_Put']:
                if col in display_df.columns: formatters[col] = '{:,.0f}'
            for col in ['Gamma_Put', 'Gamma_Call']:
                if col in display_df.columns: formatters[col] = '{:.4f}'
            for col in ['LTP_Call', 'LTP_Put', 'IV_Call', 'OI_Lakh_Put', 'OI_Lakh_Call']:
                 if col in display_df.columns: formatters[col] = '{:.2f}' if display_df[col].dtype == float else '{}'
            st.dataframe(display_df.style.format(formatters), use_container_width=True)

        st.markdown("---")
        st.subheader("Formulas Reference:")
        st.markdown("""
        - **`OI_Put` / `OI_Call` (Actual Open Interest):** `OI_Lakh_Put * 100,000` or `OI_Lakh_Call * 100,000`.
        - **`Exposure_Put` / `Exposure_Call` (Gamma Exposure at a specific Strike):**
          `Actual_OI_at_Strike * (Current Market Price)² * Gamma_at_Strike * 0.01`
        - **`Net_Call_Exposure_At_Strike` (Net Call Gamma Exposure at a specific Strike):**
          `Exposure_Call_At_Strike - Exposure_Put_At_Strike`
        
        **Overall Market Metrics (Summarized at the top of Tab 1 & 2):**
        - **`Total Market Gamma`:** Sum of all `Gamma_Put` and `Gamma_Call` values.
        - **`Total Put/Call Gamma Exposure` (Overall Market):** Sum of individual strike exposures.
        - **`Net Market Gamma Exposure` (Overall Market):** `Total_Call_Gamma_Exposure - Total_Put_Gamma_Exposure`.
        - **`Gamma Flip Level`:** Strike where `Net_Call_Exposure_At_Strike` is closest to zero.
        - **`Call/Put Wall`:** Strike with highest OTM (or nearest available) Call/Put Gamma Exposure respectively.
        """)
        st.caption("Note: Gamma Exposure reflects potential dealer hedging. It's not a direct premium measure. Higher exposure often indicates levels dealers may defend.")
    else: st.info("No data processed yet.")