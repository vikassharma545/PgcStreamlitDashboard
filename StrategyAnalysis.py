import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

st.set_page_config(page_title="PNL Strategy Visualizer", layout="wide")
st.title("📈 Multi-Strategy PNL Dashboard")

uploaded_files = st.file_uploader("Upload up to 10 CSV or Parquet files", type=["csv", "parquet"], accept_multiple_files=True)

if uploaded_files:
    if len(uploaded_files) > 10:
        st.warning("Please upload up to 10 files only.")
    else:
        fig_cum = go.Figure()
        fig_dd = go.Figure()
        summary_data = []
        heatmap_frames = []
        combined_pnl = pd.DataFrame()

        for file in uploaded_files:
            name = file.name.replace(".csv", "").replace(".parquet", "")
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            elif file.name.endswith(".parquet"):
                df = pd.read_parquet(file)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values("Date").reset_index(drop=True)

            all_pnl_columns = [col for col in df.columns if col.endswith(".PNL")]

            selected_pnl_columns = st.multiselect(
                f"Select PNL columns for: {name}",
                options=all_pnl_columns,
                default=all_pnl_columns
            )

            for pnl_col in selected_pnl_columns:
                df['CumulativePNL'] = df[pnl_col].cumsum()
                df['PeakPNL'] = df['CumulativePNL'].cummax()
                df['Drawdown'] = df['CumulativePNL'] - df['PeakPNL']
                df['DrawdownPct'] = df['Drawdown'] / df['PeakPNL'].replace(0, 1) * 100
                df['IsWin'] = (df[pnl_col] > 0).astype(int)

                fig_cum.add_trace(go.Scatter(x=df['Date'], y=df['CumulativePNL'], mode='lines', name=f"{name} - {pnl_col}"))
                fig_dd.add_trace(go.Scatter(x=df['Date'], y=df['Drawdown'], mode='lines', name=f"{name} - {pnl_col}"))

                df_monthly = df.groupby(df['Date'].dt.to_period('M'))[pnl_col].sum().reset_index()
                df_monthly['Date'] = df_monthly['Date'].astype(str)
                df_monthly.set_index('Date', inplace=True)
                df_monthly.rename(columns={pnl_col: f"{name} - {pnl_col}"}, inplace=True)
                heatmap_frames.append(df_monthly)

                df_comb = df[['Date', pnl_col]].copy()
                df_comb = df_comb.rename(columns={pnl_col: f"{name} - {pnl_col}"})
                df_comb.set_index("Date", inplace=True)
                combined_pnl = pd.concat([combined_pnl, df_comb], axis=1)

                total_trades = len(df)
                avg_pnl = df[pnl_col].mean()
                median_pnl = df[pnl_col].median()
                std_pnl = df[pnl_col].std()
                win_rate = (df[pnl_col] > 0).mean() * 100
                loss_rate = (df[pnl_col] < 0).mean() * 100

                avg_win = df[df[pnl_col] > 0][pnl_col].mean()
                avg_loss = df[df[pnl_col] < 0][pnl_col].mean()
                win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan

                gross_profit = df[df[pnl_col] > 0][pnl_col].sum()
                gross_loss = df[df[pnl_col] < 0][pnl_col].sum()
                profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

                df['WinStreak'] = df['IsWin'] * (df['IsWin'].groupby((df['IsWin'] == 0).cumsum()).cumcount() + 1)
                df['LossStreak'] = (1 - df['IsWin']) * ((1 - df['IsWin']).groupby((df['IsWin'] == 1).cumsum()).cumcount() + 1)
                max_consec_win = df['WinStreak'].max()
                max_consec_loss = df['LossStreak'].max()

                max_drawdown = df['Drawdown'].min()
                max_drawdown_pct = df['DrawdownPct'].min()

                summary_data.append({
                    "File Name": name,
                    "PNL Column": pnl_col,
                    "Total Trades": total_trades,
                    "Average PNL": round(avg_pnl, 2),
                    "Median PNL": round(median_pnl, 2),
                    "Std Dev (Volatility)": round(std_pnl, 2),
                    "Win Rate (%)": round(win_rate, 2),
                    "Loss Rate (%)": round(loss_rate, 2),
                    "Win/Loss Ratio": round(win_loss_ratio, 2) if not np.isnan(win_loss_ratio) else 'N/A',
                    "Profit Factor": round(profit_factor, 2) if not np.isnan(profit_factor) else 'N/A',
                    "Max Consecutive Wins": int(max_consec_win),
                    "Max Consecutive Losses": int(max_consec_loss),
                    "Max Drawdown": round(max_drawdown, 2),
                    "Max Drawdown (%)": round(max_drawdown_pct, 2)
                })

        fig_cum.update_layout(title="Cumulative PNL Over Time", xaxis_title="Date", yaxis_title="Cumulative PNL")
        fig_dd.update_layout(title="Drawdown Over Time", xaxis_title="Date", yaxis_title="Drawdown")

        st.plotly_chart(fig_cum, use_container_width=True)
        st.plotly_chart(fig_dd, use_container_width=True)

        if heatmap_frames:
            combined_heatmap = pd.concat(heatmap_frames, axis=1).fillna(0)
            heatmap_data = combined_heatmap.T
            total_row = heatmap_data.sum().to_frame().T
            total_row.index = ['Combined Total']
            heatmap_data = pd.concat([heatmap_data, total_row])

            st.subheader("📊 Combined Monthly Returns Heatmap")
            st.plotly_chart(px.imshow(
                heatmap_data,
                text_auto=True,
                color_continuous_scale='RdYlGn',
                title="Combined Strategy Monthly Returns"
            ), use_container_width=True)

        st.subheader("📋 Strategy Summary Metrics")
        st.dataframe(pd.DataFrame(summary_data))

        if not combined_pnl.empty:
            combined_total = combined_pnl.fillna(0).sum(axis=1)
            combined_df = pd.DataFrame({"Date": combined_total.index, "PNL": combined_total.values})
            combined_df['CumulativePNL'] = combined_df['PNL'].cumsum()
            combined_df['PeakPNL'] = combined_df['CumulativePNL'].cummax()
            combined_df['Drawdown'] = combined_df['CumulativePNL'] - combined_df['PeakPNL']
            combined_df['IsWin'] = (combined_df['PNL'] > 0).astype(int)
            combined_df['WinStreak'] = combined_df['IsWin'] * (combined_df['IsWin'].groupby((combined_df['IsWin'] == 0).cumsum()).cumcount() + 1)
            combined_df['LossStreak'] = (1 - combined_df['IsWin']) * ((1 - combined_df['IsWin']).groupby((combined_df['IsWin'] == 1).cumsum()).cumcount() + 1)
            max_consec_win = combined_df['WinStreak'].max()
            max_consec_loss = combined_df['LossStreak'].max()

            win_rate = (combined_df['PNL'] > 0).mean() * 100
            loss_rate = (combined_df['PNL'] < 0).mean() * 100
            avg_win = combined_df[combined_df['PNL'] > 0]['PNL'].mean()
            avg_loss = combined_df[combined_df['PNL'] < 0]['PNL'].mean()
            win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.nan
            gross_profit = combined_df[combined_df['PNL'] > 0]['PNL'].sum()
            gross_loss = combined_df[combined_df['PNL'] < 0]['PNL'].sum()
            profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.nan

            avg_combined_pnl = combined_df['PNL'].mean()
            median_combined_pnl = combined_df['PNL'].median()
            std_combined_pnl = combined_df['PNL'].std()
            max_combined_drawdown = combined_df['Drawdown'].min()
            max_combined_drawdown_pct = (max_combined_drawdown / combined_df['PeakPNL'].replace(0, 1)).min() * 100

            st.subheader("📋 Combined Summary Metrics")
            st.table({
                "Average PNL": [round(avg_combined_pnl, 2)],
                "Median PNL": [round(median_combined_pnl, 2)],
                "Std Dev (Volatility)": [round(std_combined_pnl, 2)],
                "Win Rate (%)": [round(win_rate, 2)],
                "Loss Rate (%)": [round(loss_rate, 2)],
                "Win/Loss Ratio": [round(win_loss_ratio, 2) if not np.isnan(win_loss_ratio) else 'N/A'],
                "Profit Factor": [round(profit_factor, 2) if not np.isnan(profit_factor) else 'N/A'],
                "Max Consecutive Wins": [int(max_consec_win)],
                "Max Consecutive Losses": [int(max_consec_loss)],
                "Max Drawdown": [round(max_combined_drawdown, 2)],
                "Max Drawdown (%)": [round(max_combined_drawdown_pct, 2)]
            })

            fig_combined = go.Figure()
            fig_combined.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['CumulativePNL'], mode='lines', name="Combined PNL"))
            fig_combined.add_trace(go.Scatter(x=combined_df['Date'], y=combined_df['Drawdown'], mode='lines', name="Drawdown", line=dict(dash='dot')))
            fig_combined.update_layout(title="📈 Combined Cumulative PNL & Drawdown", xaxis_title="Date", yaxis_title="Value")
            st.plotly_chart(fig_combined, use_container_width=True)
