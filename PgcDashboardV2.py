import os
import re
import datetime
import operator
import numpy as np
import pandas as pd
import polars as pl
from glob import glob
from numba import njit
import streamlit as st
import concurrent.futures
import plotly.express as px
from functools import reduce
os.environ["POLARS_MAX_THREADS"] = str(max(1, round(os.cpu_count() * 0.7)))
pl.enable_string_cache()

dte_file = pd.read_csv(f"C:/PICKLE/DTE.csv", parse_dates=['Date'], dayfirst=True).set_index("Date")

def sort_mixed_list(values):
    
    def parse_value(value):
        match = re.match(r"^(\d+\.\d+|\d+)([a-zA-Z]*)$", value)
        if match:
            if match.group(2) == "":
                return (0, float(match.group(1)), "")
            else:
                return (1, float(match.group(1)), match.group(2))
        else:
            return (2, float('inf'), value)
        
    return sorted(values, key=parse_value)

@st.cache_data
def get_parquet_files(folder_path):
    EXT = "*.parquet"
    return [file for path, subdir, files in os.walk(folder_path) for file in glob(os.path.join(path, EXT))]

@st.cache_data
def get_code_index_cols(parquet_files):
    code = parquet_files[0].split('\\')[-1].split(' ')[2]
    indices = sorted(set([f.split('\\')[-1].split(' ')[0] for f in parquet_files]))
    
    df = pd.read_parquet(max(parquet_files, key=lambda f: os.path.getsize(f)))
    name_columns = [c for c in list(df.columns) if c.startswith('P_')]
    pnl_columns = [c for c in list(df.columns) if c.endswith('PNL')]
    return code, indices, name_columns, pnl_columns

@njit(fastmath=True)
def calculate_metrics(mtm_values):
    
    n = len(mtm_values)
    total_sum, total_squared, win_sum, loss_sum, win_count, loss_count = 0, 0, 0, 0, 0, 0

    # Variables for drawdown
    peak = float('-inf')    
    max_dd, max_dd_days, current_dd_days, cumulative_pnl = 0, 0, 0, 0
    current_drawdown, drawdown_sums, drawdown_count = 0, 0, 0
    
    # Main loop for calculation
    for value in mtm_values:
        total_sum += value
        total_squared += value * value
        
        # Win/Loss aggregation
        if value > 0:
            win_sum += value
            win_count += 1
        elif value < 0:
            loss_sum += value
            loss_count += 1
        
        # Update cumulative PNL
        cumulative_pnl += value

        # Peak and Drawdown Calculation
        if cumulative_pnl > peak:
            # If a drawdown was active, close it and record
            if current_drawdown > 0:
                drawdown_sums += current_drawdown
                drawdown_count += 1

            # Reset drawdown variables
            peak = cumulative_pnl
            current_drawdown = 0.0
            current_dd_days = 0
        else:
            # Update drawdown
            current_drawdown = peak - cumulative_pnl

            # Update the max drawdown if this is the largest seen
            if current_drawdown > max_dd:
                max_dd = current_drawdown

            # Track the number of days in drawdown
            current_dd_days += 1
            if current_dd_days > max_dd_days:
                max_dd_days = current_dd_days

    # Final drawdown collection if it was not reset
    if current_drawdown > 0:
        drawdown_sums += current_drawdown
        drawdown_count += 1

    # Calculating Metrics
    avg_pnl = total_sum / n if n > 0 else 0.0
    std_dev = np.sqrt((total_squared / n) - (avg_pnl ** 2)) if n > 0 else 0.0
    win_rate = (win_count / n) * 100 if n > 0 else 0.0
    win_loss_ratio = (win_count / loss_count) if loss_count > 0 else np.nan
    profit_factor = (win_sum / abs(loss_sum)) if loss_sum < 0 else np.nan

    # Average Drawdown Calculation
    avg_max_dd = drawdown_sums / drawdown_count if drawdown_count > 0 else 0.0

    return (total_sum, avg_pnl, std_dev, win_rate, win_loss_ratio, profit_factor, avg_max_dd, max_dd, max_dd_days)

st.set_page_config(page_title="PGC DashBoard", layout="wide",page_icon="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/icon.png")
st.image(image = "https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png", width=300)
if st.sidebar.button("⭐Build By- Vikas Sharma", type="tertiary", icon=":material/thumb_up:"):
    st.sidebar.balloons()

folder_path = st.text_input(label="label", label_visibility="hidden", placeholder="Enter the folder path containing Parquet files")

if folder_path:
    
    if os.path.exists(folder_path):

        parquet_files = get_parquet_files(folder_path)

        if parquet_files:
            
            code, indices, name_columns, pnl_columns = get_code_index_cols(parquet_files)
            total_chunk_nos = sorted(set([k.split('-')[-1].split('.')[0] for k in glob(f'{folder_path}/*')]))
            
            with st.expander("Uploded Files details", expanded=True):
                st.write(f"**Total File Uploaded**: {len(parquet_files)}")
                st.write(f"**Code**: {code}")
                st.write(f"**Indices**: {indices}")
                st.write(f"**Parameter cols**: {', '.join(name_columns)}")
                st.write(f"**PNL cols**: {', '.join(pnl_columns)}")
                
            if 'button_clicked' not in st.session_state:
                st.session_state['button_clicked'] = st.button("Run Processing")
            else:
                st.session_state['button_clicked'] = True

            if st.session_state['button_clicked'] or ('dashboard_data' in st.session_state):

                with st.expander("DashBoard Files details", expanded=True):
                    st.markdown("### Building DashBoard files ...")
                    
                    total_steps = len(indices) * len(total_chunk_nos)
                    progress_bar_count = 0
                    progress_bar = st.progress(progress_bar_count)
                    status_text = st.empty()
                    status_text2 = st.empty()
                    
                    dashboard_data_list = []
                    for index in indices:
                        
                        chunk_nos = sorted(set([k.split('-')[-1].split('.')[0] for k in glob(f'{folder_path}/{index}*')]))
                        for chunk_no in chunk_nos:
                            
                            if ('dashboard_data' in st.session_state):
                                progress_bar_count += 1
                                progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                                continue
                            
                            all_files = glob(f'{folder_path}/{index}*No-{chunk_no}.parquet')
                            if len(all_files) == 0:continue
                            status_text.text(f"{index} Chunks-{chunk_no} Files-{len(all_files)} - Processing... Step {progress_bar_count+1} of {total_steps} ({(progress_bar_count) / total_steps * 100:.2f}%)")

                            status_text2.text(f"Reading Chunks...")
                            def read_and_cast(path):
                                df = pl.read_parquet(path, columns = (name_columns + ['Date', 'DTE'] + pnl_columns))
                                return df.with_columns([pl.col(name_columns).cast(pl.Utf8).cast(pl.Categorical), pl.col('Date').cast(pl.Date), pl.col('DTE').cast(pl.Int8), pl.col(pnl_columns).cast(pl.Float64)])

                            with concurrent.futures.ThreadPoolExecutor(max_workers=7) as exe:
                                dfs = list(exe.map(read_and_cast, all_files))

                            df = pl.concat(dfs)
                            df.columns = [c.replace('P_','') for c in df.columns]
                            dashboard_data_list.append(df)
                            
                            progress_bar_count += 1
                            progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                                
                    if ('dashboard_data' not in st.session_state):
                        dashboard_data = pl.concat(dashboard_data_list, how="vertical")
                        st.session_state.dashboard_data = dashboard_data
                    else:
                        dashboard_data = st.session_state.dashboard_data
                        
                    name_columns = [c.replace('P_','') for c in name_columns]
                    
                    st.success("Initial data processing completed!")
                    
                    st.write("### Basic Information")
                    st.write(f"**Number of Rows**: {dashboard_data.shape[0]}")
                    st.write(f"**Number of Columns**: {dashboard_data.shape[1]}")
                    st.write(f"**Column Names**: {', '.join(dashboard_data.columns)}")
                    st.write(f"**Data Types**:\n{dashboard_data.dtypes}")
                    
                    # Show a preview of the dataset
                    st.write("### Data Preview")
                    st.write(dashboard_data.head())  # Display the first 5 rows
                
                ###########################################################################################################################################
                
                # HeatMap Builder
                st.write("### HeatMap Builder 🔧")
                st.write(f"Data Size - {dashboard_data.shape}")
                
                analysis_matrices = ['MTM', 'Average PNL', 'Std Dev (Volatility)', 'Win Rate (%)', 'Win/Loss Ratio', 'Profit Factor', 'Avg Drawdown', 'Max Drawdown', 'Max DD Days']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    pivot_index = st.selectbox("Select Index", options=dashboard_data.columns, index=list(dashboard_data.columns).index('StartTime'), key="select_index")
                with col2:
                    pivot_column = st.selectbox("Select Column", options=dashboard_data.columns, index=list(dashboard_data.columns).index('EndTime'), key='select_pivot_col')
                with col3:
                    filt_pnl_columns = st.multiselect("Select PNL Cols", options=pnl_columns, default="Total.PNL" if "Total.PNL" in pnl_columns else None, key='select_pnl_cols')
                    
                col4, col5, col6 = st.columns(3)
                with col4:
                    start_date = st.date_input("Start Date", datetime.date(2022,1,1), key="start_date")
                with col5:
                    end_date = st.date_input("End Date", "today", key="end_date")
                with col6:
                    dtes = st.segmented_control(f"***Select DTE***", options=sorted(dashboard_data['DTE'].unique()), selection_mode="multi", default=[1], key="DTE")
                    
                if "unique_values" not in st.session_state:
                    unique_value_dict = {column:sorted(dashboard_data[column].unique()) for column in dashboard_data.columns if not (column in ['Date', 'DTE'] + pnl_columns)}
                    st.session_state['unique_values'] = unique_value_dict
                else:
                    unique_value_dict = st.session_state['unique_values']
                    
                filtered_exp = [pl.col('DTE').is_in(dtes), pl.col("Date").is_between(start_date, end_date, closed="both")]

                with st.sidebar:
                    st.title(f"🔧 Filters - {code}")
                    
                    for column in dashboard_data.columns:
                        if not column in ['Strategy', 'StartTime', 'Date', 'DTE', pivot_index, pivot_column] + pnl_columns:
                            unique_values = unique_value_dict[column]
                            if len(unique_values) > 1:
                                default_values = [unique_values[0]] if column not in ['Year', 'Day', 'DTE'] else unique_values
                                filter_values = st.segmented_control(f"***Select {column}***", options=unique_values, selection_mode="multi", default=default_values, key=f"seg_control_{column}")
                                filtered_exp.append(pl.col(column).is_in(filter_values))

                filtered_data = dashboard_data.filter(filtered_exp)
                filtered_data = filtered_data.sort('Date')
                filtered_data = filtered_data.group_by(name_columns).all()
                sum_expr = reduce(operator.add, (pl.col(c) for c in filt_pnl_columns)).alias("Combined_PNL")
                filtered_data = filtered_data.with_columns([sum_expr])
                filtered_data = filtered_data.with_columns(pl.col("Combined_PNL").map_elements(lambda x: calculate_metrics(x.to_numpy()), return_dtype=pl.List(pl.Float64)).alias("metrics"))
                filtered_data = filtered_data.with_columns([pl.col("metrics").list.get(idx).alias(col) for idx, col in enumerate(analysis_matrices)])
                
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(analysis_matrices)

                with tab1:
                    
                    agg_func = 'Sum'
                    pivot_value = analysis_matrices[0]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).sum())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(0)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale="RdYlGn", text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab2:
                    
                    agg_func = 'Mean'
                    pivot_value = analysis_matrices[1]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).mean())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(2)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale="RdYlGn", text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab3:
                    
                    agg_func = 'Mean'
                    pivot_value = analysis_matrices[2]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).mean())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(2)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale="RdYlGn", text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab4:
                    
                    agg_func = 'Mean'
                    pivot_value = analysis_matrices[3]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).mean())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(2)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale="RdYlGn", text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab5:
                    
                    agg_func = 'Mean'
                    pivot_value = analysis_matrices[4]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).mean())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(2)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale="RdYlGn", text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab6:
                    
                    agg_func = 'Mean'
                    pivot_value = analysis_matrices[5]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).mean())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(2)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale="RdYlGn", text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab7:
                    
                    agg_func = 'Mean'
                    pivot_value = analysis_matrices[6]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).mean())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(2)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale=px.colors.diverging.RdYlGn[::-1], text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab8:
                    
                    agg_func = 'Max'
                    pivot_value = analysis_matrices[7]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).max())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(0)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale=px.colors.diverging.RdYlGn[::-1], text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
                    
                with tab9:
                    
                    agg_func = 'Max'
                    pivot_value = analysis_matrices[8]
                    st.header(f"{pivot_value}-{agg_func}")
                    
                    pivot = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).max())
                    pivot = pivot.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(0)
                    
                    pivot = pivot.reindex(sort_mixed_list(pivot.index))
                    pivot.columns = [x[1] for x in pivot.columns]
                    pivot = pivot[sort_mixed_list(pivot.columns)]
                    
                    x_value, y_value = pivot.columns.astype(str), pivot.index.astype(str)
                    fig = px.imshow( pivot.values, x=x_value, y=y_value, color_continuous_scale=px.colors.diverging.RdYlGn[::-1], text_auto=True,labels={"color": f"{agg_func} of {pivot_value}"}, aspect="auto")
                    fig.update_layout(title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}", xaxis_title=pivot_column, yaxis_title=pivot_index, autosize=True, height=900,
                        font=dict( family="Comic Sans MS, Arial, sans-serif", size=20, color="black", weight="bold" ),
                        xaxis=dict( tickangle=60, tickmode="array", tickvals=x_value, ticktext=x_value, tickfont=dict(size=18), type="category"),
                        yaxis=dict(tickangle=0, tickmode="array", tickvals=y_value,ticktext=y_value, tickfont=dict(size=18),type="category"),)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No Parquet files found in the provided folder path.")
    else:
        st.error("⚠️ The specified folder could not be located. Please verify the path and try again.")
