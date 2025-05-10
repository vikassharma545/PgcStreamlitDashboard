import os
import datetime
import pandas as pd
import polars as pl
from glob import glob
import streamlit as st
import concurrent.futures
import plotly.express as px
os.environ["POLARS_MAX_THREADS"] = str(max(1, round(os.cpu_count() * 0.7)))
pl.enable_string_cache()

dte_file = pd.read_csv(f"C:/PICKLE/DTE.csv", parse_dates=['Date'], dayfirst=True).set_index("Date")

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

@st.cache_data
def get_year_day_dte_files(parquet_files):
    year_day_dte_files = {}
    for file in parquet_files:
        index = file.split('\\')[-1].split(' ')[0]
        date = datetime.datetime.strptime(file.split('\\')[-1].split(' ')[1], "%Y-%m-%d")
        year = date.year
        day = date.strftime('%A')
        dte = dte_file.loc[date, index]
        year_day_dte_files[f'{index}-{year}-{day}-{dte}'] = year_day_dte_files.get(f'{index}-{year}-{day}-{dte}', []) + [file]

    return year_day_dte_files

st.set_page_config(page_title="PGC DashBoard", layout="wide",page_icon="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/icon.png")
st.image(image="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png", width=300)
if st.sidebar.button("Build By - Vikas Sharma", type="tertiary", icon=":material/thumb_up:"):
    st.sidebar.balloons()

folder_path = st.text_input(label="label", label_visibility="hidden", placeholder="Enter the folder path containing Parquet files")

if folder_path:
    
    if os.path.exists(folder_path):

        parquet_files = get_parquet_files(folder_path)

        if parquet_files:
            
            code, indices, name_columns, pnl_columns = get_code_index_cols(parquet_files)
            
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
                    
                    year_day_dte_files = get_year_day_dte_files(parquet_files)
                    
                    total_steps = len(year_day_dte_files)
                    progress_bar_count = 0
                    progress_bar = st.progress(progress_bar_count)
                    status_text = st.empty()
                    
                    dashboard_data_list = []
                    for key, value in year_day_dte_files.items():
                        index, year, day, dte = key.split('-')
                        
                        if ('dashboard_data' in st.session_state):
                            progress_bar_count += 1
                            progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                            continue
                        
                        chunks = sorted(set([f.split(' ')[-1].split('.')[0] for f in year_day_dte_files[key]]), key=lambda x: int(x.split('-')[-1]))
                        for chunk in chunks:

                            chunks_file = [f for f in year_day_dte_files[key] if f"{chunk}." in f]
                            
                            def read_and_cast(path):
                                df = pl.read_parquet(path, columns = (name_columns+pnl_columns))
                                return df.with_columns([pl.col(name_columns).cast(pl.Utf8).cast(pl.Categorical), pl.col(pnl_columns) .cast(pl.Float64)])

                            with concurrent.futures.ThreadPoolExecutor(max_workers=7) as exe:
                                dfs = list(exe.map(read_and_cast, chunks_file))

                            data = pl.concat(dfs)
                            data = data.group_by(name_columns).agg([pl.col(col).sum() for col in pnl_columns])
                            data = data.unpivot(index=name_columns, on=pnl_columns, variable_name='PL Basis', value_name='Points')
                            data.columns = [c.replace('P_','') for c in data.columns]

                            data = data.with_columns([
                                pl.lit(int(year)).cast(pl.Int16).alias("Year"),
                                pl.lit(day).cast(pl.String).alias("Day"),
                                pl.lit(int(float(dte))).cast(pl.Int8).alias("DTE")
                            ])
                            
                            dashboard_data_list.append(data)

                        progress_bar_count += 1
                        progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                        status_text.text(f"{index} {year} {day} {dte} {chunk} - Processing... Step {progress_bar_count} of {total_steps} ({(progress_bar_count) / total_steps * 100:.2f}%)")
                    
                    if ('dashboard_data' not in st.session_state):
                        dashboard_data = pl.concat(dashboard_data_list, how="vertical")
                        st.session_state.dashboard_data = dashboard_data
                    else:
                        dashboard_data = st.session_state.dashboard_data
                    
                    st.success("Initial data processing completed!")
                    
                    st.write("### Basic Information")
                    st.write(f"**Number of Rows**: {dashboard_data.shape[0]}")
                    st.write(f"**Number of Columns**: {dashboard_data.shape[1]}")
                    st.write(f"**Column Names**: {', '.join(dashboard_data.columns)}")
                    st.write(f"**Data Types**:\n{dashboard_data.dtypes}")
                    
                    # Show a preview of the dataset
                    st.write("### Data Preview")
                    st.write(dashboard_data.head())  # Display the first 5 rows
                    
                    # Show summary statistics
                    st.write("### Summary Statistics")
                    st.write(dashboard_data.describe())
                
                ###########################################################################################################################################
                
                # HeatMap Builder
                st.write("### HeatMap Builder 🔧")
                st.write(f"Data Size - {dashboard_data.shape}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    pivot_index = st.selectbox("Select Index", options=dashboard_data.columns, index=list(dashboard_data.columns).index('StartTime'))
                with col2:
                    pivot_column = st.selectbox("Select Column", options=dashboard_data.columns, index=list(dashboard_data.columns).index('EndTime'))
                with col3:
                    pivot_value = st.selectbox("Select Values", options=dashboard_data.columns, index=list(dashboard_data.columns).index('Points'))
                    
                if "unique_values" not in st.session_state:
                    unique_value_dict = {column:sorted(dashboard_data[column].unique()) for column in dashboard_data.columns if column not in ['Points']}
                    st.session_state['unique_values'] = unique_value_dict
                else:
                    unique_value_dict = st.session_state['unique_values']

                with st.sidebar:
                    st.title("🔧 Filters")
                    
                    filtered_exp = []
                    for column in dashboard_data.columns:
                        if column not in ['Strategy', 'StartTime', 'Points', pivot_index, pivot_column, pivot_value]:
                            unique_values = unique_value_dict[column]
                            if len(unique_values) > 1:
                                default_values = [unique_values[0]] if column not in ['Year', 'Day', 'DTE'] else unique_values
                                filter_values = st.segmented_control(f"***Select {column}***", options=unique_values, selection_mode="multi", default=default_values, key=f"seg_control_{column}")
                                filtered_exp.append(pl.col(column).is_in(filter_values))
        
                filtered_data = dashboard_data.filter(filtered_exp)
                filtered_data = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).sum())
                pivot = filtered_data.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(0)

                agg_func = 'sum'
                x_value = pivot.columns.get_level_values(1).astype(str)
                y_value = pivot.index.astype(str)
                
                fig = px.imshow(
                    pivot.values,
                    x=x_value,
                    y=y_value,
                    color_continuous_scale="RdYlGn",
                    text_auto=True,
                    labels={"color": f"{agg_func} of {pivot_value}"},
                    aspect="auto"
                )
                
                fig.update_layout(
                    title=f"{code} {pivot_index} vs {pivot_column} - {agg_func.upper()} of {pivot_value}",
                    xaxis_title=pivot_column,
                    yaxis_title=pivot_index,
                    autosize=True,
                    height=900,
                    font=dict(
                        family="Arial, sans-serif",
                        size=20,
                        color="black",
                        weight="bold"
                    ),
                    xaxis=dict(
                        tickangle=60,
                        tickmode="array",
                        tickvals=x_value,
                        ticktext=x_value, 
                        tickfont=dict(size=18),
                        type="category"
                    ),
                    yaxis=dict(
                        tickangle=0, 
                        tickmode="array", 
                        tickvals=y_value,
                        ticktext=y_value, 
                        tickfont=dict(size=18),
                        type="category"
                    ),
                )

                # Display the heatmap in Streamlit
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("No Parquet files found in the provided folder path.")
    else:
        st.error("⚠️ The specified folder could not be located. Please verify the path and try again.")
