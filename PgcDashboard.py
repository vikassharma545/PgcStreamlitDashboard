import os
import re
import datetime
import pandas as pd
import polars as pl
import streamlit as st
from pathlib import Path
import concurrent.futures
import plotly.express as px
from natsort import natsorted
from tkinter import Tk, filedialog

pl.enable_string_cache()

def select_folder_gui(title="Select a Folder") -> Path | None:
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True) 
    folder_path = filedialog.askdirectory(title=title, parent=root)
    root.destroy() 
    return Path(folder_path) if folder_path else None

def select_file_gui(title="Select a File", filetypes=None) -> Path | None:
    if filetypes is None:
        filetypes = [("All files", "*.*")]

    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Ensure the dialog appears on top
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=root)
    root.destroy()
    return Path(file_path) if file_path else None

def sort_mixed_list(values):
    
    def parse_value(value):
        val_str = str(value)
        match = re.match(r"^(-?\d+\.\d+|-?\d+)([.a-zA-Z%]*)$", val_str)
        if match:
            if match.group(2) == "":
                return (0, "", float(match.group(1)))
            else:
                return (1, match.group(2), float(match.group(1)))
        else:
            return (2, val_str, float('inf'))
        
    return natsorted(values, key=parse_value)

@st.cache_data
def get_parquet_files(folder_path):
    root = Path(folder_path).expanduser().resolve()
    iterator = root.rglob("*.parquet")
    return sorted(iterator)

@st.cache_data
def get_code_index_cols(parquet_files):

    parquet_file_path = max(parquet_files, key=lambda f: os.path.getsize(f))
    df = pd.read_parquet(parquet_file_path)
    splits = parquet_file_path.stem.split()

    if len(splits) == 4: # Intraday
        code_type = "Intraday"
        index, date, code, chunk = splits
    elif len(splits) == 6: # Weekly
        code_type = "Weekly"
        index, start_date, end_date, dte, code, chunk = splits
    
    indices = sorted(set([f.stem.split()[0] for f in parquet_files]))
    name_columns = [c for c in list(df.columns) if c.startswith('P_')]
    pnl_columns = [c for c in list(df.columns) if c.endswith('PNL')]
    return code_type, code, indices, name_columns, pnl_columns

@st.cache_data
def grouping_parquet_files(parquet_files, dte_file, code_type="Intraday/Weekly"):

    if code_type == "Intraday":

        year_day_dte_files = {}
        for file in parquet_files:
            index = file.stem.split()[0]
            date = datetime.datetime.strptime(file.stem.split()[1], "%Y-%m-%d")
            year = date.year
            day = date.strftime('%A')
            dte = dte_file.loc[date, index]
            year_day_dte_files[f'{index}-{year}-{day}-{dte}'] = year_day_dte_files.get(f'{index}-{year}-{day}-{dte}', []) + [file]

        return year_day_dte_files

    elif code_type == "Weekly":

        year_dte_files = {}
        for file in parquet_files:
            index = file.stem.split()[0]
            date = datetime.datetime.strptime(file.stem.split()[1], "%Y-%m-%d")
            year = date.year
            dte = file.stem.split()[3].replace('-', '_')
            year_dte_files[f'{index}-{year}-{dte}'] = year_dte_files.get(f'{index}-{year}-{dte}', []) + [file]

        return year_dte_files
    
    else:
        raise ValueError("Invalid code_type. Must be 'Intraday' or 'Weekly'.")

st.set_page_config(page_title="PGC DashBoard", layout="wide",page_icon="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/icon.png")
st.image(image = "https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png", width=300)
if st.sidebar.button("⭐Build By- Vikas Sharma", type="tertiary", icon=":material/thumb_up:"):
    st.sidebar.balloons()

def select_dte_file_callback():
    file = select_file_gui("Select DTE File", filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
    if file:
        st.session_state["dte_file_path"] = str(file)
            
def select_folder_callback():
    folder = select_folder_gui("Select Folder containing Parquet files")
    if folder:
        st.session_state["folder_path"] = str(folder)

st.button("Select DTE File", type="primary", on_click=select_dte_file_callback, key="select_dte_file_button")
if "dte_file_path" in st.session_state:
    st.success(f"Selected DTE file: {st.session_state['dte_file_path']}")
    dte_file_path = st.session_state["dte_file_path"]
    dte_file = pd.read_csv(dte_file_path, parse_dates=['Date'], dayfirst=True).set_index("Date")

st.button("Select Parquet Folder", type="primary", on_click=select_folder_callback, key="select_folder_button")
if "selected_folder" in st.session_state:
    st.success(f"Selected folder: {st.session_state['selected_folder']}")

if "dte_file_path" in st.session_state and "folder_path" in st.session_state:
    
    folder_path = st.session_state["folder_path"]
    
    if os.path.exists(folder_path):

        parquet_files = get_parquet_files(folder_path)

        if parquet_files:
            
            code_type, code, indices, name_columns, pnl_columns = get_code_index_cols(parquet_files)
            
            with st.expander("Uploded Files details", expanded=True):
                st.write(f"**Total File Uploaded**: {len(parquet_files)}")
                st.write(f"**Code Type**: {code_type}")
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
                    
                    grouped_parquet = grouping_parquet_files(parquet_files, dte_file, code_type=code_type)
                    
                    total_steps = len(grouped_parquet)
                    progress_bar_count = 0
                    progress_bar = st.progress(progress_bar_count)
                    status_text = st.empty()
                    
                    dashboard_data_list = []
                    for key, value in grouped_parquet.items():
                        
                        if code_type == 'Intraday':
                            index, year, day, dte = key.split('-')
                        elif code_type == 'Weekly':
                            index, year, dte = key.split('-')
                        
                        if ('dashboard_data' in st.session_state):
                            progress_bar_count += 1
                            progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                            continue
                        
                        chunks = sorted(set([f.stem.split()[-1] for f in grouped_parquet[key]]), key=lambda x: int(x.split('-')[-1]))
                        for chunk in chunks:

                            chunks_file = [f for f in grouped_parquet[key] if f.stem.split()[-1] == chunk]
                            
                            def read_and_cast(path):
                                df = pl.read_parquet(path, columns = (name_columns+pnl_columns))
                                return df.with_columns([pl.col(name_columns).cast(pl.Utf8).cast(pl.Categorical), pl.col(pnl_columns).cast(pl.Float64)])

                            with concurrent.futures.ThreadPoolExecutor() as exe:
                                dfs = list(exe.map(read_and_cast, chunks_file))

                            data = pl.concat(dfs)
                            data = data.group_by(name_columns).agg([pl.col(col).sum() for col in pnl_columns])
                            data = data.unpivot(index=name_columns, on=pnl_columns, variable_name='PL Basis', value_name='Points')
                            data.columns = [c.replace('P_','', 1) if c.startswith('P_') else c for c in data.columns]
                            
                            data = data.with_columns([
                                pl.col("PL Basis").cast(pl.Categorical).alias("PL Basis")
                            ])

                            if code_type == 'Intraday':
                                data = data.with_columns([
                                    pl.lit(int(year)).cast(pl.Int16).alias("Year"),
                                    pl.lit(day).cast(pl.Categorical).alias("Day"),
                                    pl.lit(int(float(dte))).cast(pl.Int8).alias("DTE")
                                ])
                            elif code_type == 'Weekly':
                                data = data.with_columns([
                                    pl.lit(int(year)).cast(pl.Int16).alias("Year"),
                                    pl.lit(dte).cast(pl.Categorical).alias("Start.DTE-End.DTE")
                                ])
                            
                            dashboard_data_list.append(data)

                        progress_bar_count += 1
                        progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                        
                        if code_type == 'Intraday':
                            status_text.text(f"{index} {year} {day} {dte} - Processing... Step {progress_bar_count} of {total_steps} ({(progress_bar_count) / total_steps * 100:.2f}%)")
                        elif code_type == 'Weekly':
                            status_text.text(f"{index} {year} {dte} - Processing... Step {progress_bar_count} of {total_steps} ({(progress_bar_count) / total_steps * 100:.2f}%)")

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
                    unique_value_dict = {column:sort_mixed_list(dashboard_data[column].unique()) for column in dashboard_data.columns if column not in ['Strategy', 'Points']}
                    st.session_state['unique_values'] = unique_value_dict
                else:
                    unique_value_dict = st.session_state['unique_values']

                with st.sidebar:
                    st.title("🔧 Filters")
                    
                    filtered_exp = []
                    for column in dashboard_data.columns:
                        if column not in ['Strategy', 'Points', pivot_index, pivot_column, pivot_value]:
                            unique_values = unique_value_dict[column]
                            if len(unique_values) > 1:
                                
                                if code_type == 'Intraday':
                                    default_values = [unique_values[0]] if column not in ['Year', 'Day', 'DTE'] else unique_values
                                elif code_type == 'Weekly':
                                    default_values = [unique_values[0]] if column not in ['Year', 'Start.DTE-End.DTE'] else unique_values

                                filter_values = st.segmented_control(f"***Select {column}***", options=unique_values, selection_mode="multi", default=default_values, key=f"seg_control_{column}")
                                filtered_exp.append(pl.col(column).is_in(filter_values))
        
                filtered_data = dashboard_data.filter(filtered_exp)
                filtered_data = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).sum())
                pivot = filtered_data.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(0)
                
                pivot = pivot.reindex(sort_mixed_list(pivot.index))
                pivot.columns = [x[1] for x in pivot.columns]
                pivot = pivot[sort_mixed_list(pivot.columns)]

                agg_func = 'sum'
                x_value = pivot.columns.astype(str)
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
                        family="Comic Sans MS, Arial, sans-serif",
                        size=20
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
                st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No Parquet files found in the provided folder path.")
    else:
        st.error("⚠️ The specified folder could not be located. Please verify the path and try again.")
