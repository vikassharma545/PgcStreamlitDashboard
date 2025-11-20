import os
import re
import sys
import tempfile
import datetime
import pandas as pd
import polars as pl
import xlwings as xw
import streamlit as st
from pathlib import Path
import concurrent.futures
import plotly.express as px
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
        match = re.match(r"^(-?\d+\.\d+|-?\d+)([.a-zA-Z%]*)$", value)
        if match:
            if match.group(2) == "":
                return (0, "", float(match.group(1)))
            else:
                return (1, match.group(2), float(match.group(1)))
        else:
            return (2, value, float('inf'))
        
    return sorted(values, key=parse_value)

@st.cache_data
def get_parquet_files(folder_path):
    root = Path(folder_path).expanduser().resolve()
    iterator = root.rglob("*.parquet")
    return sorted(iterator)

@st.cache_data
def get_code_index_cols(parquet_files):
    code = parquet_files[0].stem.split()[2]
    indices = sorted(set([f.stem.split()[0] for f in parquet_files]))
    
    df = pd.read_parquet(parquet_files[0])
    name_columns = [c for c in list(df.columns) if c.startswith('P_')]
    pnl_columns = [c for c in list(df.columns) if c.endswith('PNL')]
    return code, indices, name_columns, pnl_columns

@st.cache_data
def get_year_day_dte_files(parquet_files, dte_file):
    year_day_dte_files = {}
    for file in parquet_files:
        index = file.stem.split()[0]
        date = datetime.datetime.strptime(file.stem.split()[1], "%Y-%m-%d")
        year = date.year
        day = date.strftime('%A')
        dte = dte_file.loc[date, index]
        year_day_dte_files[f'{index}|{year}|{day}|{dte}'] = year_day_dte_files.get(f'{index}|{year}|{day}|{dte}', []) + [file]

    return year_day_dte_files

st.set_page_config(page_title="PGC DashBoard", layout="wide",page_icon="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/icon.png")
st.image(image = "https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png", width=100)

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
                    
                    year_day_dte_files = get_year_day_dte_files(parquet_files, dte_file)
                    
                    total_steps = len(year_day_dte_files)
                    progress_bar_count = 0
                    progress_bar = st.progress(progress_bar_count)
                    status_text = st.empty()
                    
                    dashboard_data_list = []
                    for key, value in year_day_dte_files.items():
                        index, year, day, dte = key.split('|')
                        
                        if ('dashboard_data' in st.session_state):
                            progress_bar_count += 1
                            progress_bar.progress(int((progress_bar_count) / total_steps * 100))
                            continue
                        
                        chunks = sorted(set([f.stem.split()[-1] for f in year_day_dte_files[key]]), key=lambda x: int(x.split('-')[-1]))
                        for chunk in chunks:

                            chunks_file = [f for f in year_day_dte_files[key] if f.stem.split()[-1] == chunk]
                            
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

                st.title("🔧 Filters")
                filtered_exp = []
                filtered_dict = []
                for column in dashboard_data.columns:
                    if column not in ['Strategy', 'Points', pivot_index, pivot_column, pivot_value]:
                        unique_values = unique_value_dict[column]
                        if len(unique_values) > 1:
                            default_values = [unique_values[0]] if column not in ['Year', 'Day', 'DTE'] else unique_values
                            filter_values = st.segmented_control(f"***Select {column}***", options=unique_values, selection_mode="multi", default=default_values, key=f"seg_control_{column}")
                            filtered_exp.append(pl.col(column).is_in(filter_values))
                            filtered_dict.append({column: filter_values})
        
                filtered_data = dashboard_data.filter(filtered_exp)
                filtered_data = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col(pivot_value).sum())
                pivot = filtered_data.to_pandas().set_index([pivot_index, pivot_column]).unstack(fill_value=0).round(0)
                
                pivot = pivot.reindex(sort_mixed_list(pivot.index))
                pivot.columns = [x[1] for x in pivot.columns]
                pivot = pivot[sort_mixed_list(pivot.columns)]

                agg_func = 'sum'
                x_value = pivot.columns.astype(str)
                y_value = pivot.index.astype(str)
                
                file_name = f"{code}.xlsx"
                file_path = Path(tempfile.gettempdir()) / f"{code}.xlsx"
                wb = None

                # --- ATTEMPT 1: Connect to Existing Open File ---
                try:
                    wb = xw.books[file_name]
                except Exception:
                    pass # Just move to the next step

                # --- ATTEMPT 2: Open File (Standard) ---
                if wb is None:
                    try:
                        if os.path.exists(file_path):
                            wb = xw.Book(file_path)
                        else:
                            wb = xw.Book()
                            wb.save(file_path)
                    except Exception:
                        # --- ATTEMPT 3: The Fix for "Unknown name" / COM Errors ---
                        # If standard opening fails, force a NEW Excel Instance
                        try:
                            # visible=True ensures you see the new window
                            new_app = xw.App(visible=True) 
                            if os.path.exists(file_path):
                                wb = new_app.books.open(file_path)
                            else:
                                wb = new_app.books.add()
                                wb.save(file_path)
                        except Exception as e:
                            st.error(f"❌ Fatal Excel Error: Unable to start Excel. Please close all Excel instances and try again. Error: {e}")

                # --- PROCEED IF WORKBOOK EXISTS ---
                if wb:
                    sheet_name = "dashboard"
                    try:
                        sheet = wb.sheets[sheet_name]
                    except:
                        sheet = wb.sheets.add(sheet_name)

                    sheet.clear()

                    # Data Preparation for Excel
                    df_styled = pivot.copy()
                    df_styled['Grand Total'] = df_styled.sum(axis=1)
                    sum_row = df_styled.sum(axis=0)
                    df_styled.loc['Grand Total'] = sum_row

                    # ==========================================
                    # 2. WRITE TO EXCEL
                    # ==========================================
                    
                    b1_cell = sheet.range("B1")
                    b1_cell.value = pivot_column
                    b1_cell.api.Font.Bold = True
                    b1_cell.color = (220, 230, 241) # Light Blue
                    b1_cell.api.Borders.LineStyle = 1

                    start_cell = sheet.range("A2")
                    start_cell.value = df_styled

                    # ==========================================
                    # 3. APPLY STYLING (The "Look")
                    # ==========================================
                    # Get the full range of the inserted table
                    full_tbl = start_cell.expand()
                    last_row = full_tbl.last_cell.row
                    last_col = full_tbl.last_cell.column
                    
                    # Top Header Row
                    header_rng = sheet.range((start_cell.row, start_cell.column), (start_cell.row, last_col))
                    header_rng.api.Font.Bold = True
                    header_rng.color = (220, 230, 241) # Light Blue
                    header_rng.api.Borders(8).LineStyle = 1  # xlEdgeTop
                    header_rng.api.Borders(9).LineStyle = 1 # Add top border to total row (xlEdgeBottom=9)
                    
                    # Left Index Column
                    index_rng = sheet.range((start_cell.row + 1, start_cell.column), (last_row, start_cell.column))
                    index_rng.api.Font.Bold = True
                    index_rng.api.Borders(10).LineStyle = 1 # 10 = xlEdgeRight

                    # Bottom Row (Grand Total)
                    bottom_rng = sheet.range((last_row, start_cell.column), (last_row, last_col))
                    bottom_rng.api.Font.Bold = True
                    bottom_rng.color = (220, 230, 241) # Light Blue
                    bottom_rng.api.Borders(8).LineStyle = 1  # xlEdgeTop
                    bottom_rng.api.Borders(9).LineStyle = 1 # Add top border to total row (xlEdgeBottom=9)

                    # Right Column (Grand Total)
                    right_rng = sheet.range((start_cell.row, last_col), (last_row, last_col))
                    right_rng.api.Font.Bold = True
                    right_rng.api.Borders(7).LineStyle = 1 # Add top border to total row (xlEdgeBottom=7)
                    right_rng.api.Borders(10).LineStyle = 1 # Add top border to total row (xlEdgeBottom=10)
                    
                    # Data Body with Right and Bottom Borders
                    data_rng = sheet.range((start_cell.row + 1, start_cell.column + 1), (last_row, last_col))
                    data_rng.number_format = "#,##0" # Number Format (Commas, no decimals: 1,234)

                    # Data Body (The numbers inside Only)
                    data_rng = sheet.range((start_cell.row + 1, start_cell.column + 1), (last_row-1, last_col-1))
                    data_rng.api.FormatConditions.Delete()
                    cs = data_rng.api.FormatConditions.AddColorScale(3) # APPLY HEATMAP (Conditional Formatting)
                    
                    # ==========================================
                    # 4. PARAMETER TABLE (Filters Applied)
                    merged_filters = {}
                    for item in filtered_dict:
                        merged_filters.update(item)

                    # 2. Create DataFrame (Handling different list lengths)
                    if merged_filters:
                        # Using pd.Series aligns columns of different lengths (e.g., 1 Year vs 5 DTEs)
                        param_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in merged_filters.items()]))
                        param_df = param_df.fillna("")  # Replace NaNs with empty strings

                        # 3. Determine Position (2 columns to the right of the Heatmap)
                        # 'last_col' was calculated earlier in Section 3
                        param_col_idx = last_col + 2
                        param_anchor = sheet.range((start_cell.row, param_col_idx))
                        
                        # 4. Write to Excel
                        param_anchor.options(index=False).value = param_df
                        
                        # 5. Style the Parameter Table
                        n_rows = param_df.shape[0] + 1  # +1 for the Header row
                        n_cols = param_df.shape[1]
                        
                        # Define the table range using exact coordinates
                        # (Start Row, Start Col) to (End Row, End Col)
                        param_tbl = sheet.range(
                            (param_anchor.row, param_anchor.column),
                            (param_anchor.row + n_rows - 1, param_anchor.column + n_cols - 1)
                        )
                                            
                        # A. Headers (Blue & Bold)
                        # Select just the top row of the new table
                        p_headers = sheet.range((param_anchor.row, param_anchor.column), (param_anchor.row, param_tbl.last_cell.column))
                        p_headers.api.Font.Bold = True
                        p_headers.color = (255, 235, 156) 
                        p_headers.api.Borders.LineStyle = 1 
                        
                        param_tbl.api.Borders.LineStyle = 1 
                        param_tbl.api.HorizontalAlignment = -4108 # xlCenter
                        
                    # ==========================================
                    # 6. FINAL CLEANUP
                    # ==========================================
                    # Autofit columns for BOTH the heatmap and the new filter table
                    sheet.used_range.columns.autofit()
                    st.toast("✅ Excel Updated with Heatmap & Filter Parameters!")

        else:
            st.warning("No Parquet files found in the provided folder path.")
    else:
        st.error("⚠️ The specified folder could not be located. Please verify the path and try again.")
