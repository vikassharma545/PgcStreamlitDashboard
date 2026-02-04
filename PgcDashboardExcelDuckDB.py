import os
import re
import duckdb
import pickle
import tempfile
import pandas as pd
import polars as pl
import xlwings as xw
import streamlit as st
from pathlib import Path
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
def get_code_index_cols(dashboard_metadata):    
    code_type = dashboard_metadata['CodeType']
    code = dashboard_metadata['Strategy']
    indices = sorted(dashboard_metadata['Index'])
    name_columns = sorted(dashboard_metadata.keys() - {'CodeType', 'Dates', 'Strategy', 'Points'})
    pnl_columns = sort_mixed_list(dashboard_metadata['PL Basis'])
    return code_type, code, indices, name_columns, pnl_columns

@st.cache_data
def mapping_dashboard_files(parquet_files, code_type):
    
    data = []
    if code_type == 'Intraday':
        columns = ['Index', 'Year', 'Day', 'DTE', 'PL Basis', 'FilePath']
        
        for file in parquet_files:
            parts = file.stem.split('-')
            # Extract specific parts: Year (1), Day (2), DTE (3), PL Basis (4)
            data.append([file.parts[-2], int(parts[1]), parts[2], float(parts[3]), parts[4], file.as_posix()])
            
    elif code_type == 'Weekly':
        columns = ['Index', 'Year', 'Start.DTE-End.DTE', 'PL Basis', 'FilePath']
        
        for file in parquet_files:
            parts = file.stem.split('-')
            # Extract specific parts: Year (1), Start.DTE-End.DTE (2), PL Basis (3)
            data.append([file.parts[-2], int(parts[1]), parts[2], parts[3], file.as_posix()])

    return pd.DataFrame(data, columns=columns)

def load_and_filter_data(filtered_parquet_files, filter_conditions, top_level_filter_col):
    """function to load and filter parquet data"""
    import time
    start_time = time.time()
    
    if not filtered_parquet_files:
        return pl.DataFrame(), 0
    
    # Create a fresh DuckDB connection for each query
    conn = duckdb.connect(':memory:')
    
    conditions = []
    for col, vals in filter_conditions.items():
        if vals and col not in top_level_filter_col:
            # Format values for SQL: strings get quotes, numbers don't
            val_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in vals])
            conditions.append(f'"{col}" IN ({val_str})')
    
    where_sql = " AND ".join(conditions) if conditions else "1=1"
    
    # Use the connection object and materialize the result immediately
    query = f"SELECT * FROM read_parquet({filtered_parquet_files}) WHERE {where_sql}"
    filtered_data = conn.execute(query).pl()
    
    # Close the connection after use
    conn.close()
    
    elapsed_time = time.time() - start_time
    return filtered_data, elapsed_time

st.set_page_config(page_title="PGC DashBoard", layout="wide",page_icon="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/icon.png")
st.image(image = "https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png", width=100)
            
def select_folder_callback():
    folder = select_folder_gui("Select Folder containing Parquet files")
    if folder:
        st.session_state["folder_path"] = str(folder)

st.button("Select DashBoard Folder", type="primary", on_click=select_folder_callback, key="select_folder_button")
if "selected_folder" in st.session_state:
    st.success(f"Selected folder: {st.session_state['selected_folder']}")

if "folder_path" in st.session_state:
    
    folder_path = st.session_state["folder_path"]
    
    if os.path.exists(folder_path):

        parquet_files = get_parquet_files(folder_path)
        dashboard_metadata = pickle.load(open(Path(folder_path) / "Metadata.pickle", "rb"))

        if parquet_files:
            
            code_type, code, indices, name_columns, pnl_columns = get_code_index_cols(dashboard_metadata)
            mapping_dashboard_files_df = mapping_dashboard_files(parquet_files, code_type) 
    
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

            if st.session_state['button_clicked']:
                
                # HeatMap Builder
                st.write("### HeatMap Builder 🔧")
                
                col1, col2 = st.columns(2)
                with col1:
                    pivot_index = st.selectbox("Select HeatMap Index", options=name_columns, index=name_columns.index('StartTime'))
                with col2:
                    pivot_column = st.selectbox("Select HeatMap Column", options=name_columns, index=name_columns.index('EndTime'))

                st.title("🔧 Filters")
                filtered_dict = {}
                for column in name_columns:
                    if column not in [pivot_index, pivot_column]:
                        unique_values = sort_mixed_list(dashboard_metadata[column])
                        if len(unique_values) > 1:
                            
                            if code_type == 'Intraday':
                                default_values = [unique_values[0]] if column not in ['Year', 'Day', 'DTE'] else unique_values
                            elif code_type == 'Weekly':
                                default_values = [unique_values[0]] if column not in ['Year', 'Start.DTE-End.DTE'] else unique_values

                            filter_values = st.segmented_control(f"***Select {column}***", options=unique_values, selection_mode="multi", default=default_values, key=f"seg_control_{column}")
                            filtered_dict[column] = filter_values
                
                # Add a manual "Apply Filters" button to prevent automatic reruns
                apply_filters = st.button("🔄 Apply Filters & Generate Heatmap", type="primary", use_container_width=True)
                
                # Only process when button is clicked
                if apply_filters:
                    
                    if code_type == 'Intraday':
                        top_level_filter_col = ['Index', 'Year', 'Day', 'DTE', 'PL Basis']
                    elif code_type == 'Weekly':
                        top_level_filter_col = ['Index', 'Year', 'Start.DTE-End.DTE', 'PL Basis']
                        
                    temp_df = mapping_dashboard_files_df.copy()
                    for column in name_columns:
                        if column in top_level_filter_col and column in filtered_dict:
                            temp_df = temp_df[temp_df[column].isin(filtered_dict[column])]
                            
                    filtered_parquet_files = temp_df['FilePath'].tolist()
                    st.write(f"**Total Files after applying filters**: {len(filtered_parquet_files)}")
                    
                    # Use cached function to load data
                    filtered_data, load_time = load_and_filter_data(filtered_parquet_files, filtered_dict, top_level_filter_col)
                    
                    # Display timing information
                    st.success(f"✅ Data loaded in **{load_time:.2f} seconds** ({len(filtered_data):,} rows)")
                    
                    if len(filtered_data) == 0:
                        st.warning("No data found with the selected filters.")
                        st.stop()
                        
                    filtered_data = filtered_data.group_by([pivot_index, pivot_column]).agg(pl.col("Points").sum())
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
                        merged_filters = filtered_dict.copy()

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
                    st.info("👆 Adjust your filters above and click 'Apply Filters & Generate Heatmap' to update the results")

        else:
            st.warning("No Parquet files found in the provided folder path.")
    else:
        st.error("⚠️ The specified folder could not be located. Please verify the path and try again.")