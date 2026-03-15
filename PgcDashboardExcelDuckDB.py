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
    root.attributes('-topmost', True)
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
            parts = file.stem.replace('--', '-@').split('-')
            parts = [p.replace('@', '-') for p in parts]
            data.append([file.parts[-2], int(parts[1]), parts[2], float(parts[3]), parts[4], file.as_posix()])
    elif code_type == 'Weekly':
        columns = ['Index', 'Year', 'Start.DTE-End.DTE', 'PL Basis', 'FilePath']
        for file in parquet_files:
            parts = file.stem.replace('--', '-@').split('-')
            parts = [p.replace('@', '-') for p in parts]
            data.append([file.parts[-2], int(parts[1]), parts[2], parts[3], file.as_posix()])
    return pd.DataFrame(data, columns=columns)

def load_and_filter_data(filtered_parquet_files, filter_conditions, top_level_filter_col):
    import time
    start_time = time.time()
    if not filtered_parquet_files:
        return pl.DataFrame(), 0
    
    conn = duckdb.connect(':memory:')
    conditions = []
    for col, vals in filter_conditions.items():
        if vals and col not in top_level_filter_col:
            val_str = ", ".join([f"'{v}'" if isinstance(v, str) else str(v) for v in vals])
            conditions.append(f'"{col}" IN ({val_str})')
            
    where_sql = " AND ".join(conditions) if conditions else "1=1"
    query = f"SELECT * FROM read_parquet({filtered_parquet_files}) WHERE {where_sql}"
    filtered_data = conn.execute(query).pl()
    conn.close()
    
    elapsed_time = time.time() - start_time
    return filtered_data, elapsed_time

# ─────────────────────────────────────────────
#  Page Config & Theme-Resilient CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="PGC DashBoard", layout="wide", initial_sidebar_state="collapsed",
    page_icon="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/icon.png"
)

st.markdown("""<style>
/* Hide Sidebar Toggle safely */
[data-testid="collapsedControl"] { display: none; }
.block-container { padding: 3rem 2rem 1.5rem 2rem; max-width: 100%; }

/* ── Header Grouping Block ── */
.dashboard-header-block {
    border: 2px solid #4472c4;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
    background: color-mix(in srgb, var(--secondary-background-color) 20%, transparent);
    box-shadow: 0 4px 12px rgba(68, 114, 196, 0.15);
}

/* ── Top bar ── */
.topbar {
    background: color-mix(in srgb, #1f3864 85%, var(--background-color));
    color: #ffffff;
    padding: 12px 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.topbar-left { display: flex; align-items: center; gap: 15px; }
.topbar h2 { margin: 0; font-size: 1.5rem; color: #ffffff; line-height: 1; }
.topbar .info { font-size: 0.9rem; opacity: 0.85; line-height: 1; }

/* ── Context Bar (Path) ── */
.context-bar {
    background: var(--secondary-background-color);
    padding: 8px 12px;
    border-radius: 4px;
    margin-top: -8px; 
    margin-bottom: 16px;
    font-size: 0.85rem;
    color: var(--text-color);
    display: flex;
    align-items: center;
    gap: 8px;
    border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
}
.context-label {
    font-weight: 700;
    color: #4472c4;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.context-val-path {
    font-family: monospace;
    opacity: 0.85;
    word-break: break-all; 
}

/* ── Stat cells (Flexbox wrapped & gap optimized) ── */
.stat-row {
    display: flex;
    flex-wrap: wrap; 
    gap: 1px; 
    border: 1px solid var(--secondary-background-color);
    border-radius: 6px;
    overflow: hidden;
    background: var(--secondary-background-color); 
    box-shadow: 0 1px 2px rgba(0,0,0,0.05);
}
.stat-cell {
    flex: 1;
    min-width: 150px; 
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 12px 8px;
    background: var(--background-color);
}
.stat-cell .sv { 
    font-size: 1.4rem; 
    font-weight: 700; 
    color: var(--text-color); 
    margin-top: 4px; 
    line-height: 1; 
    text-align: center;
}
.stat-cell .sl { 
    font-size: 0.7rem; 
    text-transform: uppercase; 
    color: #4472c4; 
    font-weight: 700; 
    letter-spacing: 0.5px; 
    line-height: 1; 
    text-align: center;
}

/* ── Excel-style slicer box ── */
.slicer-box {
    border: 1px solid var(--secondary-background-color);
    border-radius: 6px;
    background: var(--background-color);
    margin-bottom: 12px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.slicer-header {
    background: #4472c4;
    color: #ffffff;
    font-size: 0.85rem;
    font-weight: 600;
    padding: 6px 12px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.slicer-body {
    padding: 4px; 
}

/* ── Status bar (Theme Resilient) ── */
.sb { 
    display: flex; 
    align-items: center; 
    background: color-mix(in srgb, #548235 15%, var(--background-color)); 
    border-left: 4px solid #548235; 
    padding: 8px 16px; 
    font-size: 0.9rem; 
    color: var(--text-color); 
    margin: 8px 0; 
    border-radius: 4px; 
}
.sb-blue { 
    background: color-mix(in srgb, #4472c4 15%, var(--background-color)); 
    border-left-color: #4472c4; 
}
</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Folder Selection
# ─────────────────────────────────────────────

def select_folder_callback():
    folder = select_folder_gui("Select Folder containing Parquet files")
    if folder:
        st.session_state["folder_path"] = str(folder)

if "folder_path" not in st.session_state:
    st.markdown("""
<div style="text-align:center;padding:60px 0">
<img src="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png" width="120" style="margin-bottom: 20px;">
<h2 style="color:var(--text-color); margin-bottom: 5px;">PGC Dashboard</h2>
<p style="color:#6b7280; margin-bottom: 25px;">Select a dashboard folder to begin</p>
</div>
""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.button("📂 Select DashBoard Folder", type="primary", on_click=select_folder_callback, use_container_width=True)
    st.stop()

folder_path = st.session_state["folder_path"]
if not os.path.exists(folder_path):
    st.error("⚠️ The specified folder could not be located. Please verify the path and try again.")
    st.button("📂 Select DashBoard Folder", type="primary", on_click=select_folder_callback)
    st.stop()

parquet_files = get_parquet_files(folder_path)
dashboard_metadata = pickle.load(open(Path(folder_path) / "Metadata.pickle", "rb"))
if not parquet_files:
    st.warning("No Parquet files found in the provided folder path.")
    st.stop()

code_type, code, indices, name_columns, pnl_columns = get_code_index_cols(dashboard_metadata)
mapping_dashboard_files_df = mapping_dashboard_files(parquet_files, code_type)

# ─────────────────────────────────────────────
#  Top Bar, Context Bar & Stats (Grouped into Header Panel)
# ─────────────────────────────────────────────

st.markdown(f"""
<div class="dashboard-header-block">
<div class="topbar">
<div class="topbar-left">
<img src="https://raw.githubusercontent.com/vikassharma545/PgcStreamlitDashboard/main/img/logo.png" width="45" style="border-radius: 4px;">
<h2>{code}</h2>
</div>
<div class="info">{code_type} &nbsp;|&nbsp; {len(indices)} Indices &nbsp;|&nbsp; {len(parquet_files):,} Files</div>
</div>
<div class="context-bar">
<span class="context-label">📁 Source Path:</span> 
<span class="context-val-path">{folder_path}</span>
</div>
<div class="stat-row">
<div class="stat-cell"><div class="sl">Strategy Code</div><div class="sv">{code}</div></div>
<div class="stat-cell"><div class="sl">Total Files</div><div class="sv">{len(parquet_files):,}</div></div>
<div class="stat-cell"><div class="sl">Indices</div><div class="sv">{', '.join(indices)}</div></div>
<div class="stat-cell"><div class="sl">Parameter Cols</div><div class="sv">{', '.join(name_columns)}</div></div>
<div class="stat-cell"><div class="sl">PNL Cols</div><div class="sv">{', '.join(pnl_columns)}</div></div>
</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  HeatMap Builder Axis Selection
# ─────────────────────────────────────────────

st.write("### HeatMap Builder 🔧")

# Removed the button, changed to 2 columns
ax1, ax2 = st.columns(2, vertical_alignment="bottom")
with ax1:
    st.markdown('<div class="slicer-box"><div class="slicer-header">Select HeatMap Index</div><div class="slicer-body">', unsafe_allow_html=True)
    pivot_index = st.selectbox("Row", options=name_columns, index=name_columns.index('StartTime'), label_visibility="collapsed")
    st.markdown('</div></div>', unsafe_allow_html=True)
with ax2:
    st.markdown('<div class="slicer-box"><div class="slicer-header">Select HeatMap Column</div><div class="slicer-body">', unsafe_allow_html=True)
    pivot_column = st.selectbox("Col", options=name_columns, index=name_columns.index('EndTime'), label_visibility="collapsed")
    st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Filters (Excel slicer boxes)
# ─────────────────────────────────────────────

filtered_dict = {}
filter_columns = [c for c in name_columns if c not in [pivot_index, pivot_column]]
filter_data = {}
for column in filter_columns:
    unique_values = sort_mixed_list(dashboard_metadata[column])
    if len(unique_values) > 1:
        if code_type == 'Intraday':
            default_values = [unique_values[0]] if column not in ['Year', 'Day', 'DTE'] else unique_values
        elif code_type == 'Weekly':
            default_values = [unique_values[0]] if column not in ['Year', 'Start.DTE-End.DTE'] else unique_values
        filter_data[column] = (unique_values, default_values)

filter_keys = list(filter_data.keys())
cols_per_row = 3
for row_start in range(0, len(filter_keys), cols_per_row):
    row_keys = filter_keys[row_start:row_start + cols_per_row]
    cols = st.columns(cols_per_row)
    for i, key in enumerate(row_keys):
        unique_values, default_values = filter_data[key]
        with cols[i]:
            st.markdown(f'<div class="slicer-box"><div class="slicer-header">{key}</div><div class="slicer-body">', unsafe_allow_html=True)
            val = st.segmented_control(key, options=unique_values, selection_mode="multi", default=default_values, key=f"seg_control_{key}", label_visibility="collapsed")
            filtered_dict[key] = val
            st.markdown('</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  Results & Excel Output (Instantly Executes)
# ─────────────────────────────────────────────

if code_type == 'Intraday':
    top_level_filter_col = ['Index', 'Year', 'Day', 'DTE', 'PL Basis']
elif code_type == 'Weekly':
    top_level_filter_col = ['Index', 'Year', 'Start.DTE-End.DTE', 'PL Basis']
    
temp_df = mapping_dashboard_files_df.copy()
for column in name_columns:
    if column in top_level_filter_col and column in filtered_dict:
        temp_df = temp_df[temp_df[column].isin(filtered_dict[column])]
        
filtered_parquet_files = temp_df['FilePath'].tolist()

r1, r2 = st.columns(2)
with r1:
    st.markdown(f'<div class="sb sb-blue">📁 &nbsp;<b>Total Files matching filters:</b> {len(filtered_parquet_files)}</div>', unsafe_allow_html=True)

with st.spinner("Calculating..."):
    filtered_data, load_time = load_and_filter_data(filtered_parquet_files, filtered_dict, top_level_filter_col)

with r2:
    st.markdown(f'<div class="sb">✅ &nbsp;Data loaded in <b>{load_time:.2f} seconds</b> ({len(filtered_data):,} rows)</div>', unsafe_allow_html=True)

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
    pass 

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
        try:
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
    except Exception:
        sheet = wb.sheets.add(sheet_name)

    sheet.clear()

    df_styled = pivot.copy()
    df_styled['Grand Total'] = df_styled.sum(axis=1)
    sum_row = df_styled.sum(axis=0)
    df_styled.loc['Grand Total'] = sum_row
    
    b1_cell = sheet.range("B1")
    b1_cell.value = pivot_column
    b1_cell.api.Font.Bold = True
    b1_cell.color = (220, 230, 241)
    b1_cell.api.Borders.LineStyle = 1

    start_cell = sheet.range("A2")
    start_cell.value = df_styled

    full_tbl = start_cell.expand()
    last_row = full_tbl.last_cell.row
    last_col = full_tbl.last_cell.column
    
    header_rng = sheet.range((start_cell.row, start_cell.column), (start_cell.row, last_col))
    header_rng.api.Font.Bold = True
    header_rng.color = (220, 230, 241)
    header_rng.api.Borders(8).LineStyle = 1  
    header_rng.api.Borders(9).LineStyle = 1 
    
    index_rng = sheet.range((start_cell.row + 1, start_cell.column), (last_row, start_cell.column))
    index_rng.api.Font.Bold = True
    index_rng.api.Borders(10).LineStyle = 1 

    bottom_rng = sheet.range((last_row, start_cell.column), (last_row, last_col))
    bottom_rng.api.Font.Bold = True
    bottom_rng.color = (220, 230, 241)
    bottom_rng.api.Borders(8).LineStyle = 1  
    bottom_rng.api.Borders(9).LineStyle = 1 

    right_rng = sheet.range((start_cell.row, last_col), (last_row, last_col))
    right_rng.api.Font.Bold = True
    right_rng.api.Borders(7).LineStyle = 1 
    right_rng.api.Borders(10).LineStyle = 1 
    
    data_rng = sheet.range((start_cell.row + 1, start_cell.column + 1), (last_row, last_col))
    data_rng.number_format = "#,##0" 

    data_rng = sheet.range((start_cell.row + 1, start_cell.column + 1), (last_row-1, last_col-1))
    data_rng.api.FormatConditions.Delete()
    cs = data_rng.api.FormatConditions.AddColorScale(3) 
    
    merged_filters = filtered_dict.copy()

    if merged_filters:
        param_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in merged_filters.items()]))
        param_df = param_df.fillna("")  

        param_col_idx = last_col + 2
        param_anchor = sheet.range((start_cell.row, param_col_idx))
        param_anchor.options(index=False).value = param_df
        
        n_rows = param_df.shape[0] + 1  
        n_cols = param_df.shape[1]
        
        param_tbl = sheet.range(
            (param_anchor.row, param_anchor.column),
            (param_anchor.row + n_rows - 1, param_anchor.column + n_cols - 1)
        )
                            
        p_headers = sheet.range((param_anchor.row, param_anchor.column), (param_anchor.row, param_tbl.last_cell.column))
        p_headers.api.Font.Bold = True
        p_headers.color = (255, 235, 156) 
        p_headers.api.Borders.LineStyle = 1 
        
        param_tbl.api.Borders.LineStyle = 1 
        param_tbl.api.HorizontalAlignment = -4108 
        
    sheet.used_range.columns.autofit()
    st.toast("✅ Excel Updated with Heatmap & Filter Parameters!")