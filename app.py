import datetime
import pandas as pd
import polars as pl
import streamlit as st
import plotly.express as px

dte_file = pd.read_csv(f"C:/PICKLE/DTE.csv", parse_dates=['Date'], dayfirst=True).set_index("Date")

st.set_page_config(page_title="PGC DashBoard",layout="wide")
st.title("📊 PGC Dashboard")

uploaded_files = st.file_uploader("Upload Parquet files", type=["parquet"], accept_multiple_files=True)

if st.button("Run Processing") or ('dashboard_data' in st.session_state):
    if uploaded_files:
        
        code = uploaded_files[0].name.split()[2]
        indices = sorted(set([f.name.split()[0] for f in uploaded_files]))
        uploaded_files_count = len(uploaded_files)
        
        df = pd.read_parquet(max(uploaded_files, key=lambda f: f.size))
        name_columns = [c for c in list(df.columns) if c.startswith('P_')]
        pnl_columns = [c for c in list(df.columns) if c.endswith('PNL')]
    
        year_day_dte_files = {}
        for file in uploaded_files:
            index = file.name.split()[0]
            date = datetime.datetime.strptime(file.name.split()[1], "%Y-%m-%d")
            year = date.year
            day = date.strftime('%A')
            dte = dte_file.loc[date, index]
            year_day_dte_files[f'{index}-{year}-{day}-{dte}'] = year_day_dte_files.get(f'{index}-{year}-{day}-{dte}', []) + [file]
        
        with st.expander("Uploaded Files details", expanded=True):
            st.write(f"**Total File Uploaded**: {uploaded_files_count}")
            st.write(f"**Code**: {code}")
            st.write(f"**Indices**: {indices}")
            st.write(f"**Name cols**: {', '.join(name_columns)}")
            st.write(f"**PNL cols**: {', '.join(pnl_columns)}")
            
            st.markdown("### Building DashBoard files ...")
            
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
                
                chunks = sorted(set([f.name.split()[-1].split('.')[0] for f in year_day_dte_files[key]]), key=lambda x: int(x.split('-')[-1]))
                for chunk in chunks:

                    chunks_file = [f for f in year_day_dte_files[key] if f"{chunk}." in f.name]
                    data = pl.read_parquet(chunks_file, columns=(name_columns+pnl_columns), use_pyarrow=True)
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
            
            st.session_state.dashboard_data = dashboard_data
            
            
        ############################################
        # HeatMap Builder
        
        filtered_data = dashboard_data.clone()
        st.write(f"Actual Size - {filtered_data.shape}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pivot_index = st.selectbox("Select Index", options=filtered_data.columns, index=list(filtered_data.columns).index('StartTime'))
        with col2:
            pivot_column = st.selectbox("Select Column", options=filtered_data.columns, index=list(filtered_data.columns).index('SL'))
        with col3:
            pivot_value = st.selectbox("Select Values", options=filtered_data.columns, index=list(filtered_data.columns).index('Points'))
        
        agg_func = 'sum'
        
        with st.sidebar:
            st.title("🔧 Filters")
            
            for column in filtered_data.columns:
                if column not in ['Strategy', 'StartTime', 'Points', pivot_index, pivot_column, pivot_value]:
                    unique_values = sorted(filtered_data[column].unique())
                    if len(unique_values) > 1:
                        filter_values = st.segmented_control(f"***Select {column}***", options=unique_values, selection_mode="multi", default=[unique_values[0]])
                        filtered_data = filtered_data.filter(pl.col(column).is_in(filter_values))
                    
        filtered_data = filtered_data.to_pandas()
        
        pivot = pd.pivot_table(
            filtered_data,
            index=pivot_index,
            columns=pivot_column,
            values=pivot_value,
            aggfunc='sum',
            fill_value=0
        ).round(0)
        
        fig = px.imshow(
            pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
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
                tickvals=pivot.columns.astype(str),
                ticktext=pivot.columns.astype(str), 
                tickfont=dict(size=18),
                type="category"
            ),
            yaxis=dict(
                tickangle=0, 
                tickmode="array", 
                tickvals=pivot.index.astype(str),
                ticktext=pivot.index.astype(str), 
                tickfont=dict(size=18),
                type="category"
            ),
        )

        # Display the heatmap in Streamlit
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No File were uploaded !!!")