import os
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
import warnings

warnings.filterwarnings("ignore")

def empty_csv_files():
    try:
        with open('waiting_times.csv', 'w') as f:
            f.truncate(0)
        with open('people_count_intervals.csv', 'w') as f:
            f.truncate(0)
        st.success('CSV files emptied successfully.')
    except Exception as e:
        st.error(f'Error emptying CSV files: {e}')

def remove_duplicates(df):
    df = df.sort_values('ID', ascending=False).drop_duplicates('ID')
    return df

def display_results():
    st.set_page_config(
        page_title="Analysis Result",
        page_icon="🛍️",
        layout="wide"
    )
    st.title("Analysis Result")
    
    if st.sidebar.button("Empty CSV Files"):
        empty_csv_files()

    st.write("#")

    try:
        waiting_times_file = 'waiting_times.csv'
        if os.path.exists(waiting_times_file):
            waiting_df = pd.read_csv(waiting_times_file)
            if waiting_df.empty:
                st.warning('Engagement Times CSV file is empty.')
            else:
                waiting_df = remove_duplicates(waiting_df)
                st.subheader("Waiting Time Estimates")
                st.dataframe(waiting_df, use_container_width=True)
        else:
            st.warning('Engagement Times CSV file not found.')
        
    except EmptyDataError:
        st.warning('Engagement Times CSV file is empty.')

if __name__ == '__main__':
    try:
        display_results()
    except SystemExit:
        pass
