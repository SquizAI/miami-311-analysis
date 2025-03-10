import streamlit as st
import traceback
import os, sys

st.title("Intermediate Debug")

try:
    import data_loader
    st.success("Data loader imported")

    df = data_loader.create_simulated_data(n_records=100)
    st.success(f"Created simulated data with {len(df)} rows")
    st.dataframe(df.head(3))

    df_processed = data_loader.preprocess_311_data(df)
    st.success("Preprocessed data successfully")

    import analysis
    st.success("Analysis module imported")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.code(traceback.format_exc())
