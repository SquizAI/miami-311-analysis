import streamlit as st
import os
import traceback

st.title("Ultra Minimal App")
st.write("This is the minimal version of the app")

st.write("Files in directory:")
st.write(os.listdir("."))

st.write("Data directory exists:", os.path.exists("data"))
if os.path.exists("data"):
    st.write("Data files:", os.listdir("data"))

try:
    import pandas as pd
    if os.path.exists("data/auto-mpg.csv"):
        df = pd.read_csv("data/auto-mpg.csv")
        st.write(f"Successfully loaded {len(df)} rows from auto-mpg.csv")
        st.dataframe(df.head())
except Exception as e:
    st.error(f"Error: {e}")
    st.code(traceback.format_exc())
