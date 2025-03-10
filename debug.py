import streamlit as st
import os
st.title("Emergency Debug")
st.write("Files in directory:", os.listdir("."))
st.write("Data dir exists:", os.path.exists("data"))
if os.path.exists("data"): st.write("Data files:", os.listdir("data"))
