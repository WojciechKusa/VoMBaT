import streamlit as st


st.title("Comparison of evaluation measures for technology-assisted reviews")


st.write(
    """
    This app compares different evaluation measures for a technology-assisted review (TAR) system.
    On the left, you can select two subpages:
    1. Comparison of evaluation measures for a fixed recall level
    2. Comparison of evaluation measures for all recall levels [takes a while to load]
    3. Estimations of savings in time and money for different datasets depending on the model quality
    4. Count of manual and automatic assessments for different datasets depending on the model quality
    
    For each of these pages a set of predefined datasets was prepared. 
    You can select one of these datasets and then select the evaluation measures you want to compare.
    """
)
