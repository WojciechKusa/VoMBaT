import streamlit as st


st.title("Comparison of evaluation measures for Technology-Assisted Reviews")


st.write(
    """
    This app compares different evaluation measures for a Technology-Assisted Review (TAR) system.
    On the left, you can select four subpages:
    1. Comparison of evaluation measures for a fixed recall level
    2. Comparison of evaluation measures for all recall levels [takes a while to load]
    3. Simulations of savings in time and money for different datasets depending on the model quality
    4. Simulations of manual and automatic assessments for different datasets depending on the model quality
    
    TAR system can be constructed both as a classifier or as a ranker.
    In the case of a classifier, we first need to select a threshold for the probability of relevance for which \
    the model obtains a a fixed recall level.
    In the case of a ranker, we can directly select a fixed recall level.
    
    ________
    
    For each of these pages a set of predefined systematic review datasets was prepared. 
    You can select one of these datasets and then select the evaluation measures you want to compare.
    You can also define custom dataset by selecting it's size and the percentage of relevant documents (includes). 

    There are two types of predefined datasets:
    - three synthetic datasets showing extreme options for the distribution of relevant documents (includes) in the dataset
    - fifteen real datasets from systematic reviews in the field of medicine introduced in \
    *Cohen, A. M., Hersh, W. R., Peterson, K., & Yen, P. Y. (2006). \
    Reducing workload in systematic review preparation using automated citation classification. \
    Journal of the American Medical Informatics Association, 13(2), 206-219.*
    
    More information about the datasets can be found in the last section of the sidebar.
    """
)
