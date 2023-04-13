import streamlit as st

st.title(
    "VoMBaT🐻 - Visualisation of Evaluation Measure Behaviour for High-Recall Search Tasks"
)

st.write(
    """
    This app compares different evaluation measures for High-Recall Search Tasks.
    On the left, you can select four subpages:
    1. Comparison of evaluation measures for a fixed recall level
    2. Comparison of evaluation measures for all recall levels 
    3. Simulations of savings in time and money for different datasets depending on the model quality
    4. Simulations of manual and automatic assessments for different datasets depending on the model quality
    5. Custom evaluation measures comparison
    
    Examples of High-Recall Search Tasks are:
    - citation screening for systematic literature reviews
    - eDiscovery (electronic discovery) for legal cases 
    
    These systems can be constructed both as a classifier or as a ranker.
    In the case of a classifier, we first need to select a threshold for the probability of relevance for which \
    the model obtains a a fixed recall level.
    In the case of a ranker, we can directly select a fixed recall level.
    
    ________
    
    For each of these pages a set of predefined systematic review datasets' parameters was prepared. 
    You can select one of these datasets and then select the evaluation measures you want to compare.
    You can also define custom dataset by selecting it's size and the percentage of relevant documents (includes). 

    There are two types of predefined datasets' parameters:
    - Three synthetic examples of dataset parameters showing extreme options for the distribution of relevant \
    documents ($\mathcal{I}$) in the dataset: 
        - balanced, 
        - heavily unbalanced towards positive class (example of a very good search query), 
        - heavily unbalanced towards negative class (very typical in systematic reviews).
    - Fifteen datasets which use the $N$ and $\mathcal{I}$ values from systematic reviews \
    in the field of medicine introduced in \
    *Cohen, A. M., Hersh, W. R., Peterson, K., & Yen, P. Y. (2006). \
    Reducing workload in systematic review preparation using automated citation classification. \
    Journal of the American Medical Informatics Association, 13(2), 206-219.*
    
    """
)
