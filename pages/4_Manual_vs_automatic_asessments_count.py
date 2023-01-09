import copy

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import json

from src.utils import get_dataset_parameters, calculate_metrics

time_per_document = 0.5  # seconds
cost_per_hour = 30
assessments_per_document = 2

with open("data/datasets.json", "r") as f:
    datasets = json.load(f)


# Sidebar
st.sidebar.write("### Dataset parameters")

emp = st.sidebar.empty()
dataset_type = emp.selectbox(
    label="Select a dataset type", options=datasets.keys(),
    key='dataset_picker'
)

if dataset_type == "Custom":
    _dataset_size = st.session_state.dataset_size
    _i_percentage = st.session_state.i_percentage
    _i = int(_dataset_size * _i_percentage / 100)
    _e = _dataset_size - _i
else:
    _dataset_size, _i, _e, _i_percentage = get_dataset_parameters(dataset_type=dataset_type)


def check_dataset_size():
    if _dataset_size != st.session_state.dataset_size:
        st.session_state['dataset_picker'] = 'Custom'


def check_i_percentage():
    if _i_percentage != st.session_state.i_percentage:
        st.session_state['dataset_picker'] = 'Custom'


dataset_size = st.sidebar.slider("Dataset size", 100, 5000, _dataset_size, 50, key='dataset_size', on_change=check_dataset_size)
i_percentage = st.sidebar.slider(
    "Percentage of relevant documents (includes)", 1.0, 99.0, _i_percentage, 1.0, key='i_percentage', on_change=check_i_percentage
)


i = int(dataset_size * i_percentage / 100)
e = dataset_size - i
st.sidebar.write("Number of relevant documents (includes): ", i)
st.sidebar.write("Number of non-relevant documents (excludes): ", e)

st.title("Manual / automatic assessments count")

st.write(
    """
        This page displays the expected number of documents that would be screened manually 
        and automatically, assuming one wants to achieve a specific recall level and the 
        algorithm achieves some specific value of TNR.
         """
)
estimated_recall = st.slider("Desired recall value: ", 1, 100, 95, 1)
estimated_recall /= 100

FN = int(i * (1 - estimated_recall))
TP = i - FN

metrics = calculate_metrics(dataset_size=dataset_size, e=e, i=i, recall=estimated_recall)

df = pd.DataFrame(
    metrics
)


step = max(df.loc[1, "TNR"] - df.loc[0, "TNR"], 0.005)
selected_tnr = st.slider(
    "TNR (nWSS) score obtained by an algorithm: ", 0.0, 0.0, 1.0, step
)

st.write(
    "TPR: ",
    estimated_recall,
    "FNR: ",
    np.around(1 - estimated_recall, decimals=2),
    "FPR: ",
    np.around(1 - selected_tnr, decimals=2),
    "TNR: ",
    selected_tnr,
)


selected_fp = df[
    (df["TNR"] > selected_tnr - 0.002) & (df["TNR"] < selected_tnr + 0.001)
]["FP"].values[0]
selected_tn = df[
    (df["TNR"] > selected_tnr - 0.002) & (df["TNR"] < selected_tnr + 0.001)
]["TN"].values[0]

st.markdown(
    f"| Screened |  |  | Number of |  \n\
| ----------- | ----------- | ----------- | ----------- | \n\
| Manually T | {TP+selected_fp}  | TP | {TP} | \n\
| Manually F | | FP | {selected_fp} | \n\
| Automatically T | {FN+selected_tn} | FN | {FN} | \n\
| Automatically F | | TN | {selected_tn} |"
)
st.markdown("")

x = TP
y = selected_fp
z = FN
w = selected_tn
# draw two bar plots stacking x, y, z and w. And other when x+y and z+w are stacked
st.bar_chart(
    pd.DataFrame(
        {
            "Man. screened includes": [TP],
            "Man. screened excludes": [selected_fp],
            "Autom. screened includes": [FN],
            "Autom. screened excludes": [selected_tn],
        }
    )
)
