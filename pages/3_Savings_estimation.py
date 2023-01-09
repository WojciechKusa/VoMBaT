import copy

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import json

from src.utils import get_dataset_parameters, calculate_metrics, defined_metrics

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

st.title("Estimation of time and money savings depending on evaluation measures values")

estimated_recall = st.slider("Estimated recall: ", 1, 100, 95, 1)


estimated_recall /= 100

metrics = calculate_metrics(dataset_size=dataset_size, e=e, i=i, recall=estimated_recall)

df = pd.DataFrame(
    metrics
)

df["hours_saved"] = 2 * df['TN'] * time_per_document / 60
df["cost_saved"] = df['hours_saved'] * cost_per_hour



options = st.multiselect(
    "Select measures",
    (
        defined_metrics
    ),
    default=["TNR", "WSS", "precision", "F05_score", "F3_score"],
)

# st.write("### Estimation of time and money savings depending on evaluation measures values.")
st.write(
    "Time spent per document: ",
    time_per_document,
    "minutes, per user. ",
    assessments_per_document,
    " assessments per document.",
)
st.write("Cost per annotator: ", cost_per_hour, "€ per hour.")

sampling_step = np.around(np.max(df['TN']), decimals=-1) / 10
sampled_df = df[df["TN"] % sampling_step == 0]
sampled_df.reset_index(inplace=True, drop=True)
sampled_df = sampled_df[
    [
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    + options
    + [
        "hours_saved",
        "cost_saved",
    ]
]

# pd.io.formats.style.Styler
st.dataframe(sampled_df.style.hide(axis="index"), height=422)
