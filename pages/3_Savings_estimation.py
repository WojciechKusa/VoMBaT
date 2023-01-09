import json

import numpy as np
import pandas as pd
import streamlit as st

from src.utils import get_dataset_parameters, calculate_metrics, defined_metrics

with open("data/datasets.json", "r") as f:
    datasets = json.load(f)

# Sidebar
st.sidebar.write("### Dataset parameters")

emp = st.sidebar.empty()
dataset_type = emp.selectbox(
    label="Select a dataset type", options=datasets.keys(), key="dataset_picker"
)

if dataset_type == "Custom":
    _dataset_size = st.session_state.dataset_size
    _i_percentage = st.session_state.i_percentage
    _i = int(_dataset_size * _i_percentage / 100)
    _e = _dataset_size - _i
else:
    _dataset_size, _i, _e, _i_percentage = get_dataset_parameters(
        dataset_type=dataset_type
    )


def check_dataset_size():
    if _dataset_size != st.session_state.dataset_size:
        st.session_state["dataset_picker"] = "Custom"


def check_i_percentage():
    if _i_percentage != st.session_state.i_percentage:
        st.session_state["dataset_picker"] = "Custom"


dataset_size = st.sidebar.slider(
    "Dataset size",
    100,
    5000,
    _dataset_size,
    50,
    key="dataset_size",
    on_change=check_dataset_size,
)
i_percentage = st.sidebar.slider(
    "Percentage of relevant documents (includes)",
    1.0,
    99.0,
    _i_percentage,
    1.0,
    key="i_percentage",
    on_change=check_i_percentage,
)

i = int(dataset_size * i_percentage / 100)
e = dataset_size - i
st.sidebar.write("Number of relevant documents (includes): ", i)
st.sidebar.write("Number of non-relevant documents (excludes): ", e)

st.title("Estimation of time and money savings depending on evaluation measures values")

st.write(
    "This page presents the time and money savings that can be achieved depending on the value of evaluation measures. "
    "The goal is to determine the minimum value of the evaluation measures that can be accepted in order to reduce the "
    "manual screening time and the cost of the evaluation. "
    "Average time spent per document, number of manual assessments per document and cost of manual annotators can be "
    "adjusted using the sliders below. "
    "The dataset size, the percentage of relevant documents and the minimum satisfiable recall can also be set."
)

st.write(
    "When screening the dataset manually, every document needs to be assessed. "
    "Savings can come when the automatic assessment is good enough to avoid manual assessment of some documents. "
    "This is equal to removing True Negatives (TN). "
    "Depending on how many TNs the model can discard, the higher the savings are."
)

estimated_recall = st.slider("Required recall: ", 1, 100, 95, 1)
estimated_recall /= 100

cost_per_hour = st.number_input(
    "Cost of manual annotator per hour [€]:", 10, 100, 30, 5
)
time_per_document = st.number_input(
    "Average time spent per document per annotator in seconds:", 5, 180, 30, 5
)
assessments_per_document = st.number_input(
    "Number of annotators assessing each document:", 1, 5, 2
)

hours_per_document = time_per_document / 60 / 60

metrics = calculate_metrics(
    dataset_size=dataset_size, e=e, i=i, recall=estimated_recall
)

df = pd.DataFrame(metrics)

df["Hours saved"] = assessments_per_document * df["TN"] * hours_per_document
df["Cost saved"] = df["Hours saved"] * cost_per_hour

st.metric(
    label="Cost of manually annotating the whole dataset",
    value=f"{cost_per_hour * hours_per_document * assessments_per_document * dataset_size :.0f} €",
)

st.markdown("---")

options = st.multiselect(
    "Select measures",
    (defined_metrics),
    default=["TNR", "WSS", "Precision", "F05_score", "F3_score"],
    max_selections=6,
)

sampling_step = np.around(np.max(df["TN"]), decimals=-1) / 10
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
        "Hours saved",
        "Cost saved",
    ]
]

hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """
st.markdown(hide_table_row_index, unsafe_allow_html=True)

st.table(sampled_df.style.format({"Hours saved": "{:.2f}", "Cost saved": "{:.0f}"}))
