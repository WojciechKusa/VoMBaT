import numpy as np
import pandas as pd
import streamlit as st

from src.utils import (
    calculate_metrics,
    defined_metrics,
    draw_sidebar,
)

e, i, dataset_size = draw_sidebar()

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

col1, col2 = st.columns(2)
col1.metric(
    label="Cost of manually annotating the whole dataset",
    value=f"{cost_per_hour * hours_per_document * assessments_per_document * dataset_size :.0f} €",
)
col2.metric(
    label="Time needed for manually annotating the whole dataset",
    value=f"{hours_per_document * assessments_per_document * dataset_size :.0f} hours",
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
