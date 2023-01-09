import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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

st.title("Manual / automatic assessments count")

st.write(
    """       
        When one fixes the recall level, the number of relevant documents (includes) that would be screened 
        manually and automatically is fixed. 
        Relevant documents included automatically are equal to TP, whereas includes left for a manual review are equal 
        to FN. 
        The number of irrelevant documents (excludes) that would be screened manually and automatically depends on the
        models quality (TNR). 
        The higher the TNR score, the more irrelevant documents are excluded automatically (TN). 
        The remaining irrelevant documents need to be reviewed manually (FP).
         
        This page displays the expected number of documents that would be screened manually 
        and automatically, assuming one wants to achieve a specific recall level.
        Values are presented as stacked barplots for eleven different values of TNR.
    """
)
estimated_recall = st.slider("Desired recall value: ", 1, 100, 95, 1)
estimated_recall /= 100

FN = int(i * (1 - estimated_recall))
TP = i - FN

metrics = calculate_metrics(
    dataset_size=dataset_size, e=e, i=i, recall=estimated_recall
)

df = pd.DataFrame(metrics)


out_df = pd.DataFrame()
for selected_tnr in range(0, 101, 10):
    selected_tnr /= 100.0

    selected_fp = df[
        (df["TNR"] > selected_tnr - 0.002) & (df["TNR"] < selected_tnr + 0.002)
    ]["FP"].values[0]
    selected_tn = df[
        (df["TNR"] > selected_tnr - 0.002) & (df["TNR"] < selected_tnr + 0.002)
    ]["TN"].values[0]

    values_dict = {
        "Includes left for manual review": FN,
        "Automatically included": TP,
        "Excludes left for manual review": selected_fp,
        "Automatically excluded": selected_tn,
    }

    for key, value in values_dict.items():
        out_df = out_df.append(
            {
                "type": key,
                "count": value,
                "TNR": selected_tnr,
            },
            ignore_index=True,
        )


fig = px.bar(
    out_df,
    x="TNR",
    y="count",
    color="type",
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    xaxis_title=r"TNR",
    yaxis_title=r"Document count",
    legend_title_text="Type of documents",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(tickmode="array", tickvals=out_df["TNR"].unique()),
)
st.plotly_chart(fig)
