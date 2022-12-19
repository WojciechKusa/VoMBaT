import copy

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import json

from src.utils import get_dataset_parameters

with open("data/datasets.json", "r") as f:
    datasets = json.load(f)

# Sidebar
st.sidebar.write("### Dataset parameters")
dataset_type = st.sidebar.selectbox(
    label="Select a dataset type", options=datasets.keys()
)
_dataset_size, _i, _e, _i_percentage = get_dataset_parameters(dataset_type=dataset_type)

dataset_size = st.sidebar.slider("Dataset size", 100, 5000, _dataset_size, 50)
i_percentage = st.sidebar.slider(
    "Percentage of relevant documents (includes)", 1.0, 99.0, _i_percentage, 1.0
)


i = int(dataset_size * i_percentage / 100)
e = dataset_size - i
st.sidebar.write("Number of relevant documents (includes): ", i)
st.sidebar.write("Number of non-relevant documents (excludes): ", e)

# describe the application and all pages:

st.title("Comparison of evaluation measures for all levels of recall")

options = st.multiselect(
    "Select measures",
    (
        "nWSS",
        "WSS",
        "precision",
        "F1_score",
        "F05_score",
        "F3_score",
        "FDR",
        "NPV",
        "FOR",
        "accuracy",
        "normalisedF1",
        "normalisedF3",
        "normalisedF05",
        "reTNR",
        "nreTNR",
    ),
    default=["nWSS", "WSS", "precision", "F05_score", "F3_score"],
)
columns = [x[1] for x in options]  # todo ????

st.write(
    f"### Evaluation measure scores versus the number of True Negatives (TNs) for all possible levels of recall"
)
st.write("3D plot of F1, F0.5, WSS and TNR for different recall and TN levels")


# 3D plot of F1, F3 and WSS for different recall and TN levels
# X axis = recall
# Y axis = TN
# Z axis = F1, F3 or WSS
import plotly.express as px

df_3d = pd.DataFrame()
all_recalls = np.linspace(0.1, 1, 100)
all_TNs = np.linspace(0, e, 100)
all_F1s = []
all_F3s = []
all_WSSs = []
for recall in all_recalls:
    TP = recall * i
    FN = (1 - recall) * i
    for TN in all_TNs:
        FP = e - TN
        precision = TP / (TP + FP)
        TNR = TN / e
        df_3d = df_3d.append(
            {
                "recall": recall,
                "TN": TN,
                "F1": 2 * precision * recall / (precision + recall),
                "F05": (1 + 0.5**2) * precision * recall / (0.5**2 * precision + recall),
                "F3": 10 * precision * recall / (9 * precision + recall),
                "WSS": (TN + FN) / dataset_size - (1 - recall),
                "TNR": TNR,
            },
            ignore_index=True,
        )

# add streamlit new page

fig = px.scatter_3d(
    df_3d,
    x="TN",
    y="recall",
    z="F1",
    color="F1",
    opacity=0.7,
    width=800,
    height=600,
)
fig.update_layout(
    scene=dict(
        xaxis_title="TN",
        yaxis_title="recall",
        zaxis_title="F1",
    ),
)
st.plotly_chart(fig)

fig = px.scatter_3d(
    df_3d,
    x="TN",
    y="recall",
    z="F05",
    color="recall",
    opacity=0.7,
    width=800,
    height=600,
)
fig.update_layout(
    scene=dict(
        xaxis_title="TN",
        yaxis_title="recall",
        zaxis_title="F05",
    ),
)
st.plotly_chart(fig)

fig = px.scatter_3d(
    df_3d,
    x="TN",
    y="recall",
    z="WSS",
    color="recall",
    opacity=0.7,
    width=800,
    height=600,
)
fig.update_layout(
    scene=dict(
        xaxis_title="TN",
        yaxis_title="recall",
        zaxis_title="WSS",
    ),
)
st.plotly_chart(fig)

fig = px.scatter_3d(
    df_3d,
    x="TN",
    y="recall",
    z="TNR",
    color="recall",
    opacity=0.7,
    width=800,
    height=600,
)
fig.update_layout(
    scene=dict(
        xaxis_title="TN",
        yaxis_title="recall",
        zaxis_title="TNR",
    ),
)
st.plotly_chart(fig)
