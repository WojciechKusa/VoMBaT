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
        "TNR",
        "WSS",
        "precision",
        "F1",
        "F05",
        "F3",
        "FDR",
        "NPV",
        "FOR",
        "accuracy",
        "normalisedF1",
        "normalisedF3",
        "normalisedF05",
        # "reTNR",
        # "nreTNR",
    ),
    default=["TNR", "WSS", "F05", "F3"],
    max_selections=4,
)
columns = [x[1] for x in options]  # todo ????

st.write(
    f"### Evaluation measure scores versus the number of True Negatives (TNs) for all possible levels of recall"
)
st.write("3D plot of evaluation measures for all recall and TN levels")


# 3D plot of F1, F3 and WSS for different recall and TN levels
# X axis = recall
# Y axis = TN
# Z axis = F1, F3 or WSS
import plotly.express as px

df_3d = pd.DataFrame()
all_recalls = np.linspace(0.1, 1, 30)
all_TNs = np.linspace(0, e, 50)
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
        TPR = recall
        estimated_recall = recall/100

        accuracy = (TP + TN) / dataset_size
        # precision = TP / (TP + FP)
        F1_score = 2 * precision * TPR / (precision + TPR)
        F05_score = (1 + 0.5 ** 2) * precision * TPR / (0.5 ** 2 * precision + TPR)
        F3_score = 10 * precision * TPR / (9 * precision + TPR)
        FDR = 1 - precision

        NPV = TN / (TN + FN)
        FOR = 1 - NPV

        # st.write("TPR: ", TPR, "FNR: ", np.around(1 - TPR, decimals=2))

        normalisedF1 = ((estimated_recall + 1) * i * TN) / (e * (estimated_recall * i + i + FP))
        normalisedF3 = ((estimated_recall + 9) * i * TN) / (
                e * (estimated_recall * i + 9 * i + FP)
        )
        normalisedF05 = ((estimated_recall + 0.25) * i * TN) / (
                e * (estimated_recall * i + 0.25 * i + FP)
        )

        df_3d = df_3d.append(
            {
                "recall": recall,
                "TN": TN,
                "F1": 2 * precision * recall / (precision + recall),
                "F05": (1 + 0.5**2)
                * precision
                * recall
                / (0.5**2 * precision + recall),
                "F3": 10 * precision * recall / (9 * precision + recall),
                "WSS": (TN + FN) / dataset_size - (1 - recall),
                "TNR": TNR,
                "normalisedF1": normalisedF1,
                "normalisedF3": normalisedF3,
                "normalisedF05": normalisedF05,
                "precision": precision,
                "accuracy": accuracy,
                "FDR": FDR,
                "FOR": FOR,
                "NPV": NPV,
            },
            ignore_index=True,
        )

for measure in options:
    fig = px.scatter_3d(
        df_3d,
        x="TN",
        y="recall",
        z=measure,
        color=measure,
        opacity=0.7,
        width=800,
        height=600,
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="TN",
            yaxis_title="recall",
            zaxis_title=measure,
        ),
    )
    st.plotly_chart(fig)
