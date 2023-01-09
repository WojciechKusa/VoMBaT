import copy

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import json

from src.utils import get_dataset_parameters

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

FN = int(i * (1 - estimated_recall))
TP = i - FN

TN = np.array(range(e + 1))
FP = e - TN

hours_saved = 2 * TN * time_per_document / 60
cost_saved = hours_saved * cost_per_hour

TPR = TP / i  # recall
FPR = FP / e

nWSS = TN / e  # TNR
WSS = (TN + FN) / dataset_size - (1 - estimated_recall)

accuracy = (TP + TN) / dataset_size
precision = TP / (TP + FP)
F1_score = 2 * precision * TPR / (precision + TPR)
F05_score = (1 + 0.5**2) * precision * TPR / (0.5**2 * precision + TPR)
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

# reTNR -- like reLU but with TNR for scores==0 when random is better. also normalised
reTNR = copy.deepcopy(nWSS)
for _index_i in range(len(reTNR) - 1, -1, -1):
    if WSS[_index_i] > 0:
        continue
    else:
        reTNR[_index_i] = reTNR[_index_i + 1]
nreTNR = (reTNR - min(reTNR)) / (max(reTNR) - min(reTNR))


df = pd.DataFrame(
    {
        "nWSS": nWSS,
        "WSS": WSS,
        "TN": TN,
        "FN": FN,
        "TP": TP,
        "FP": FP,
        "precision": precision,
        "recall": TPR,
        "F1_score": F1_score,
        "F05_score": F05_score,
        "F3_score": F3_score,
        "FDR": FDR,
        "NPV": NPV,
        "FOR": FOR,
        "accuracy": accuracy,
        "hours_saved": hours_saved,
        "cost_saved": cost_saved,
        "normalisedF1": normalisedF1,
        "normalisedF3": normalisedF3,
        "normalisedF05": normalisedF05,
        "reTNR": reTNR,
        "nreTNR": nreTNR,
    }
)

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

# st.write("### Estimation of time and money savings depending on evaluation measures values.")
st.write(
    "Time spent per document: ",
    time_per_document,
    "minutes, per user. ",
    assessments_per_document,
    " assessments per document.",
)
st.write("Cost per annotator: ", cost_per_hour, "€ per hour.")

sampling_step = np.around(np.max(TN), decimals=-1) / 10
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
