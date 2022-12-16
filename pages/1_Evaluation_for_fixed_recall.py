import copy

import streamlit as st
import numpy as np
import pandas as pd
from typing import Tuple
import json

time_per_document = 0.5  # seconds
cost_per_hour = 30
assessments_per_document = 2

with open("data/datasets.json", "r") as f:
    datasets = json.load(f)


def get_dataset_parameters(dataset_type: str) -> Tuple[int, int, int, int]:
    i_percentage = (
        100 * datasets[dataset_type]["includes"] / datasets[dataset_type]["size"]
    )
    return (
        datasets[dataset_type]["size"],
        datasets[dataset_type]["includes"],
        datasets[dataset_type]["excludes"],
        i_percentage,
    )


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
st.sidebar.markdown("***")

st.sidebar.write("### Expectation on recall")
estimated_recall = st.sidebar.slider("Estimated recall", 1, 100, 95, 1)


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

st.sidebar.write("TPR: ", TPR, "FNR: ", np.around(1 - TPR, decimals=2))

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

st.title("Comparison of evaluation measures for a fixed level of recall")

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
    f"### Evaluation measure scores versus the number of True Negatives (TNs) for {100*estimated_recall:0.0f}% recall"
)
st.line_chart(df, x="TN", y=options)

step = max(df.loc[1, "nWSS"] - df.loc[0, "nWSS"], 0.005)
selected_tnr = st.slider(
    "Select your desired value of TNR (nWSS):", 0.0, 0.0, 1.0, step
)
selected_fp = df[
    (df["nWSS"] > selected_tnr - 0.002) & (df["nWSS"] < selected_tnr + 0.01)
]["FP"].values[0]
selected_tn = df[
    (df["nWSS"] > selected_tnr - 0.002) & (df["nWSS"] < selected_tnr + 0.01)
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

latex_code = r"""
\begin{align}
F &= FP + TN \\
T &= TP + FN \\
\text{TP@r\%} &= r \cdot T \\
\text{FN@r\%} &= (1 - r) \cdot T \\
WSS@r\% &= \frac{TN + FN}{N} - \left(1 - r\right) \\
nWSS@r\% = TNR@r\% &= \frac{TN}{TN + FP} \\
reTNR &= \max\left\{TNR@r\%, TNR@r\% + \epsilon\right\} \\
nreTNR &= \frac{reTNR - \min(reTNR)}{\max(reTNR) - \min(reTNR)} \\
F_1@r\% &= \frac{2TP}{2TP + FP + FN} \\
F_2@r\% &= \frac{5TP}{5TP + 4FN + FP} \\
F_3@r\% &= \frac{10TP}{10TP + 9FP + FN} \\
F_{0.5}@r\% &= \frac{1.25TP}{1.25TP + 0.25FP + FN} \\ 
normalisedF_1@r\% &= \frac{(r + 1) \cdot T \cdot TN}{F \cdot (r \cdot T+ T + FP)} \\
normalisedF_{beta}@r\% &= \frac{(r + \beta^2) \cdot T \cdot TN}{F \cdot (r \cdot T+ \beta^2 \cdot T + FP)} \\ 
PPV = Precision@r\% &= \frac{TP}{TP + FP} \\
FDR@r\% &= \frac{FP}{TP + FP} \\
NPV@r\% &= \frac{TN}{TN + FN} \\
FOR@r\% &= \frac{FN}{TN + FN} \\
Accuracy &= \frac{TP + TN}{TP + TN + FP + FN} 
\end{align}
"""

with st.expander("Show measures' definitions"):
    st.latex(latex_code)

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
        # "WSS",
        # "nWSS",
        # "precision",
        # "F3_score",
    ] + options +
    [
        "hours_saved",
        "cost_saved",
    ]
]

# pd.io.formats.style.Styler
st.dataframe(sampled_df.style.hide(axis="index"), height=422)


# F1, F3 and work saved over sampling (WSS) for standard systematic reviews
# Precision, F0.5 and F1 when considering rapid reviews


# print ROC curve
# st.write("### Plot of TPR and FPR (ROC curve)")
# df = pd.DataFrame({"TPR": TPR, "FPR": FPR})
# st.line_chart(df, x="FPR", y="TPR")
