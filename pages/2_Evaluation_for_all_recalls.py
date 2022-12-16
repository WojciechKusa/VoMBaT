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
    for TN in all_TNs:
        TP = recall * i
        FN = (1 - recall) * i
        precision = TP / (TP + TN)
        TPR = TP / (TP + FN)
        FP = TN * (1 - TPR)
        TNR = TN / (TN + FP)
        # df_3d = pd.concat(
        #     [
        #         df_3d,
        #         pd.DataFrame(
        #             {
        #                 "recall": recall,
        #                 "TN": TN,
        #                 "F1": 2 * precision * TPR / (precision + TPR),
        #                 "F3": 10 * precision * TPR / (9 * precision + TPR),
        #                 "WSS": WSS,
        #             },
        #             index=[0],
        #         ),
        #     ],
        #     ignore_index=True,
        #     axis=0,
        # )
        df_3d = df_3d.append(
            {
                "recall": recall,
                "TN": TN,
                "F1": 2 * precision * TPR / (precision + TPR),
                "F05": (1 + 0.5**2) * precision * TPR / (0.5**2 * precision + TPR),
                "F3": 10 * precision * TPR / (9 * precision + TPR),
                "WSS": (TN + FN) / dataset_size - (1 - recall),
                "TNR": TNR,
            },
            ignore_index=True,
        )

# add streamlit new page

st.write("### 3D plot of F1, F3, WSS and TNR for different recall and TN levels")
fig = px.scatter_3d(
    df_3d,
    x="TN",
    y="recall",
    z="F1",
    color="recall",
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



# 2D plot of F1, F3 and WSS for different recall and TN levels
# X axis = recall
# Y axis = F1, F3 or WSS
# Z axis = TN
# fig = go.Figure()
# for TN in np.linspace(0, e, 100):
#     TP = recall * i
#     FN = (1 - recall) * i
#     precision = TP / (TP + TN)
#     TPR = TP / (TP + FN)
#     FP = TN * (1 - TPR)
#     TNR = TN / (TN + FP)
#     fig.add_trace(
#         go.Scatter(
#             x=all_recalls,
#             y=2 * precision * TPR / (precision + TPR),
#             mode="lines",
#             name="TN = " + str(TN),
#         )
#     )
# fig.update_layout(
#     xaxis_title="recall",
#     yaxis_title="F1",
# )
# st.plotly_chart(fig)