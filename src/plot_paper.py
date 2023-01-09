# plots used in the WSS paper
import json

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.utils import get_dataset_parameters

time_per_document = 0.5
cost_per_hour = 15

with open("../data/datasets.json", "r") as f:
    datasets = json.load(f)

st.title("Evaluation metrics for a fixed level of recall")

# Sidebar
st.sidebar.write("### Dataset parameters")
dataset_type = st.sidebar.selectbox(
    label="Pick a dataset type", options=datasets.keys()
)
_dataset_size, _i, _e, _i_percentage = get_dataset_parameters(dataset_type=dataset_type)

dataset_size = st.sidebar.slider("Dataset size", 100, 5000, _dataset_size, 50)
i_percentage = st.sidebar.slider(
    "Percentage of 'positive' documents (includes)", 1.0, 99.0, _i_percentage, 1.0
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

TN = np.array(range(0, e + 1))
FP = e - TN

hours_saved = 2 * TN * time_per_document / 60
cost_saved = hours_saved * cost_per_hour

TPR = TP / i  # recall
FPR = FP / e

nWSS = TN / e  # TNR
WSS = (TN + FN) / dataset_size - (1 - estimated_recall)

accuracy = (TP + TN) / dataset_size
precision = TP / (TP + FP)


def auc(x, y):
    return np.trapz(y, x)


auc_dataset_size = 1000

# tpr which first achieves 0.8 recall and then plateaus
TPR_A = [x / auc_dataset_size for x in list(range(auc_dataset_size))]
TPR_A[0] = 0
TPR_A[1] = 0.04
TPR_A[2] = 0.08
TPR_A[3:] = [0.655 * (1.224**x) for x in TPR_A[3:]]
TPR_A[990:] = [1 for x in TPR_A[990:]]

print(TPR_A)
# tpr which slowly rises linearly and then highly jumps to 0.95 at position 950
TPR_B = [x / auc_dataset_size for x in list(range(auc_dataset_size))]
_model_b_change_position = 340
TPR_B[:_model_b_change_position] = [
    x / 910 for x in list(range(_model_b_change_position))
]
TPR_B[_model_b_change_position:] = [0.997 for x in TPR_B[_model_b_change_position:]]
TPR_B[990:] = [1 for x in TPR_B[990:]]

new_FPR = [x / auc_dataset_size for x in list(range(auc_dataset_size))]
new_TNR = [1 - x for x in new_FPR]

roc_auc_a = auc(new_FPR, TPR_A)
roc_auc_b = auc(new_FPR, TPR_B)

fig, ax = plt.subplots(figsize=(7, 4.5))
plt.plot(
    new_FPR,
    TPR_A,
    color="cornflowerblue",
    lw=1.5,
    label="Model A ROC curve (area = %0.2f)" % roc_auc_a,
)
plt.plot(
    new_FPR,
    TPR_B,
    color="crimson",
    lw=1.5,
    label="Model B ROC curve (area = %0.2f)" % roc_auc_b,
)
plt.plot([0, 1], [0, 1], color="coral", lw=1.5, linestyle="--")
plt.grid(axis="y", alpha=0.5)

# plt.plot([0, 1], [0.8, 0.8], color='gray', lw=1, linestyle='--')
# plt.plot([0, 1], [0.7, 0.7], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.001])
secx = ax.secondary_xaxis("top", functions=(lambda x: x / 1, lambda x: -x / 1))
secx.set_xlabel("True Negative Rate")

# add y label tick at 0.71 and from 0.0 to 1.0 at 0.1 intervals
plt.yticks(
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
# plt.title('Receiver operating characteristic curves')
plt.legend(loc="lower right")
st.pyplot(fig)

roc_auc_a = auc(new_TNR, TPR_A)
roc_auc_b = auc(new_TNR, TPR_B)
fig, ax = plt.subplots(figsize=(7, 4.5))
plt.plot(
    new_TNR,
    TPR_A,
    color="cornflowerblue",
    lw=1.5,
    label="Model A ROC curve (area = %0.2f)" % roc_auc_a,
)
plt.plot(
    new_TNR,
    TPR_B,
    color="crimson",
    lw=1.5,
    label="Model B ROC curve (area = %0.2f)" % roc_auc_b,
)
plt.plot([0, 1], [0, 1], color="coral", lw=1.5, linestyle="--")
plt.grid(axis="y", alpha=0.5)
plt.ylim([0.0, 1.001])
plt.xlim([0.0, 1.0])
plt.legend(loc="lower right")
st.pyplot(fig)

# plt.style.use('ggplot')
# plt.style.use('seaborn-dark-palette')
# plt.style.use('seaborn-whitegrid')
# plt.style.use('seaborn-poster')
# plt.style.use('seaborn-paper')
plt.style.use("seaborn-talk")

# nWSS, work saved over sampling (WSS) and precision plot for standard systematic review
fig, ax = plt.subplots(figsize=(7, 4.5))
plt.plot(TN, nWSS, lw=1.5, color="coral", label="TNR@95%R")
plt.plot(TN, WSS, lw=1.5, color="crimson", label="WSS@95%R")
plt.plot(TN, precision, lw=1.5, label="Precision@95%R")
# plt.xlim([0.0, 1000.0])
# plt.ylim([0.0, 1.0])
plt.xlabel("True Negatives")
plt.ylabel("Evaluation measure value")
# plot horizontal line every 10% of the y axis
# plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# plot horizontal grid
plt.grid(axis="y", alpha=0.5)

plt.legend(loc="upper left")
st.pyplot(fig)
