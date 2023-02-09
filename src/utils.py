import copy
import json
from typing import Tuple

import numpy as np
import streamlit as st

with open("data/datasets.json", "r") as f:
    datasets = json.load(f)


def draw_sidebar() -> tuple[int, int, int]:
    """Draws the sidebar with the dataset selection: number of documents and number of includes.
    Common across all pages.
    :return: tuple: e, i, dataset_size
     - number of excludes
     - number of includes
     - number of documents
    """
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
    st.sidebar.write("Number of relevant documents (includes, $\mathcal{I}$): ", i)
    st.sidebar.write("Number of non-relevant documents (excludes, $\mathcal{E}$): ", e)
    st.sidebar.write("Total number of documents, $\mathcal{N}$: ", dataset_size)

    return e, i, dataset_size


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


definitions = {
    "I": r"\mathcal{I} &= TP + FN ",
    "E": r"\mathcal{E} &= FP + TN ",
    "N": r"\mathcal{N} &= \mathcal{I} + \mathcal{E} ",
    "TP": r"\text{TP@r\%} &= r \cdot \mathcal{I} ",
    "FN": r"\text{FN@r\%} &= (1 - r) \cdot \mathcal{I} ",
    "WSS": r"WSS@r\% &= \frac{TN + FN}{N} - \left(1 - r\right) ",
    "Precision": r"Precision@r\% &= \frac{TP}{TP+FP} ",
    "normalisedPrecision": r"nPrecision@r\% &= \frac{TP \cdot TN}{\mathcal{E} \cdot (TP + FP)} ",
    "Prevalence": r"Prevalence@r\% &= \frac{\mathcal{I}}{\mathcal{N}} ",
    "DFR": r"DFR@r\% &= \frac{Prevalence \cdot Recall}{Precision} ",
    "TNR": r"TNR@r\% = nWSS@r\% &= \frac{TN}{TN + FP} ",
    "reTNR": r"reTNR@r\% &= \begin{cases} TNR@r\%, & \text{if } \frac{FP@r\%}{\mathcal{E}} < r\% \\ FNR@r\%, & \text{otherwise} \end{cases} ",
    "nreTNR": r"nreTNR &= \frac{reTNR - \min(reTNR)}{\max(reTNR) - \min(reTNR)} ",
    "F1_score": r"F_1@r\% &= \frac{2TP}{2TP + FP + FN} ",
    "F2_score": r"F_2@r\% &= \frac{5TP}{5TP + 4FN + FP} ",
    "F3_score": r"F_3@r\% &= \frac{10TP}{10TP + 9FP + FN} ",
    "F05_score": r"F_{0.5}@r\% &= \frac{1.25TP}{1.25TP + 0.25FP + FN} ",
    "normalisedF1": r"normalisedF_1@r\% &= \frac{(r + 1) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot (r \cdot \mathcal{I}+ \mathcal{I} + FP)} ",
    "normalisedF3": r"normalisedF_{3}@r\% &= \frac{(r + 9) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot ((r + 9 )\cdot \mathcal{I} + FP)} ",
    "normalisedF05": r"normalisedF_{0.5}@r\% &= \frac{(r + 0.25) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot ((r + 0.25) \cdot \mathcal{I} + FP)} ",
    "normalisedFB": r"normalisedF_{beta}@r\% &= \frac{(r + \beta^2) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot (r \cdot \mathcal{I}+ \beta^2 \cdot \mathcal{I} + FP)} ",
    "PPV": r"PPV@r\% = Precision@r\% &= \frac{TP}{TP + FP} ",
    "FDR": r"FDR@r\% &= \frac{FP}{TP + FP} ",
    "NPV": r"NPV@r\% &= \frac{TN}{TN + FN} ",
    "FOR": r"FOR@r\% &= \frac{FN}{TN + FN} ",
    "LR+": r"LR+@r\% &= \frac{TPR}{FPR}",
    "LR-": r"LR-@r\% &= \frac{FNR}{TNR} ",
    "DOR": r"DOR@r\% &= \frac{LR+@r\%}{LR-@r\%} ",
    "Accuracy": r"Accuracy@r\% &= \frac{TP + TN}{TP + TN + FP + FN}",
    "Balanced accuracy": r"BA@r\% &= \frac{TPR + TNR}{2} ",
    "MCC": r"MCC@r\% &= \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}} ",
}
measures_definition = r"\begin{align}"
for definition in definitions.values():
    measures_definition += definition + r"\\"
measures_definition += r"\end{align}"


def calculate_metrics(i, e, recall, dataset_size):
    metrics = {}
    FN = int(i * (1 - recall))
    TP = i - FN

    TN = np.array(range(e + 1))
    FP = e - TN

    FPR = FP / e
    FNR = FN / i
    TNR = TN / e  # nWSS
    WSS = (TN + FN) / dataset_size - (1 - recall)

    accuracy = (TP + TN) / dataset_size
    precision = TP / (TP + FP)
    n_precision = (recall * i * TN) / (e * (TP + FP))
    prevalence = i / dataset_size
    F1_score = 2 * precision * recall / (precision + recall)
    F05_score = (1 + 0.5**2) * precision * recall / (0.5**2 * precision + recall)
    F3_score = 10 * precision * recall / (9 * precision + recall)
    FDR = 1 - precision
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    DFR = prevalence * recall / precision
    # https://www.gibsondunn.com/wp-content/uploads/documents/publications/Evans-Metrics-that-Matter-Inside-Counsel-1.2015.pdf

    NPV = TN / (TN + FN)
    FOR = 1 - NPV

    LRminus = FNR / TNR
    LRplus = recall / FPR
    DOR = LRplus / LRminus
    balanced_accuracy = (recall + TNR) / 2

    normalisedF1 = ((recall + 1) * i * TN) / (e * (recall * i + i + FP))
    normalisedF3 = ((recall + 9) * i * TN) / (e * (recall * i + 9 * i + FP))
    normalisedF05 = ((recall + 0.25) * i * TN) / (e * (recall * i + 0.25 * i + FP))

    # reTNR -- like reLU but with TNR for scores==0 when random is better. also normalised
    reTNR = copy.deepcopy(TNR)
    for _index_i in range(len(reTNR) - 1, -1, -1):
        if WSS[_index_i] > 0:
            continue
        else:
            reTNR[_index_i] = reTNR[_index_i + 1]
    nreTNR = (reTNR - min(reTNR)) / (max(reTNR) - min(reTNR))

    # return all variables as dict
    metrics["TP"] = TP
    metrics["TN"] = TN
    metrics["FP"] = FP
    metrics["FN"] = FN
    metrics["FPR"] = FPR
    metrics["TNR"] = TNR
    metrics["WSS"] = WSS
    metrics["Prevalence"] = prevalence
    metrics["DFR"] = DFR
    metrics["Accuracy"] = accuracy
    metrics["Balanced accuracy"] = balanced_accuracy
    metrics["Precision"] = precision
    metrics["normalisedPrecision"] = n_precision
    metrics["F1_score"] = F1_score
    metrics["F05_score"] = F05_score
    metrics["F3_score"] = F3_score
    metrics["FDR"] = FDR
    metrics["NPV"] = NPV
    metrics["FOR"] = FOR
    metrics["LR-"] = LRminus
    metrics["LR+"] = LRplus
    metrics["DOR"] = DOR
    metrics["normalisedF1"] = normalisedF1
    metrics["normalisedF3"] = normalisedF3
    metrics["normalisedF05"] = normalisedF05
    metrics["reTNR"] = reTNR
    metrics["nreTNR"] = nreTNR
    metrics["recall"] = recall
    metrics["MCC"] = MCC

    return metrics


defined_metrics = [
    "TNR",
    "WSS",
    "Precision",
    "normalisedPrecision",
    "Prevalence",
    "DFR",
    "F1_score",
    "F05_score",
    "F3_score",
    "FDR",
    "NPV",
    "FOR",
    "LR-",
    "LR+",
    "DOR",
    "MCC",
    "Accuracy",
    "Balanced accuracy",
    "normalisedF1",
    "normalisedF3",
    "normalisedF05",
    "reTNR",
    "nreTNR",
]
