import copy
import json
from typing import Tuple

import numpy as np

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


definitions = {
    'E': r"\mathcal{E} &= FP + TN ",
    'I': r"\mathcal{I} &= TP + FN ",
    'TP': r"\text{TP@r\%} &= r \cdot \mathcal{I} ",
    'FN': r"\text{FN@r\%} &= (1 - r) \cdot \mathcal{I} ",
    'WSS': r"WSS@r\% &= \frac{TN + FN}{N} - \left(1 - r\right) ",
    'TNR': r"TNR@r\% = nWSS@r\% &= \frac{TN}{TN + FP} ",
    'reTNR': r"reTNR@r\% &= \begin{cases} TNR@r\%, & \text{if } \frac{FP@r\%}{\mathcal{E}} < r\% \\ FNR@r\%, & \text{otherwise} \end{cases} ",
    'nreTNR': r"nreTNR &= \frac{reTNR - \min(reTNR)}{\max(reTNR) - \min(reTNR)} ",
    'F1_score': r"F_1@r\% &= \frac{2TP}{2TP + FP + FN} ",
    'F2_score': r"F_2@r\% &= \frac{5TP}{5TP + 4FN + FP} ",
    'F3_score': r"F_3@r\% &= \frac{10TP}{10TP + 9FP + FN} ",
    'F05_score': r"F_{0.5}@r\% &= \frac{1.25TP}{1.25TP + 0.25FP + FN} ",
    'normalisedF_1': r"normalisedF_1@r\% &= \frac{(r + 1) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot (r \cdot \mathcal{I}+ \mathcal{I} + FP)} ",
    'normalisedF_B': r"normalisedF_{beta}@r\% &= \frac{(r + \beta^2) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot (r \cdot \mathcal{I}+ \beta^2 \cdot \mathcal{I} + FP)} ",
    'PPV': r"PPV = Precision@r\% &= \frac{TP}{TP + FP} ",
    'FDR': r"FDR@r\% &= \frac{FP}{TP + FP} ",
    'NPV': r"NPV@r\% &= \frac{TN}{TN + FN} ",
    'FOR': r"FOR@r\% &= \frac{FN}{TN + FN} ",
    'Accuracy': r"Accuracy &= \frac{TP + TN}{TP + TN + FP + FN}",
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
    TNR = TN / e  # nWSS
    WSS = (TN + FN) / dataset_size - (1 - recall)

    accuracy = (TP + TN) / dataset_size
    precision = TP / (TP + FP)
    F1_score = 2 * precision * recall / (precision + recall)
    F05_score = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)
    F3_score = 10 * precision * recall / (9 * precision + recall)
    FDR = 1 - precision

    NPV = TN / (TN + FN)
    FOR = 1 - NPV

    normalisedF1 = ((recall + 1) * i * TN) / (e * (recall * i + i + FP))
    normalisedF3 = ((recall + 9) * i * TN) / (
        e * (recall * i + 9 * i + FP)
    )
    normalisedF05 = ((recall + 0.25) * i * TN) / (
        e * (recall * i + 0.25 * i + FP)
    )

    # reTNR -- like reLU but with TNR for scores==0 when random is better. also normalised
    reTNR = copy.deepcopy(TNR)
    print(len(reTNR) - 1, reTNR[len(reTNR) - 1], WSS[len(reTNR) - 1])
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
    metrics["accuracy"] = accuracy
    metrics["precision"] = precision
    metrics["F1_score"] = F1_score
    metrics["F05_score"] = F05_score
    metrics["F3_score"] = F3_score
    metrics["FDR"] = FDR
    metrics["NPV"] = NPV
    metrics["FOR"] = FOR
    metrics["normalisedF1"] = normalisedF1
    metrics["normalisedF3"] = normalisedF3
    metrics["normalisedF05"] = normalisedF05
    metrics["reTNR"] = reTNR
    metrics["nreTNR"] = nreTNR
    metrics["recall"] = recall

    return metrics


defined_metrics = [
    "TNR",
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
]
