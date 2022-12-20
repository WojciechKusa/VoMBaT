import json
from typing import Tuple

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


measures_definition = r"""
\begin{align}
\mathcal{E} &= FP + TN \\
\mathcal{I} &= TP + FN \\
\text{TP@r\%} &= r \cdot \mathcal{I} \\
\text{FN@r\%} &= (1 - r) \cdot \mathcal{I} \\
WSS@r\% &= \frac{TN + FN}{N} - \left(1 - r\right) \\
nWSS@r\% = TNR@r\% &= \frac{TN}{TN + FP} \\
reTNR@r\% &= \begin{cases} TNR@r\%, & \text{if } \frac{FP@r\%}{\mathcal{E}} < r\% \\ FNR@r\%, & \text{otherwise} \end{cases} \\
nreTNR &= \frac{reTNR - \min(reTNR)}{\max(reTNR) - \min(reTNR)} \\
F_1@r\% &= \frac{2TP}{2TP + FP + FN} \\
F_2@r\% &= \frac{5TP}{5TP + 4FN + FP} \\
F_3@r\% &= \frac{10TP}{10TP + 9FP + FN} \\
F_{0.5}@r\% &= \frac{1.25TP}{1.25TP + 0.25FP + FN} \\ 
normalisedF_1@r\% &= \frac{(r + 1) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot (r \cdot \mathcal{I}+ \mathcal{I} + FP)} \\
normalisedF_{beta}@r\% &= \frac{(r + \beta^2) \cdot \mathcal{I} \cdot TN}{\mathcal{E} \cdot (r \cdot \mathcal{I}+ \beta^2 \cdot \mathcal{I} + FP)} \\ 
PPV = Precision@r\% &= \frac{TP}{TP + FP} \\
FDR@r\% &= \frac{FP}{TP + FP} \\
NPV@r\% &= \frac{TN}{TN + FN} \\
FOR@r\% &= \frac{FN}{TN + FN} \\
Accuracy &= \frac{TP + TN}{TP + TN + FP + FN} 
\end{align}
"""
