import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    measures_definition,
    calculate_metrics,
    defined_metrics,
    draw_sidebar,
)

e, i, dataset_size = draw_sidebar()

st.title("Comparison of evaluation measures for a fixed level of recall")
st.write(
    "Select a level of recall that you want to compare the measures for. "
    "The level of recall is the percentage of relevant documents that are retrieved. "
    "For example, if you select 10%, it assumes that the model retrieved correctly 10% of relevant documents. "
    "The rest of the documents are assumed to be classified non-relevant. "
    "You can see the definition of each measure below."
)

estimated_recall = st.slider("Desired recall: ", 1, 100, 95, 1)
estimated_recall /= 100

metrics = calculate_metrics(
    dataset_size=dataset_size, e=e, i=i, recall=estimated_recall
)
st.write(
    "TPR: ", estimated_recall, "FNR: ", np.around(1 - estimated_recall, decimals=2)
)

df = pd.DataFrame(metrics)

options = st.multiselect(
    "Select measures: ",
    (defined_metrics),
    default=["TNR", "WSS", "Precision", "F05_score", "F3_score"],
)

st.write(
    f"### Evaluation measure scores versus the number of True Negatives (TNs) for {100 * estimated_recall:0.0f}% recall"
)

fig = px.line(
    df,
    x="TN",
    y=options,
    width=800,
    height=450,
)
fig.update_layout(
    xaxis_title=r"TN",
    yaxis_title=r"Evaluation measure score",
    legend_title_text="Measures",
)
st.plotly_chart(fig)

with st.expander("Show definitions of all measures"):
    st.latex(measures_definition)
