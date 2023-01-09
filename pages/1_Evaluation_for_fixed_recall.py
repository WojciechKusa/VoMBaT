import copy

import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.express as px

from src.utils import get_dataset_parameters, measures_definition, calculate_metrics, defined_metrics

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

st.title("Comparison of evaluation measures for a fixed level of recall")
st.write("Select a level of recall that you want to compare the measures for. "
         "The level of recall is the percentage of relevant documents that are retrieved. "
         "For example, if you select 10%, it assumes that the model retrieved correctly 10% of relevant documents. "
         "The rest of the documents are assumed to be classified non-relevant. "
         "You can see the definition of each measure below.")

estimated_recall = st.slider("Estimated recall: ", 1, 100, 95, 1)
estimated_recall /= 100


metrics = calculate_metrics(dataset_size=dataset_size, e=e, i=i, recall=estimated_recall)
st.write("TPR: ", estimated_recall, "FNR: ", np.around(1 - estimated_recall, decimals=2))


df = pd.DataFrame(
    metrics
)

options = st.multiselect(
    "Select measures: ",
    (
        defined_metrics
    ),
    default=["TNR", "WSS", "Precision", "F05_score", "F3_score"],
)

st.write(
    f"### Evaluation measure scores versus the number of True Negatives (TNs) for {100*estimated_recall:0.0f}% recall"
)

fig = px.line(df, x="TN", y=options,
              width=800,
              height=450,
              )
fig.update_layout(
    xaxis_title=r'TN',
    yaxis_title=r'Evaluation measure score',
    legend_title_text='Measures',
)
st.plotly_chart(fig)


with st.expander("Show measures' definitions"):
    st.latex(measures_definition)
