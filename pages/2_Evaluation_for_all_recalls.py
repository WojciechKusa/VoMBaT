import copy

import streamlit as st
import numpy as np
import pandas as pd
import json
import plotly.express as px

from src.utils import get_dataset_parameters, measures_definition, calculate_metrics, defined_metrics, definitions

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

# describe the application and all pages:

st.title("Comparison of evaluation measures for all levels of recall")

options = st.multiselect(
    "Select measures",
    (
defined_metrics
    ),
    default=["TNR", "WSS", "F05_score", "F3_score"],
    max_selections=4,
)
columns = [x[1] for x in options]  # todo ????

st.write(
    f"### Evaluation measure scores depending on the number of True Negatives (TNs) and estimated recall levels"
)
st.write("This page presents 3D plots of possible evaluation measures scores for all recall and TN levels. "
         "Select the measures you want to compare in the sidebar (up to four). "
         "Each measure is plotted in a separate 3D plot. "
         "The x-axis represents the number of TNs, the y-axis represents the estimated recall level, "
         "and the z-axis represents the score of the selected measure. "
         "You can see the definition of each measure below.")


df_3d = pd.DataFrame()
all_recalls = np.linspace(0.01, 1, 30)
all_TNs = np.linspace(0, e, 50)

for recall in all_recalls:
    TP = recall * i
    FN = (1 - recall) * i
    metrics = calculate_metrics(dataset_size=dataset_size, e=e, i=i, recall=recall)

    df_3d = df_3d.append(
        pd.DataFrame(
            metrics
        ),
        ignore_index=True,
    )

for measure in options:
    st.latex(r"\begin{align*} " + "\n" + definitions[measure] + "\n" + r"\end{align*}")
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
    st.markdown("----")


with st.expander("Show measures' definitions"):
    st.latex(measures_definition)
