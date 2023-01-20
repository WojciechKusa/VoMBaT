import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import (
    measures_definition,
    calculate_metrics,
    defined_metrics,
    definitions,
    draw_sidebar,
)

e, i, dataset_size = draw_sidebar()

st.title("Comparison of evaluation measures for all levels of recall")

st.write(
    "This page presents 3D plots of possible evaluation measures scores for all recall and TN levels. "
    "Select the measures you want to compare in the sidebar (up to four). "
    "Each measure is plotted in a separate 3D plot. "
    "The x-axis represents the number of TNs, the y-axis represents the estimated recall level, "
    "and the z-axis represents the score of the selected measure. "
    "You can see the definition of each measure below."
)

options = st.multiselect(
    "Select measures",
    (defined_metrics),
    default=["TNR", "WSS", "F05_score", "F3_score"],
    max_selections=4,
)
columns = [x[1] for x in options]  # todo ????

st.write(
    f"### Evaluation measure scores depending on the number of True Negatives (TNs) and estimated recall levels"
)

df_3d = pd.DataFrame()
all_recalls = np.linspace(0.01, 1, 30)

for recall in all_recalls:
    TP = recall * i
    FN = (1 - recall) * i
    metrics = calculate_metrics(dataset_size=dataset_size, e=e, i=i, recall=recall)

    df_3d = df_3d.append(
        pd.DataFrame(metrics),
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
