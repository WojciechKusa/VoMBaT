import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils import calculate_metrics, draw_sidebar

time_per_document = 0.5  # seconds
cost_per_hour = 30
assessments_per_document = 2

e, i, dataset_size = draw_sidebar()

st.title("Manual / automatic assessments count")

st.write(
    """       
        When one fixes the recall level, the number of relevant documents (includes) that would be screened 
        manually and automatically is fixed. 
        Relevant documents included automatically are equal to TP, whereas includes left for a manual review are equal 
        to FN. 
        The number of irrelevant documents (excludes) that would be screened manually and automatically depends on the
        models quality (TNR). 
        The higher the TNR score, the more irrelevant documents are excluded automatically (TN). 
        The remaining irrelevant documents need to be reviewed manually (FP).
         
        This page displays the expected number of documents that would be screened manually 
        and automatically, assuming one wants to achieve a specific recall level.
        Values are presented as stacked barplots for eleven different values of TNR.
    """
)
estimated_recall = st.slider("Desired recall value: ", 1, 100, 95, 1)
estimated_recall /= 100

FN = int(i * (1 - estimated_recall))
TP = i - FN

metrics = calculate_metrics(
    dataset_size=dataset_size, e=e, i=i, recall=estimated_recall
)

df = pd.DataFrame(metrics)

out_df = pd.DataFrame()
for selected_tnr in range(0, 101, 10):
    selected_tnr /= 100.0

    selected_fp = df[
        (df["TNR"] > selected_tnr - 0.002) & (df["TNR"] < selected_tnr + 0.002)
    ]["FP"].values[0]
    selected_tn = df[
        (df["TNR"] > selected_tnr - 0.002) & (df["TNR"] < selected_tnr + 0.002)
    ]["TN"].values[0]

    values_dict = {
        "Includes left for manual review": FN,
        "Automatically included": TP,
        "Excludes left for manual review": selected_fp,
        "Automatically excluded": selected_tn,
    }

    for key, value in values_dict.items():
        out_df = out_df.append(
            {
                "type": key,
                "count": value,
                "TNR": selected_tnr,
            },
            ignore_index=True,
        )

fig = px.bar(
    out_df,
    x="TNR",
    y="count",
    color="type",
    color_discrete_sequence=px.colors.qualitative.D3,
)
fig.update_layout(
    xaxis_title=r"TNR",
    yaxis_title=r"Document count",
    legend_title_text="Type of documents",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(tickmode="array", tickvals=out_df["TNR"].unique()),
)
st.plotly_chart(fig)
