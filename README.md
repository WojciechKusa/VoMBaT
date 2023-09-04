# VoMBaT: Visualisation of Evaluation Measure Behaviour in Technology Assisted Reviews

This package serves as basis for the paper: _"VoMBaT: A Tool for Visualising Evaluation Measure Behaviour in High-Recall Search Tasks"_ by Wojciech Kusa, Aldo Lipani, Petr Knoth, Allan Hanbury

[![DOI:10.1145/3539618.3591802](http://img.shields.io/badge/SIGIR_2023-https://doi.org/10.1145/3539618.3591802-1F7CFA.svg)](https://doi.org/10.1145/3539618.3591802) 


High-Recall Information Retrieval (HRIR) tasks, such as Technology-Assisted Review (TAR) used in legal eDiscovery and systematic literature reviews, focus on maximising the retrieval of relevant documents 🔎📑. Traditional evaluation measures consider precision or work saved at fixed recall levels, which can sometimes misrepresent actual system performance, especially when estimating potential savings in time and cost ⏳💰. Introducing **VoMBaT** – a visual analytics tool 🖥️ designed to explore the interplay between evaluation measures and varying recall levels. Our open-source tool provides insights into 18 different evaluation measures, both general and TAR-specific, letting you contrast, compare, and simulate savings in both time and money 🕵️‍📈️️️. Explore the metrics and their potential impacts on your HRIR tasks [here](https://vombat.streamlit.app).


## Installation

Create and activate conda environment:

```bash
$ conda create --name tar_metrics_demo python==3.10.10
$ conda activate tar_metrics_demo
```

Install Python requirements:

```bash
(tar_metrics_demo)$ pip install -r requirements.txt
```

No additional dependencies and data are required.
Datasets' parameters are defined in `data/datasets.json` file.

## Running

Start Streamlit server:

```bash
(tar_metrics_demo)$ streamlit run _🏠_Homepage.py
```

You can now access the app at http://localhost:8501
